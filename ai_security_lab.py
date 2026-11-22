"""
CLI interattiva del laboratorio AI Security (Prompt Injection, Prefix Attacks & Guardrails).

Uso:
    python ai_security_lab.py
    python ai_security_lab.py --model llama3.2 --mode guarded
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from audit import append_audit_event, compute_metrics, load_audit_events, utc_now_iso
from config import (
    DEFAULT_MODEL,
    DEFAULT_SENTINEL,
    EXPERIMENTS_FILE,
    LOGS_DIR,
    SYSTEM_PROMPTS,
    guess_template_family,
)
from guardrails import check_input, check_output, wrap_untrusted_content
from ollama_client import OllamaError, chat, generate_raw_prefill, list_models, show_model


def load_experiments() -> list[dict[str, Any]]:
    if not EXPERIMENTS_FILE.exists():
        return []
    return json.loads(EXPERIMENTS_FILE.read_text(encoding="utf-8"))


def build_initial_history(mode: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPTS[mode]}]


def run_chat_turn(
    model: str,
    mode: str,
    history: list[dict[str, str]],
    user_text: str,
    log_file: Path,
    metadata: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Esegue un turno /api/chat applicando la configurazione scelta e registrando l'audit JSONL."""
    guard_enabled = mode == "guarded"
    hardened = mode in ("hardened", "guarded")
    effective_user = wrap_untrusted_content(user_text) if hardened else user_text

    messages = history + [{"role": "user", "content": effective_user}]
    event: dict[str, Any] = {
        "timestamp": utc_now_iso(),
        "endpoint": "/api/chat",
        "model": model,
        "mode": mode,
        "guardrails_enabled": guard_enabled,
        "raw_user_input": user_text,
        "request": {"model": model, "messages": messages},
        "input_blocked": False,
        "output_blocked": False,
        "guardrail_reasons": [],
        "guardrail_overhead_ms": 0.0,
    }
    if metadata:
        event.update(metadata)

    # 1. Input Guardrail
    if guard_enabled:
        in_decision = check_input(user_text)
        event["guardrail_overhead_ms"] = round(
            event["guardrail_overhead_ms"] + in_decision.latency_ms, 3
        )
        if in_decision.blocked:
            event["input_blocked"] = True
            event["guardrail_reasons"].extend(in_decision.reasons)
            event["visible_response"] = "[INPUT BLOCKED]"
            event["raw_output_violated"] = False
            event["attack_succeeded"] = False
            append_audit_event(event, log_file)
            return "[INPUT BLOCKED]", event

    # 2. Chiamata a Ollama
    try:
        payload, result, latency = chat(model=model, messages=messages)
        event["request"] = payload
        event["latency_seconds"] = latency
    except OllamaError as exc:
        event["error"] = str(exc)
        append_audit_event(event, log_file)
        return f"[ERRORE OLLAMA] {exc}", event

    answer = result.get("message", {}).get("content", "")
    forbidden = (metadata or {}).get("forbidden_markers") or [DEFAULT_SENTINEL]

    # Valutazione violazione sulla risposta grezza (utile per le metriche)
    raw_violated = any(m.lower() in answer.lower() for m in forbidden if m)
    event["raw_output_violated"] = raw_violated

    # 3. Output Guardrail
    visible = answer
    if guard_enabled:
        out_decision = check_output(answer, forbidden_markers=forbidden)
        event["guardrail_overhead_ms"] = round(
            event["guardrail_overhead_ms"] + out_decision.latency_ms, 3
        )
        if out_decision.blocked:
            event["output_blocked"] = True
            event["guardrail_reasons"].extend(out_decision.reasons)
            visible = "[OUTPUT BLOCKED]"

    event["raw_response"] = result
    event["visible_response"] = visible
    event["prompt_eval_count"] = result.get("prompt_eval_count")
    event["eval_count"] = result.get("eval_count")
    event["attack_succeeded"] = (
        bool((metadata or {}).get("is_attack"))
        and raw_violated
        and not event["output_blocked"]
    )

    append_audit_event(event, log_file)
    history.extend(
        [
            {"role": "user", "content": effective_user},
            {"role": "assistant", "content": visible},
        ]
    )
    return visible, event


def run_prefill_turn(
    model: str,
    mode: str,
    user_text: str,
    assistant_prefill: str,
    log_file: Path,
    template_family: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Esegue un esperimento di Assistant Prefill via /api/generate (raw=True)."""
    guard_enabled = mode == "guarded"
    hardened = mode in ("hardened", "guarded")
    effective_user = wrap_untrusted_content(user_text) if hardened else user_text
    family = template_family or guess_template_family(model)

    event: dict[str, Any] = {
        "timestamp": utc_now_iso(),
        "endpoint": "/api/generate (raw=true)",
        "model": model,
        "mode": mode,
        "template_family": family,
        "guardrails_enabled": guard_enabled,
        "raw_user_input": user_text,
        "assistant_prefill": assistant_prefill,
        "input_blocked": False,
        "output_blocked": False,
        "guardrail_reasons": [],
        "guardrail_overhead_ms": 0.0,
    }
    if metadata:
        event.update(metadata)

    if guard_enabled:
        in_decision = check_input(user_text, prefill_text=assistant_prefill)
        event["guardrail_overhead_ms"] = round(
            event["guardrail_overhead_ms"] + in_decision.latency_ms, 3
        )
        if in_decision.blocked:
            event["input_blocked"] = True
            event["guardrail_reasons"].extend(in_decision.reasons)
            event["visible_response"] = "[INPUT BLOCKED]"
            event["raw_output_violated"] = False
            event["attack_succeeded"] = False
            append_audit_event(event, log_file)
            return "[INPUT BLOCKED]", event

    try:
        payload, result, latency = generate_raw_prefill(
            model=model,
            system_prompt=SYSTEM_PROMPTS[mode],
            user_prompt=effective_user,
            assistant_prefill=assistant_prefill,
            template_family=family,
        )
        event["request"] = payload
        event["latency_seconds"] = latency
    except OllamaError as exc:
        event["error"] = str(exc)
        append_audit_event(event, log_file)
        return f"[ERRORE OLLAMA] {exc}", event

    continuation = result.get("response", "")
    full_answer = assistant_prefill + continuation
    forbidden = (metadata or {}).get("forbidden_markers") or [DEFAULT_SENTINEL]
    raw_violated = any(m.lower() in full_answer.lower() for m in forbidden if m)
    event["raw_output_violated"] = raw_violated

    visible = full_answer
    if guard_enabled:
        out_decision = check_output(full_answer, forbidden_markers=forbidden)
        event["guardrail_overhead_ms"] = round(
            event["guardrail_overhead_ms"] + out_decision.latency_ms, 3
        )
        if out_decision.blocked:
            event["output_blocked"] = True
            event["guardrail_reasons"].extend(out_decision.reasons)
            visible = "[OUTPUT BLOCKED]"

    event["raw_response"] = result
    event["visible_response"] = visible
    event["prompt_eval_count"] = result.get("prompt_eval_count")
    event["eval_count"] = result.get("eval_count")
    event["attack_succeeded"] = (
        bool((metadata or {}).get("is_attack"))
        and raw_violated
        and not event["output_blocked"]
    )
    append_audit_event(event, log_file)
    return visible, event


def print_help() -> None:
    print(
        "\nComandi disponibili:\n"
        "  /mode <baseline|hardened|guarded>  Cambia configurazione di sicurezza\n"
        "  /guard on | /guard off             Scorciatoia per guarded / baseline\n"
        "  /model <nome>                      Cambia modello Ollama attivo\n"
        "  /models                            Elenca i modelli installati in Ollama\n"
        "  /template                          Mostra il TEMPLATE del modello attivo (/api/show)\n"
        "  /history                           Ispeziona la history completa (system/user/assistant)\n"
        "  /reset                             Azzera la conversazione\n"
        "  /list                              Elenca gli esperimenti pronti in experiments/payloads.json\n"
        "  /run <id>                          Esegue un esperimento predefinito\n"
        "  /prefill <testo>                   Imposta un assistant prefill per il prossimo prompt\n"
        "  /metrics                           Mostra le metriche calcolate da logs/llm_audit.jsonl\n"
        "  /help                              Mostra questo aiuto\n"
        "  /exit                              Esce dal laboratorio\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Security Lab — Prefix Attack & Prompt Injection")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modello Ollama da usare")
    parser.add_argument(
        "--mode",
        choices=["baseline", "hardened", "guarded"],
        default="baseline",
        help="Modalità iniziale di sicurezza",
    )
    parser.add_argument(
        "--log-file",
        default=str(LOGS_DIR / "llm_audit.jsonl"),
        help="Percorso file JSONL per l'audit log",
    )
    args = parser.parse_args()

    model = args.model
    mode = args.mode
    log_file = Path(args.log_file)
    history = build_initial_history(mode)
    pending_prefill: str | None = None
    experiments = {e["id"]: e for e in load_experiments()}

    print("=" * 72)
    print("  AI SECURITY LAB — Prompt Injection, Prefix Attacks & Guardrails")
    print("=" * 72)
    print(f"Modello attivo : {model}")
    print(f"Modalità       : {mode}")
    print(f"Audit log      : {log_file}")
    print("Digita /help per vedere tutti i comandi.\n")

    while True:
        try:
            prompt_tag = f"[{model} | {mode}]"
            if pending_prefill is not None:
                prompt_tag += " [PREFILL ARMED]"
            user = input(f"{prompt_tag} user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nUscita dal laboratorio.")
            break

        if not user:
            continue

        if user == "/exit":
            break
        if user == "/help":
            print_help()
            continue
        if user == "/reset":
            history = build_initial_history(mode)
            pending_prefill = None
            print("Conversazione azzerata.")
            continue
        if user in ("/guard on", "/guard off"):
            mode = "guarded" if user.endswith("on") else "baseline"
            history = build_initial_history(mode)
            print(f"Modalità impostata a: {mode} (conversazione azzerata)")
            continue
        if user.startswith("/mode "):
            new_mode = user.split(maxsplit=1)[1].strip().lower()
            if new_mode not in SYSTEM_PROMPTS:
                print("Modalità non valida. Usa: baseline, hardened, guarded.")
            else:
                mode = new_mode
                history = build_initial_history(mode)
                print(f"Modalità impostata a: {mode} (conversazione azzerata)")
            continue
        if user == "/models":
            try:
                models = list_models()
                print("Modelli Ollama disponibili:", ", ".join(models) if models else "(nessuno)")
            except OllamaError as exc:
                print("Errore:", exc)
            continue
        if user.startswith("/model "):
            model = user.split(maxsplit=1)[1].strip()
            history = build_initial_history(mode)
            print(f"Modello attivo: {model} (conversazione azzerata)")
            continue
        if user == "/template":
            try:
                info = show_model(model)
                print("\n--- TEMPLATE MODELLO ---")
                print(info.get("template", "(template non disponibile)"))
                print("------------------------\n")
            except OllamaError as exc:
                print("Errore:", exc)
            continue
        if user == "/history":
            print(json.dumps(history, indent=2, ensure_ascii=False))
            continue
        if user == "/list":
            for exp_id, exp in experiments.items():
                print(f"  {exp_id:16s} [{exp['category']}] {exp['name']}")
            continue
        if user.startswith("/prefill "):
            pending_prefill = user.split(maxsplit=1)[1]
            print(f"Assistant prefill armato: {pending_prefill!r}")
            print("Il prossimo messaggio verrà inviato via /api/generate (raw=True).")
            continue
        if user == "/metrics":
            events = load_audit_events(log_file)
            metrics = compute_metrics(events)
            print(json.dumps(metrics, indent=2, ensure_ascii=False))
            continue
        if user.startswith("/run "):
            exp_id = user.split(maxsplit=1)[1].strip()
            exp = experiments.get(exp_id)
            if not exp:
                print(f"Esperimento '{exp_id}' non trovato. Usa /list.")
                continue
            print(f"\n>>> Esecuzione {exp['id']}: {exp['name']}")
            print(f"Comportamento atteso: {exp['expected_behavior']}")
            meta = {
                "experiment_id": exp["id"],
                "category": exp["category"],
                "is_attack": exp["is_attack"],
                "forbidden_markers": exp.get("forbidden_markers", []),
            }
            if exp.get("endpoint") == "raw_prefill":
                visible, event = run_prefill_turn(
                    model=model,
                    mode=mode,
                    user_text=exp["prompt"],
                    assistant_prefill=exp.get("assistant_prefill", ""),
                    log_file=log_file,
                    metadata=meta,
                )
            else:
                temp_history = build_initial_history(mode)
                visible, event = run_chat_turn(
                    model=model,
                    mode=mode,
                    history=temp_history,
                    user_text=exp["prompt"],
                    log_file=log_file,
                    metadata=meta,
                )
            print("\nassistant>", visible)
            print(
                f"[diagnostica] blocked(in={event['input_blocked']}, out={event['output_blocked']}) | "
                f"reasons={event['guardrail_reasons']} | latency={event.get('latency_seconds')}s | "
                f"in_tok={event.get('prompt_eval_count')} out_tok={event.get('eval_count')}\n"
            )
            continue

        # Invio normale o prefill armato
        if pending_prefill is not None:
            visible, event = run_prefill_turn(
                model=model,
                mode=mode,
                user_text=user,
                assistant_prefill=pending_prefill,
                log_file=log_file,
            )
            pending_prefill = None
        else:
            visible, event = run_chat_turn(
                model=model,
                mode=mode,
                history=history,
                user_text=user,
                log_file=log_file,
            )

        print("\nassistant>", visible)
        print(
            f"[diagnostica] blocked(in={event['input_blocked']}, out={event['output_blocked']}) | "
            f"reasons={event['guardrail_reasons']} | latency={event.get('latency_seconds')}s | "
            f"in_tok={event.get('prompt_eval_count')} out_tok={event.get('eval_count')}\n"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Errore fatale: {exc}", file=sys.stderr)
        sys.exit(1)
