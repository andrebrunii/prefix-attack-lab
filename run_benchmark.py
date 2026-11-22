"""
Benchmark automatico per confrontare baseline, hardened e guarded
su tutti i payload in experiments/payloads.json.

Uso:
    python run_benchmark.py --model llama3.2
    python run_benchmark.py --models llama3.2 qwen2.5:7b --modes baseline hardened guarded
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_security_lab import build_initial_history, load_experiments, run_chat_turn, run_prefill_turn
from audit import compute_metrics
from config import DEFAULT_MODEL, LOGS_DIR, REPORTS_DIR


def run_suite(models: list[str], modes: list[str]) -> dict[str, Any]:
    experiments = load_experiments()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"benchmark_{timestamp}.jsonl"
    summary: dict[str, Any] = {
        "timestamp": timestamp,
        "log_file": str(log_file),
        "results": {},
    }

    for model in models:
        summary["results"][model] = {}
        for mode in modes:
            print(f"\n=== Modello: {model} | Modalità: {mode} ===")
            mode_events: list[dict[str, Any]] = []
            for exp in experiments:
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
                    history = build_initial_history(mode)
                    visible, event = run_chat_turn(
                        model=model,
                        mode=mode,
                        history=history,
                        user_text=exp["prompt"],
                        log_file=log_file,
                        metadata=meta,
                    )
                mode_events.append(event)
                status = (
                    "BLOCKED"
                    if event["input_blocked"] or event["output_blocked"]
                    else ("ATTACK_OK" if event.get("attack_succeeded") else "SAFE/PASS")
                )
                print(f"  [{status:9s}] {exp['id']:16s} -> {visible[:70].replace(chr(10), ' ')}")

            metrics = compute_metrics(mode_events)
            summary["results"][model][mode] = metrics

    return summary


def format_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        f"# Benchmark Report — {summary['timestamp']}",
        "",
        f"Audit JSONL: `{summary['log_file']}`",
        "",
        "| Modello | Modalità | ASR (%) | Detection (%) | FPR (%) | Output Violation (%) | Guard Overhead (ms) | Avg Latency (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, mode_map in summary["results"].items():
        for mode, m in mode_map.items():
            lines.append(
                f"| {model} | {mode} | {m['attack_success_rate_pct']} | "
                f"{m['detection_rate_pct']} | {m['false_positive_rate_pct']} | "
                f"{m['output_violation_rate_pct']} | {m['avg_guardrail_overhead_ms']} | "
                f"{m['avg_model_latency_s']} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Esegue il benchmark automatico del Prefix Attack Lab")
    parser.add_argument("--model", help="Singolo modello Ollama da testare")
    parser.add_argument("--models", nargs="+", help="Lista di modelli Ollama da confrontare")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["baseline", "hardened", "guarded"],
        choices=["baseline", "hardened", "guarded"],
        help="Modalità da confrontare",
    )
    args = parser.parse_args()

    models = args.models or ([args.model] if args.model else [DEFAULT_MODEL])
    summary = run_suite(models, args.modes)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = summary["timestamp"]
    json_path = REPORTS_DIR / f"report_{ts}.json"
    md_path = REPORTS_DIR / f"report_{ts}.md"

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md_text = format_markdown_report(summary)
    md_path.write_text(md_text, encoding="utf-8")

    print("\n" + md_text)
    print(f"Report salvati in:\n  - {md_path}\n  - {json_path}")


if __name__ == "__main__":
    main()
