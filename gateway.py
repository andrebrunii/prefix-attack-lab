"""
AI Security Gateway locale compatibile con le API OpenAI (/v1/chat/completions).

Inoltra le richieste a Ollama (http://127.0.0.1:11434/v1/...) applicando:
- configurazione baseline / hardened / guarded
- input guardrail & output guardrail
- audit log completo in logs/gateway_audit.jsonl

Uso:
    python gateway.py --port 8080 --mode guarded
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from audit import append_audit_event, utc_now_iso
from config import LOGS_DIR, OLLAMA_BASE_URL, SYSTEM_PROMPTS
from guardrails import check_input, check_output, wrap_untrusted_content

GATEWAY_LOG = LOGS_DIR / "gateway_audit.jsonl"


def make_openai_blocked_response(model: str, message: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-guard-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": message},
                "finish_reason": "content_filter",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def create_handler(mode: str, log_file: Path):
    class SecurityGatewayHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, body: dict[str, Any]) -> None:
            raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"status": "ok", "mode": mode, "upstream": OLLAMA_BASE_URL})
                return
            if self.path == "/v1/models":
                try:
                    with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/v1/models", timeout=30) as resp:
                        data = json.load(resp)
                    self._send_json(200, data)
                except Exception as exc:
                    self._send_json(502, {"error": f"Upstream Ollama non raggiungibile: {exc}"})
                return
            self._send_json(404, {"error": "Endpoint non supportato dal gateway"})

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._send_json(404, {"error": "Usa /v1/chat/completions"})
                return

            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json(400, {"error": "JSON non valido"})
                return

            # Forza stream=False nel gateway didattico per poter ispezionare l'output completo
            payload["stream"] = False
            model = payload.get("model", "unknown")
            messages = list(payload.get("messages", []))
            guard_enabled = mode == "guarded"
            hardened = mode in ("hardened", "guarded")

            # Estrae il testo degli ultimi messaggi utente per l'input guardrail
            user_texts = [m.get("content", "") for m in messages if m.get("role") == "user"]
            combined_user = "\n".join(str(t) for t in user_texts)

            # Rileva eventuale assistant prefill (messaggio finale con role=assistant)
            trailing_prefill = ""
            if messages and messages[-1].get("role") == "assistant":
                trailing_prefill = str(messages[-1].get("content", ""))

            event: dict[str, Any] = {
                "timestamp": utc_now_iso(),
                "source": "openai_gateway",
                "model": model,
                "mode": mode,
                "guardrails_enabled": guard_enabled,
                "input_blocked": False,
                "output_blocked": False,
                "guardrail_reasons": [],
                "guardrail_overhead_ms": 0.0,
            }

            if guard_enabled:
                in_dec = check_input(combined_user, prefill_text=trailing_prefill)
                event["guardrail_overhead_ms"] = round(
                    event["guardrail_overhead_ms"] + in_dec.latency_ms, 3
                )
                if in_dec.blocked:
                    event["input_blocked"] = True
                    event["guardrail_reasons"].extend(in_dec.reasons)
                    event["request"] = payload
                    event["visible_response"] = "[INPUT BLOCKED]"
                    append_audit_event(event, log_file)
                    self._send_json(200, make_openai_blocked_response(model, "[INPUT BLOCKED]"))
                    return

            if hardened:
                # Anteponi il system prompt hardened e incapsula eventuali blocchi [DOCUMENTO]
                patched_messages = [{"role": "system", "content": SYSTEM_PROMPTS[mode]}]
                for m in messages:
                    if m.get("role") == "user":
                        patched_messages.append(
                            {"role": "user", "content": wrap_untrusted_content(str(m.get("content", "")))}
                        )
                    else:
                        patched_messages.append(m)
                payload["messages"] = patched_messages

            event["request"] = payload

            req = urllib.request.Request(
                f"{OLLAMA_BASE_URL}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    result = json.load(resp)
                event["latency_seconds"] = round(time.perf_counter() - start, 3)
            except (urllib.error.URLError, urllib.error.HTTPError) as exc:
                event["error"] = str(exc)
                append_audit_event(event, log_file)
                self._send_json(502, {"error": f"Errore Ollama upstream: {exc}"})
                return

            event["raw_response"] = result
            answer = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            if guard_enabled:
                out_dec = check_output(answer)
                event["guardrail_overhead_ms"] = round(
                    event["guardrail_overhead_ms"] + out_dec.latency_ms, 3
                )
                if out_dec.blocked:
                    event["output_blocked"] = True
                    event["guardrail_reasons"].extend(out_dec.reasons)
                    event["visible_response"] = "[OUTPUT BLOCKED]"
                    append_audit_event(event, log_file)
                    self._send_json(200, make_openai_blocked_response(model, "[OUTPUT BLOCKED]"))
                    return

            event["visible_response"] = answer
            append_audit_event(event, log_file)
            self._send_json(200, result)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[gateway] {self.address_string()} - {format % args}")

    return SecurityGatewayHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Security Gateway compatibile OpenAI API")
    parser.add_argument("--host", default="127.0.0.1", help="Host di ascolto")
    parser.add_argument("--port", type=int, default=8080, help="Porta di ascolto")
    parser.add_argument(
        "--mode",
        choices=["baseline", "hardened", "guarded"],
        default="guarded",
        help="Modalità di sicurezza del gateway",
    )
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), create_handler(args.mode, GATEWAY_LOG))
    print(f"AI Security Gateway attivo su http://{args.host}:{args.port}")
    print(f"Modalità: {args.mode} | Upstream Ollama: {OLLAMA_BASE_URL} | Log: {GATEWAY_LOG}")
    print("Premi Ctrl+C per fermare il server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nGateway arrestato.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
