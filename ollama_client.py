"""Client HTTP senza dipendenze esterne per interagire con Ollama locale."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from config import DEFAULT_OPTIONS, DEFAULT_TIMEOUT, OLLAMA_BASE_URL, PREFILL_TEMPLATES, guess_template_family


class OllamaError(RuntimeError):
    """Errore nella comunicazione con l'istanza locale di Ollama."""


def _post_json(url: str, payload: dict[str, Any], timeout: int = DEFAULT_TIMEOUT) -> tuple[dict[str, Any], float]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise OllamaError(f"HTTP {exc.code} da {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"Impossibile connettersi a Ollama su {OLLAMA_BASE_URL}. "
            f"Verifica che Ollama sia avviato. Dettaglio: {exc.reason}"
        ) from exc
    latency = round(time.perf_counter() - start, 3)
    return data, latency


def _get_json(url: str, timeout: int = 30) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"Impossibile connettersi a Ollama su {OLLAMA_BASE_URL}. Dettaglio: {exc.reason}"
        ) from exc


def list_models() -> list[str]:
    """Restituisce i nomi dei modelli installati in Ollama (/api/tags)."""
    data = _get_json(f"{OLLAMA_BASE_URL}/api/tags")
    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]


def show_model(model: str) -> dict[str, Any]:
    """Restituisce metadati, template e Modelfile del modello (/api/show)."""
    data, _ = _post_json(f"{OLLAMA_BASE_URL}/api/show", {"name": model}, timeout=30)
    return data


def chat(
    model: str,
    messages: list[dict[str, str]],
    options: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    """Invia una richiesta a /api/chat e restituisce (payload, raw_response, latency_s)."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options if options is not None else dict(DEFAULT_OPTIONS),
    }
    result, latency = _post_json(f"{OLLAMA_BASE_URL}/api/chat", payload, timeout=timeout)
    return payload, result, latency


def generate_raw_prefill(
    model: str,
    system_prompt: str,
    user_prompt: str,
    assistant_prefill: str,
    template_family: str | None = None,
    options: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    """
    Esegue un vero esperimento di Assistant Prefill usando /api/generate con raw=True.
    Restituisce (payload, raw_response, latency_s).
    """
    family = template_family or guess_template_family(model)
    tpl = PREFILL_TEMPLATES.get(family, PREFILL_TEMPLATES["plain"])
    raw_prompt = tpl["prompt"].format(
        system=system_prompt,
        user=user_prompt,
        prefill=assistant_prefill,
    )
    merged_options = dict(DEFAULT_OPTIONS)
    if options:
        merged_options.update(options)
    merged_options.setdefault("stop", tpl["stop"])

    payload = {
        "model": model,
        "prompt": raw_prompt,
        "raw": True,
        "stream": False,
        "options": merged_options,
    }
    result, latency = _post_json(f"{OLLAMA_BASE_URL}/api/generate", payload, timeout=timeout)
    return payload, result, latency
