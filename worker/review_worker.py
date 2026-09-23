#!/usr/bin/env python3
"""review_worker.py — optional LLM review backend for the banned-words module.

The Rust module spawns this ONLY when `llm_review` is enabled in config.json.
Protocol:
  stdin   JSONL requests  {"text": "..."}
  stdout  JSONL responses {"risk": 0-1, "label": "safe"|"unsafe", "error": ...}
  stderr  logs + a final READY line (module can wait on it if it wants)

Engines (all optional; the module degrades to word-list-only when unavailable):
  deberta     microsoft/deberta-v3-small (open) — true 0-1 sigmoid risk score.
  llama-guard meta-llama/Llama-Guard-3-1B (HF-gated, needs HF_TOKEN) — verdict
              generation mapped to 1.0 (unsafe) / 0.0 (safe). Best-effort on
              CPU (slow); the module's 2s review timeout may pass it through.
  auto        use deberta; fall back to llama-guard only if a token exists.
"""

import argparse
import json
import os
import sys


def load_model(engine):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    if engine == "llama-guard":
        model_id = "meta-llama/Llama-Guard-3-1B"
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("llama-guard is HF-gated: set HF_TOKEN")
        tok = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
        model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=token)
        return tok, model, "llama-guard"

    # deberta is the default + the fast 0-1 scorer.
    model_id = "microsoft/deberta-v3-small"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=1)
    return tok, model, "deberta"


def score_deberta(tok, model, text):
    import torch

    enc = tok(text, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    risk = float(torch.sigmoid(out.logits[0][0]))
    return round(risk, 4), "unsafe" if risk > 0.5 else "safe"


def score_llama_guard(tok, model, text):
    import torch

    prompt = (
        "Task: classify the message as safe or unsafe.\n\n"
        "User message: " + text + "\n\nAnswer:"
    )
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=16, do_sample=False)
    verdict = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    unsafe = verdict.strip().lower().startswith("unsafe")
    return (1.0, "unsafe") if unsafe else (0.0, "safe")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="deberta", help="deberta | llama-guard | auto | mock")
    ap.add_argument("--mock-risk", type=float, default=0.9,
                    help="risk returned by the mock engine (protocol testing without models)")
    args = ap.parse_args()

    engine = args.engine
    if engine == "auto":
        engine = "llama-guard" if os.environ.get("HF_TOKEN") else "deberta"

    if engine == "mock":
        # Protocol smoke-test engine: no transformers/torch needed.
        print(f"READY engine=mock", file=sys.stderr, flush=True)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                print(json.dumps({"risk": args.mock_risk,
                                  "label": "unsafe" if args.mock_risk > 0.5 else "safe"}),
                      flush=True)
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({"error": str(exc)}), flush=True)
        return

    try:
        tok, model, actual = load_model(engine)
        model.eval()
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"fatal": str(exc)}), file=sys.stderr)
        print(json.dumps({"error": f"model load failed: {exc}"}), flush=True)
        sys.exit(2)

    print(f"READY engine={actual}", file=sys.stderr, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            text = req.get("text", "")
            if actual == "llama-guard":
                risk, label = score_llama_guard(tok, model, text)
            else:
                risk, label = score_deberta(tok, model, text)
            print(json.dumps({"risk": risk, "label": label}), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()