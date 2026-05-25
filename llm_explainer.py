"""
llm_explainer.py
----------------
Optional narration layer. Takes the REAL output of the local scikit-learn
models and produces a plain-language explanation. All numbers come from local
code; the LLM is used only to phrase them. If no API key is set or the network
call fails, a deterministic template explanation is returned so the feature
never breaks offline.
"""

from __future__ import annotations

from typing import Any

import requests


SYSTEM_PROMPT = (
    "You are explaining the output of a locally-trained scikit-learn model. "
    "Interpret these REAL numbers for a non-technical reader. "
    "Do not invent any numbers; only explain the ones given. "
    "State one honest caveat."
)


def explain_with_grok(
    model_output: dict[str, Any],
    api_key: str = "",
    base_url: str = "https://api.x.ai/v1/chat/completions",
    model: str = "grok-2-latest",
    timeout: int = 30,
) -> str:
    """
    Narrate `model_output` in plain language.

    `model_output` must include the real numbers from the local model so the
    LLM cannot fabricate predictions. Falls back to a deterministic template
    if `api_key` is empty or the request fails.
    """
    if not api_key:
        return _template_explanation(model_output)

    try:
        response = requests.post(
            base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _render_numbers(model_output)},
                ],
                "temperature": 0.2,
                "max_tokens": 600,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"].strip()
    except Exception:
        return _template_explanation(model_output)


def _render_numbers(model_output: dict[str, Any]) -> str:
    lines = [
        f"Task type: {model_output.get('task_type', 'unknown')}",
        f"Decision Tree accuracy: {model_output.get('dt_accuracy', 'n/a')}",
        f"Random Forest accuracy: {model_output.get('rf_accuracy', 'n/a')}",
        f"Concrete prediction: {model_output.get('prediction', 'n/a')}",
    ]
    confidence = model_output.get("prediction_confidence")
    if confidence is not None:
        lines.append(f"Prediction confidence: {confidence}")
    facts = model_output.get("facts_used")
    if facts:
        lines.append(f"User-supplied facts: {facts}")
    importances = model_output.get("top_importances") or []
    if importances:
        lines.append("Top permutation importances:")
        for feat, score in importances:
            lines.append(f"  - {feat}: {score:.4f}")
    rules = model_output.get("matched_rules") or []
    if rules:
        lines.append("Matched decision-tree rules:")
        for rule in rules:
            conditions = " AND ".join(rule.get("conditions", []))
            lines.append(
                f"  - IF {conditions} THEN {rule.get('prediction')} "
                f"(confidence {rule.get('confidence')}, {rule.get('samples')} samples)"
            )
    return "\n".join(lines)


def _template_explanation(model_output: dict[str, Any]) -> str:
    """Deterministic fallback used when no LLM is available."""
    task = model_output.get("task_type", "prediction")
    prediction = model_output.get("prediction", "n/a")
    dt_acc = model_output.get("dt_accuracy")
    rf_acc = model_output.get("rf_accuracy")
    importances = model_output.get("top_importances") or []
    rules = model_output.get("matched_rules") or []

    parts: list[str] = [f"**Local model {task} result:** `{prediction}`"]

    confidence = model_output.get("prediction_confidence")
    if confidence is not None:
        try:
            parts.append(f"The model is **{float(confidence):.0%} confident** in this prediction.")
        except (TypeError, ValueError):
            pass

    if rf_acc is not None or dt_acc is not None:
        bits = []
        try:
            if rf_acc is not None:
                bits.append(f"Random Forest accuracy {float(rf_acc):.1%}")
            if dt_acc is not None:
                bits.append(f"Decision Tree accuracy {float(dt_acc):.1%}")
        except (TypeError, ValueError):
            pass
        if bits:
            parts.append("Model performance on held-out data: " + ", ".join(bits) + ".")

    if importances:
        top = ", ".join(f"`{f}` ({s:.3f})" for f, s in importances[:3])
        parts.append(f"The features that most influence this outcome are {top}.")

    if rules:
        first = rules[0]
        cond_text = " AND ".join(first.get("conditions", []))
        parts.append(
            "The decision tree reached this prediction via the rule: "
            f"IF {cond_text} THEN {first.get('prediction')} "
            f"(confidence {first.get('confidence')}, based on {first.get('samples')} samples)."
        )

    parts.append(
        "_Caveat: this is a deterministic template explanation (no LLM available). "
        "The numbers above were produced by a locally-trained scikit-learn model — "
        "not by any external service._"
    )
    return "\n\n".join(parts)
