"""
Streamlit app for a data-trained analyst chatbot.

The app accepts structured data (CSV, Excel, JSON) and lightweight text data,
trains local sklearn models when a usable target exists, and answers questions
as a data analyst grounded in the active dataset. When no dataset is provided,
it behaves as a general analyst assistant and invites the user to attach data.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import data_loader as dl
import reasoner
import trainer
from abduction import try_bayesian_abduce
from llm_explainer import explain_with_grok


SAMPLE_ROWS = 10_000

# Default Grok configuration — paste key here or set GROK_API_KEY env var
DEFAULT_GROK_API_KEY = ""  # paste key here or set GROK_API_KEY env var
DEFAULT_GROK_BASE_URL = "https://api.x.ai/v1/chat/completions"
DEFAULT_GROK_MODEL = "grok-2-latest"

LOCAL_PREDICTION_NOTE = (
    "\n\n*Predictions are computed locally; explanation phrased by Grok.*"
)


def _resolve_grok_key() -> str:
    return os.environ.get("GROK_API_KEY") or DEFAULT_GROK_API_KEY


st.set_page_config(
    page_title="Analyst ML Chatbot",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #ffffff;
    --muted: #cbd8e8;
    --panel: rgba(7, 13, 25, 0.92);
    --panel-2: rgba(17, 31, 50, 0.92);
    --line: rgba(175, 211, 238, 0.34);
    --cyan: #49d6d0;
    --green: #79e68f;
    --amber: #f6c66a;
    --rose: #ff7c9b;
}

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

.stApp {
    color: var(--ink);
    background:
        radial-gradient(circle at 12% 10%, rgba(73, 214, 208, 0.24), transparent 30%),
        radial-gradient(circle at 78% 0%, rgba(246, 198, 106, 0.14), transparent 28%),
        linear-gradient(125deg, #07111f 0%, #0d1d32 46%, #132923 100%);
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px);
    background-size: 44px 44px;
    mask-image: linear-gradient(to bottom, rgba(0,0,0,0.65), transparent 78%);
}

.main .block-container {
    padding-top: 1.2rem;
    max-width: 1420px;
}

[data-testid="stSidebar"] {
    background: rgba(5, 12, 24, 0.9);
    border-right: 1px solid var(--line);
}

h1, h2, h3 {
    letter-spacing: 0;
}

.app-head {
    position: relative;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 14px 16px;
    background: rgba(5, 12, 22, 0.84);
    box-shadow: 0 14px 38px rgba(0, 0, 0, 0.24);
    animation: liftIn 520ms ease-out both;
}

.app-head-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
}

.app-title {
    font-size: 1.2rem;
    font-weight: 800;
    color: #ffffff;
}

.status-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 0;
}

.status-pill {
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 8px 12px;
    background: rgba(255,255,255,0.06);
    color: var(--muted);
    font-size: 0.84rem;
    backdrop-filter: blur(12px);
}

.panel {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 18px;
    background: var(--panel);
    box-shadow: 0 18px 46px rgba(0, 0, 0, 0.25);
    animation: liftIn 520ms ease-out both;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin: 18px 0;
}

.metric {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 15px;
    background: var(--panel-2);
}

.metric .value {
    display: block;
    font-size: 1.55rem;
    font-weight: 800;
    color: var(--ink);
}

.metric .label {
    display: block;
    margin-top: 5px;
    color: var(--muted);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.chat-wrap {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 16px;
    background: rgba(4, 10, 20, 0.9);
    min-height: 620px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
}

.insight {
    border-left: 3px solid var(--cyan);
    background: rgba(73,214,208,0.09);
    padding: 12px 14px;
    border-radius: 8px;
    margin: 10px 0;
}

.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 12px 0 4px;
}

.chip {
    border: 1px solid rgba(73,214,208,0.28);
    border-radius: 999px;
    padding: 7px 10px;
    color: #d7fffb;
    background: rgba(73,214,208,0.08);
    font-size: 0.82rem;
}

.stButton > button {
    border-radius: 8px;
    border: 1px solid rgba(73,214,208,0.34);
    background: linear-gradient(135deg, #49d6d0, #79e68f);
    color: #03111a;
    font-weight: 800;
    transition: transform 160ms ease, box-shadow 160ms ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 28px rgba(73, 214, 208, 0.18);
}

div[data-testid="stChatMessage"] {
    border: 1px solid rgba(175, 211, 238, 0.28);
    border-radius: 8px;
    padding: 12px;
    margin: 10px 0;
    background: rgba(17, 31, 50, 0.94);
    color: #ffffff !important;
}

div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] div {
    color: #ffffff !important;
}

div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(33, 74, 96, 0.94);
    border-color: rgba(73, 214, 208, 0.42);
}

div[data-testid="stChatInput"] textarea {
    color: #ffffff !important;
    background: rgba(10, 20, 34, 0.96) !important;
    border: 1px solid rgba(73, 214, 208, 0.38) !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: #b9c9d8 !important;
}

div[data-testid="stAlert"] {
    color: #ffffff;
}

div[data-testid="stChatMessage"] {
    animation: liftIn 240ms ease-out both;
}

@keyframes liftIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 900px) {
    .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .app-head { padding: 12px; }
}
</style>
""",
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults = {
        "df_raw": None,
        "df_clean": None,
        "train_result": None,
        "target_col": None,
        "chat": [],
        "text_corpus": None,
        "text_vectors": None,
        "text_vectorizer": None,
        "data_profile": None,
        "cleaning_report": None,
        "api_provider": "xAI Grok",
        "api_key": _resolve_grok_key(),
        "api_model": DEFAULT_GROK_MODEL,
        "api_base_url": DEFAULT_GROK_BASE_URL,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def read_uploaded_file(uploaded_file: Any) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    if name.endswith(".json"):
        raw = json.load(uploaded_file)
        if isinstance(raw, list):
            return pd.DataFrame(raw)
        if isinstance(raw, dict):
            try:
                return pd.json_normalize(raw)
            except Exception:
                return pd.DataFrame([raw])
    if name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        return text_to_frame(text)
    return pd.read_csv(uploaded_file)


def read_local_file(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".csv"}:
        return dl.load_data(path)
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if isinstance(raw, list):
            return pd.DataFrame(raw)
        if isinstance(raw, dict):
            return pd.json_normalize(raw)
    if suffix == ".txt":
        return text_to_frame(path.read_text(encoding="utf-8", errors="ignore"))
    return dl.load_data(path)


def text_to_frame(text: str) -> pd.DataFrame:
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n|(?<=[.!?])\s+", text) if chunk.strip()]
    if not chunks:
        chunks = ["No readable text was provided."]
    return pd.DataFrame(
        {
            "document_id": range(1, len(chunks) + 1),
            "content": chunks,
            "char_count": [len(chunk) for chunk in chunks],
            "word_count": [len(chunk.split()) for chunk in chunks],
        }
    )


def detect_target(df: pd.DataFrame, requested: str | None) -> str:
    if requested and requested in df.columns:
        return requested

    names = {c.lower(): c for c in df.columns}
    for candidate in ["sales_category", "target", "label", "class", "outcome", "status", "category"]:
        if candidate in names:
            return names[candidate]

    for candidate in ["net_sales", "sales", "revenue", "amount", "profit", "total"]:
        if candidate in names and pd.api.types.is_numeric_dtype(df[names[candidate]]):
            return f"{names[candidate]}_tier"

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return f"{numeric_cols[0]}_tier"

    object_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= min(20, max(2, len(df) // 3))]
    if object_cols:
        return object_cols[0]

    return "analyst_cluster"


def ensure_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    working = df.copy()
    if target_col in working.columns and working[target_col].nunique(dropna=True) >= 2:
        return working

    base_name = target_col.replace("_tier", "")
    if base_name in working.columns and pd.api.types.is_numeric_dtype(working[base_name]):
        series = working[base_name].fillna(working[base_name].median())
    else:
        numeric_cols = [c for c in working.columns if pd.api.types.is_numeric_dtype(working[c])]
        if numeric_cols:
            base_name = numeric_cols[0]
            series = working[base_name].fillna(working[base_name].median())
        else:
            working["text_signal"] = working.astype(str).agg(" ".join, axis=1).str.len()
            base_name = "text_signal"
            series = working[base_name]

    low_q = series.quantile(0.33)
    high_q = series.quantile(0.67)
    if low_q == high_q:
        working[target_col] = np.where(series >= series.median(), "high", "low")
    else:
        working[target_col] = pd.cut(
            series,
            bins=[-np.inf, low_q, high_q, np.inf],
            labels=["low", "medium", "high"],
        ).astype(str)
    return working


def build_text_index(df: pd.DataFrame) -> tuple[list[str] | None, TfidfVectorizer | None, Any | None]:
    text_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not text_cols:
        return None, None, None

    rows = df[text_cols].fillna("").astype(str).agg(" | ".join, axis=1).tolist()
    rows = [row for row in rows if row.strip()]
    if not rows:
        return None, None, None

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=4000, ngram_range=(1, 2))
        vectors = vectorizer.fit_transform(rows)
        return rows, vectorizer, vectors
    except ValueError:
        return None, None, None


def profile_data(df: pd.DataFrame, target_col: str | None) -> dict[str, Any]:
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in df.columns if c not in numeric]
    missing = df.isna().sum().sort_values(ascending=False)
    top_missing = missing[missing > 0].head(8).to_dict()

    profile: dict[str, Any] = {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric": numeric,
        "categorical": categorical,
        "missing": top_missing,
        "target": target_col,
    }

    if target_col and target_col in df.columns:
        profile["target_distribution"] = df[target_col].astype(str).value_counts().head(8).to_dict()

    if numeric:
        stats = df[numeric].describe().T
        profile["numeric_stats"] = stats[["mean", "std", "min", "max"]].round(3).to_dict("index")

    return profile


def parse_facts(prompt: str, feature_names: list[str]) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    for feature in feature_names:
        pattern = rf"{re.escape(feature)}\s*(?:=|is|:)\s*([A-Za-z0-9_.-]+)"
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            value = match.group(1)
            try:
                facts[feature] = float(value)
            except ValueError:
                facts[feature] = value
    return facts


def compact_table(items: dict[str, Any], limit: int = 8) -> str:
    if not items:
        return "None detected."
    lines = []
    for key, value in list(items.items())[:limit]:
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def provider_settings(provider: str, custom_base_url: str, custom_model: str) -> tuple[str, str]:
    presets = {
        "Groq": ("https://api.groq.com/openai/v1/chat/completions", "llama-3.1-8b-instant"),
        "xAI Grok": ("https://api.x.ai/v1/chat/completions", "grok-2-latest"),
        "OpenRouter": ("https://openrouter.ai/api/v1/chat/completions", "meta-llama/llama-3.1-8b-instruct:free"),
        "Custom": (custom_base_url.strip(), custom_model.strip()),
    }
    return presets.get(provider, ("", ""))


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

_PREDICTION_MENU_TRIGGERS = [
    "i want to make a prediction",
    "i want to predict",
    "make a prediction",
    "show prediction options",
    "what can you predict",
    "prediction options",
    "i want to make a prediction from this data",
]


def _is_prediction_menu_trigger(prompt: str) -> bool:
    """Return True if the prompt is a generic prediction trigger (Phase 1)."""
    low = prompt.lower().strip()
    if any(trigger in low for trigger in _PREDICTION_MENU_TRIGGERS):
        return True
    if low in ("predict", "prediction", "make prediction"):
        return True
    return False


def _is_specific_prediction(prompt: str) -> bool:
    """Return True if the prompt is a specific prediction request (Phase 2)."""
    low = prompt.lower().strip()
    if _is_prediction_menu_trigger(low):
        return False
    keywords = [
        "predict", "forecast", "project", "estimate future",
        "next month", "next quarter", "next year", "next week",
        "trend", "will be", "going to be", "expected",
        "classify", "which category", "what will",
        "how much", "how many", "projection",
    ]
    return any(kw in low for kw in keywords)


def _build_prediction_menu() -> str:
    """Analyze the loaded dataset and return a tailored menu of prediction options."""
    df = st.session_state.df_raw
    profile = st.session_state.data_profile or {}
    target_col = st.session_state.target_col

    if df is None:
        return "Please load a dataset first, then I can show you prediction options."

    numeric_cols = [c for c in profile.get("numeric", []) if c != target_col]
    categorical_cols = profile.get("categorical", [])

    # Detect time/date columns
    time_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if not time_cols:
        time_cols = [
            c for c in df.columns
            if any(kw in c.lower() for kw in ["date", "time"])
            and c in df.select_dtypes(include=["object", "datetime"]).columns
        ]

    options: list[str] = []

    # Time-series forecasting options
    if time_cols and numeric_cols:
        for col in numeric_cols[:3]:
            options.append(f"📈 Forecast **{col}** trends over time (time-series prediction)")

    # Classification option
    if target_col and target_col in df.columns:
        classes = df[target_col].astype(str).value_counts().head(4).index.tolist()
        classes_str = ", ".join(classes[:3])
        if len(classes) > 3:
            classes_str += ", ..."
        options.append(
            f"🏷️ Classify new records into **{target_col}** ({classes_str})"
        )

    # Numeric prediction options
    for col in numeric_cols[:3]:
        options.append(f"🔢 Predict **{col}** based on other features")

    options.append(
        "📊 **Custom prediction** — describe what you'd like to predict in your own words"
    )

    col_list = ", ".join(f"`{c}`" for c in list(df.columns)[:15])
    if len(df.columns) > 15:
        col_list += f", ... ({len(df.columns) - 15} more)"

    menu = "Based on your loaded dataset, here are the predictions I can make:\n\n"
    for i, opt in enumerate(options, 1):
        menu += f"{i}. {opt}\n"
    menu += (
        f"\n**Available columns**: {col_list}\n\n"
        "Just type what you'd like to predict — for example:\n"
        "- *\"Forecast monthly net_sales for the next 3 months\"*\n"
        "- *\"Which category will a new record fall into?\"*\n"
        "- *\"Predict unit_price based on quantity and discount\"*\n"
    )
    return menu


def predict_locally(prompt: str) -> dict:
    """
    Run the locally-trained scikit-learn models against facts parsed from the
    prompt and return a structured dict of REAL numbers. No LLM is involved.
    """
    if not st.session_state.train_result:
        return {"error": "I need a trained model before making predictions. Click **Train agent** in the sidebar first."}

    try:
        models = trainer.load_models()
        rules_payload = trainer.load_rules()
        importance_payload = trainer.load_importances()
    except FileNotFoundError as exc:
        return {"error": f"Trained artifacts are missing on disk: {exc}"}

    rf = models["rf"]
    le = models["le"]
    feature_names = rules_payload["feature_names"]

    df_clean = st.session_state.df_clean
    if df_clean is None:
        return {"error": "Cleaned dataset is not in session state — retrain to refresh."}

    missing_cols = [f for f in feature_names if f not in df_clean.columns]
    if missing_cols:
        return {"error": f"Cleaned data is missing trained feature columns: {missing_cols}"}

    facts = parse_facts(prompt, feature_names)

    medians = df_clean[feature_names].median(numeric_only=True)
    row: list[float] = []
    for feat in feature_names:
        if feat in facts:
            try:
                row.append(float(facts[feat]))
                continue
            except (ValueError, TypeError):
                pass
        row.append(float(medians.get(feat, 0.0)))

    X = np.array(row, dtype=float).reshape(1, -1)
    try:
        pred_encoded = int(rf.predict(X)[0])
        pred_label = str(le.inverse_transform([pred_encoded])[0])
        proba = rf.predict_proba(X)[0]
        confidence = float(proba[pred_encoded])
    except Exception as exc:
        return {"error": f"Local model prediction failed: {exc}"}

    matched_rules = reasoner.deduce(facts, top_k=3) if facts else []
    importances_ranked = sorted(
        importance_payload["importances"].items(), key=lambda kv: kv[1], reverse=True
    )[:5]

    return {
        "task_type": "classification",
        "prediction": pred_label,
        "prediction_confidence": round(confidence, 4),
        "dt_accuracy": rules_payload.get("accuracy"),
        "rf_accuracy": importance_payload.get("rf_accuracy"),
        "facts_used": facts,
        "feature_vector": {feat: row[i] for i, feat in enumerate(feature_names)},
        "top_importances": importances_ranked,
        "matched_rules": matched_rules,
    }


def _format_prediction_response(model_output: dict, explanation: str) -> str:
    facts = model_output.get("facts_used") or {}
    facts_str = (
        ", ".join(f"`{k}={v}`" for k, v in facts.items())
        if facts
        else "median values for all features (no specific facts supplied)"
    )
    header_lines = [
        "### Local prediction",
        f"- **Predicted {model_output.get('task_type', 'class')}:** `{model_output['prediction']}`",
        f"- **Confidence:** {model_output['prediction_confidence']:.1%}",
        f"- **Inputs used:** {facts_str}",
    ]
    return "\n".join(header_lines) + "\n\n### Explanation\n" + explanation + LOCAL_PREDICTION_NOTE


# ---------------------------------------------------------------------------
# Missing-value analysis helper
# ---------------------------------------------------------------------------

def _analyze_missing_values() -> str:
    """Build a comprehensive missing-value report with types and handling techniques."""
    df = st.session_state.df_raw
    if df is None:
        return "No dataset loaded. Please upload data first."

    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(df.isna().sum().sum())
    overall_pct = (total_missing / total_cells * 100) if total_cells else 0

    missing_series = df.isna().sum()
    cols_with_missing = missing_series[missing_series > 0].sort_values(ascending=False)

    if cols_with_missing.empty:
        return (
            "## \u2705 No Missing Values Detected\n\n"
            f"Your dataset has **{df.shape[0]:,} rows** and **{df.shape[1]} columns** "
            "with zero missing values. The data is complete and ready for analysis."
        )

    lines: list[str] = [
        "## \U0001f50d Missing Value Analysis\n",
        f"**Dataset**: {df.shape[0]:,} rows \u00d7 {df.shape[1]} columns "
        f"| **Total missing cells**: {total_missing:,} ({overall_pct:.2f}%)\n",
        f"**Columns affected**: {len(cols_with_missing)} out of {df.shape[1]}\n",
        "---",
        "",
    ]

    # Per-column analysis
    lines.append("### Per-Column Breakdown\n")
    lines.append("| Column | Type | Missing | % | Missingness Type | Recommended Technique |")
    lines.append("|--------|------|---------|---|------------------|----------------------|")

    technique_details: list[str] = []

    for col, count in cols_with_missing.items():
        pct = count / len(df) * 100
        dtype = str(df[col].dtype)
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        is_datetime = pd.api.types.is_datetime64_any_dtype(df[col])
        nunique = df[col].nunique(dropna=True)
        nunique_ratio = nunique / len(df) if len(df) > 0 else 0

        # Classify missingness type (heuristic)
        if pct > 60:
            miss_type = "\u26a0\ufe0f Structural"
            miss_explain = "Extremely high missing rate suggests the column may not apply to all records"
        elif pct > 30:
            miss_type = "\U0001f534 Likely MNAR"
            miss_explain = "Missing Not At Random \u2014 the absence likely depends on the value itself"
        else:
            # Check correlation with other columns having missing values
            other_missing = df.drop(columns=[col]).isna().any(axis=1)
            col_missing = df[col].isna()
            if other_missing.sum() > 0:
                overlap = (col_missing & other_missing).sum()
                overlap_ratio = overlap / col_missing.sum() if col_missing.sum() > 0 else 0
                if overlap_ratio > 0.6:
                    miss_type = "\U0001f7e1 Likely MAR"
                    miss_explain = "Missing At Random \u2014 missingness correlates with other columns"
                else:
                    miss_type = "\U0001f7e2 Likely MCAR"
                    miss_explain = "Missing Completely At Random \u2014 no obvious pattern detected"
            else:
                miss_type = "\U0001f7e2 Likely MCAR"
                miss_explain = "Missing Completely At Random \u2014 no obvious pattern detected"

        # Determine handling technique
        if pct > 60:
            technique = "Drop column"
            technique_long = (
                f"**{col}** ({pct:.1f}% missing): **Drop this column**. "
                "With over 60% missing data, imputation would introduce more noise than signal. "
                "Use `df.drop(columns=['{col}'])`."
            )
        elif is_numeric and pct <= 5:
            skew = df[col].skew() if df[col].notna().sum() > 2 else 0
            if abs(skew) > 1:
                technique = "Median imputation"
                technique_long = (
                    f"**{col}** ({pct:.1f}% missing, skewed distribution): **Median imputation**. "
                    f"The data is skewed (skewness={skew:.2f}), so median is more robust than mean. "
                    f"Use `df['{col}'].fillna(df['{col}'].median())`."
                )
            else:
                technique = "Mean imputation"
                technique_long = (
                    f"**{col}** ({pct:.1f}% missing, ~normal distribution): **Mean imputation**. "
                    f"Use `df['{col}'].fillna(df['{col}'].mean())`."
                )
        elif is_numeric and pct <= 30:
            technique = "KNN / Iterative imputation"
            technique_long = (
                f"**{col}** ({pct:.1f}% missing): **KNN Imputer or Iterative Imputer**. "
                "Moderate missingness in a numeric column benefits from model-based imputation. "
                "Use `sklearn.impute.KNNImputer(n_neighbors=5)` or `IterativeImputer()`."
            )
        elif is_numeric:
            technique = "Flag + Median"
            technique_long = (
                f"**{col}** ({pct:.1f}% missing): **Create a missing-indicator flag + median fill**. "
                f"Add `df['{col}_was_missing'] = df['{col}'].isna().astype(int)` then fill with median. "
                "The flag preserves the information that the value was missing."
            )
        elif is_datetime:
            technique = "Forward/Backward fill"
            technique_long = (
                f"**{col}** ({pct:.1f}% missing): **Forward fill (ffill) or backward fill (bfill)**. "
                f"Use `df['{col}'].fillna(method='ffill')` for time-series continuity."
            )
        elif nunique_ratio > 0.5:
            technique = "Drop column (high cardinality)"
            technique_long = (
                f"**{col}** ({pct:.1f}% missing, {nunique} unique values): "
                "**Consider dropping** \u2014 high-cardinality categorical columns with missing data "
                "are difficult to impute meaningfully."
            )
        elif pct <= 10:
            technique = "Mode imputation"
            technique_long = (
                f"**{col}** ({pct:.1f}% missing, categorical): **Mode (most frequent) imputation**. "
                f"Use `df['{col}'].fillna(df['{col}'].mode()[0])`."
            )
        else:
            technique = "Mode or 'Unknown'"
            technique_long = (
                f"**{col}** ({pct:.1f}% missing, categorical): **Fill with mode or a new category 'Unknown'**. "
                f"Use `df['{col}'].fillna('Unknown')` to explicitly mark missing records."
            )

        col_type = "numeric" if is_numeric else ("datetime" if is_datetime else "categorical")
        lines.append(f"| {col} | {col_type} | {count:,} | {pct:.1f}% | {miss_type} | {technique} |")
        technique_details.append(technique_long)

    # Missingness types explanation
    lines.extend([
        "",
        "---",
        "",
        "### \U0001f4d6 Missing Value Types Explained\n",
        "| Type | Meaning | Example |",
        "|------|---------|--------|",
        "| **MCAR** (Missing Completely At Random) | No pattern \u2014 missingness is purely random | Data entry errors, random sensor failures |",
        "| **MAR** (Missing At Random) | Missingness depends on *other observed* columns | Income missing more often for younger respondents |",
        "| **MNAR** (Missing Not At Random) | Missingness depends on the *missing value itself* | High earners skip income questions |",
        "| **Structural** | Column doesn\u2019t apply to most records | \u2018Spouse name\u2019 missing for unmarried people |",
    ])

    # Detailed handling recommendations
    lines.extend([
        "",
        "---",
        "",
        "### \U0001f6e0\ufe0f Recommended Handling Techniques\n",
    ])
    for detail in technique_details:
        lines.append(f"- {detail}")

    # Quick-apply summary
    lines.extend([
        "",
        "---",
        "",
        "### \u26a1 Quick Summary\n",
        f"- **Columns to consider dropping** (>60% missing): "
        + (", ".join(f"`{c}`" for c, cnt in cols_with_missing.items() if cnt / len(df) > 0.6) or "None"),
        f"- **Numeric columns to impute**: "
        + (", ".join(
            f"`{c}`" for c, cnt in cols_with_missing.items()
            if pd.api.types.is_numeric_dtype(df[c]) and cnt / len(df) <= 0.6
        ) or "None"),
        f"- **Categorical columns to fill**: "
        + (", ".join(
            f"`{c}`" for c, cnt in cols_with_missing.items()
            if not pd.api.types.is_numeric_dtype(df[c]) and cnt / len(df) <= 0.6
        ) or "None"),
    ])

    return "\n".join(lines)


def answer_without_data(prompt: str) -> str:
    return (
        "I can help as a data analyst right now, but I will become much sharper after you upload "
        "CSV, Excel, JSON, or text data. Ask me for analysis plans, KPI definitions, chart ideas, "
        "data cleaning steps, experiment design, forecasting strategy, or sales diagnosis. Once "
        "data is loaded, I will train local machine-learning models and answer from the evidence."
    )


def answer_with_text_retrieval(prompt: str) -> str | None:
    rows = st.session_state.text_corpus
    vectorizer = st.session_state.text_vectorizer
    vectors = st.session_state.text_vectors
    if rows is None or vectorizer is None or vectors is None:
        return None

    query_vec = vectorizer.transform([prompt])
    scores = cosine_similarity(query_vec, vectors).ravel()
    best_idx = scores.argsort()[::-1][:4]
    if len(best_idx) == 0 or scores[best_idx[0]] < 0.04:
        return None

    evidence = "\n".join(
        f"- Match {i + 1} ({scores[idx]:.0%} relevance): {rows[idx][:420]}"
        for i, idx in enumerate(best_idx)
    )
    return (
        "I found the closest evidence in your text data:\n\n"
        f"{evidence}\n\n"
        "Analyst read: the strongest answer should be based on the first match, with the later "
        "matches used as supporting context. If you ask a more specific follow-up, I can narrow it."
    )


def answer_with_data(prompt: str) -> str:
    df = st.session_state.df_raw
    profile = st.session_state.data_profile or {}
    train_result = st.session_state.train_result
    target_col = st.session_state.target_col
    low_prompt = prompt.lower()

    if df is None:
        return answer_without_data(prompt)

    retrieval = answer_with_text_retrieval(prompt)
    if retrieval and any(word in low_prompt for word in ["document", "text", "say", "mention", "find", "what is"]):
        return retrieval

    if any(word in low_prompt for word in ["overview", "summary", "describe", "profile", "dataset"]):
        return (
            f"Dataset overview:\n\n"
            f"- Rows: {profile.get('rows', len(df)):,}\n"
            f"- Columns: {profile.get('columns', len(df.columns)):,}\n"
            f"- Numeric columns: {len(profile.get('numeric', []))}\n"
            f"- Categorical/text columns: {len(profile.get('categorical', []))}\n"
            f"- Active target: {target_col or 'not selected'}\n\n"
            f"Target distribution:\n{compact_table(profile.get('target_distribution', {}))}\n\n"
            f"Missing values:\n{compact_table(profile.get('missing', {}))}"
        )

    if any(word in low_prompt for word in ["missing", "null", "clean", "quality"]):
        return _analyze_missing_values()

    if any(word in low_prompt for word in ["important", "importance", "driver", "drivers", "influence"]):
        if train_result and Path("feature_importances.pkl").exists():
            imp_data = trainer.load_importances()
            ranked = sorted(imp_data["importances"].items(), key=lambda item: item[1], reverse=True)[:8]
            lines = [f"- {feature}: {score:.4f}" for feature, score in ranked]
            return (
                "Top model drivers from permutation importance:\n\n"
                + "\n".join(lines)
                + "\n\nThese are the fields that most changed model accuracy when shuffled, so they are "
                "the best first suspects for business investigation."
            )
        return "I need a trained model before I can rank drivers. Load data and click Train agent."

    if any(word in low_prompt for word in ["why", "cause", "explain"]) and train_result:
        classes = train_result["class_names"]
        observed = next((c for c in classes if c.lower() in low_prompt), classes[-1])
        try:
            imp_result = reasoner.abduce(observed, top_k=5)
            bayes = None
            if st.session_state.df_clean is not None:
                bayes = try_bayesian_abduce(st.session_state.df_clean, observed, target_col, top_k=3)

            lines = [
                f"Best explanation for outcome '{observed}':",
                "",
                *[
                    f"- {c['feature']}: importance {c['importance']:.4f}. {c['direction_hint']}."
                    for c in imp_result["causes"]
                ],
            ]
            if bayes:
                lines.extend(["", "Bayesian posterior hints:"])
                lines.extend(
                    f"- {b['feature']} around {b['feature_value_range']}: score {b['posterior_score']:.4f}"
                    for b in bayes
                )
            return "\n".join(lines)
        except Exception as exc:
            return f"I tried to explain that outcome, but the model could not complete abduction: {exc}"

    if any(word in low_prompt for word in ["predict", "classify", "forecast", "estimate"]):
        if train_result:
            return _build_prediction_menu()
        return (
            "I need a trained model before making predictions. "
            "Load your data and click **Train agent** in the sidebar first."
        )

    numeric_cols = profile.get("numeric", [])
    mentioned_numeric = [col for col in numeric_cols if col.lower() in low_prompt]
    if mentioned_numeric:
        col = mentioned_numeric[0]
        stats = df[col].describe()
        return (
            f"Column read for '{col}':\n\n"
            f"- Mean: {stats['mean']:.3f}\n"
            f"- Median: {stats['50%']:.3f}\n"
            f"- Minimum: {stats['min']:.3f}\n"
            f"- Maximum: {stats['max']:.3f}\n\n"
            "This is a quick statistical answer. Ask for drivers or causes if you want model reasoning."
        )

    return (
        "Here is the analyst read from the active data: start with the dataset profile, inspect missing "
        "values, then use the trained model drivers to focus the business story. I can answer specific "
        "questions like `summarize this dataset`, `what are the top drivers`, `why high`, "
        "`predict if feature=value`, or `show missing values`."
    )


def generate_response(prompt: str) -> str:
    # Phase 1: Generic prediction trigger → show dataset-aware prediction menu
    if st.session_state.df_raw is not None and _is_prediction_menu_trigger(prompt):
        return _build_prediction_menu()

    # Phase 2: Specific prediction request → LOCAL model produces the prediction,
    # Grok (if configured) only narrates the resulting numbers.
    if st.session_state.df_raw is not None and _is_specific_prediction(prompt):
        model_output = predict_locally(prompt)
        if "error" in model_output:
            return model_output["error"]

        explanation = explain_with_grok(
            model_output,
            api_key=st.session_state.get("api_key") or _resolve_grok_key(),
            base_url=st.session_state.get("api_base_url") or DEFAULT_GROK_BASE_URL,
            model=st.session_state.get("api_model") or DEFAULT_GROK_MODEL,
        )
        return _format_prediction_response(model_output, explanation)

    # All other questions are answered from local data only — no LLM enhancement.
    return answer_with_data(prompt)


def train_agent(df: pd.DataFrame, target_request: str, tree_depth: int, n_estimators: int) -> None:
    target_col = detect_target(df, target_request.strip() or None)
    model_df = ensure_target(df, target_col)
    clean_df, cleaning_report = dl.clean_data(model_df, target_col=target_col)

    if clean_df[target_col].nunique(dropna=True) < 2:
        raise ValueError("The target has fewer than two classes after cleaning.")

    result = trainer.train(
        clean_df,
        target_col=target_col,
        max_tree_depth=tree_depth,
        n_estimators=n_estimators,
    )

    corpus, vectorizer, vectors = build_text_index(df)

    st.session_state.df_raw = model_df
    st.session_state.df_clean = clean_df
    st.session_state.train_result = result
    st.session_state.target_col = target_col
    st.session_state.text_corpus = corpus
    st.session_state.text_vectorizer = vectorizer
    st.session_state.text_vectors = vectors
    st.session_state.data_profile = profile_data(model_df, target_col)
    st.session_state.cleaning_report = cleaning_report

    st.session_state.chat.append(
        {
            "role": "assistant",
            "content": (
                f"Data loaded and model trained. I learned {len(model_df):,} rows, "
                f"{len(model_df.columns):,} columns, and I am using `{target_col}` as the outcome. "
                "Ask me for a summary, drivers, causes, predictions, missing values, or document evidence."
            ),
        }
    )


def render_metrics() -> None:
    profile = st.session_state.data_profile or {}
    result = st.session_state.train_result or {}
    st.markdown(
        f"""
<div class="metric-grid">
    <div class="metric"><span class="value">{profile.get('rows', 0):,}</span><span class="label">Rows Learned</span></div>
    <div class="metric"><span class="value">{profile.get('columns', 0):,}</span><span class="label">Signals</span></div>
    <div class="metric"><span class="value">{result.get('rf_accuracy', 0):.0%}</span><span class="label">Model Accuracy</span></div>
    <div class="metric"><span class="value">{result.get('n_rules', 0):,}</span><span class="label">Reasoning Rules</span></div>
</div>
""",
        unsafe_allow_html=True,
    )


init_state()


with st.sidebar:
    st.markdown("## Analyst ML Chatbot")
    st.divider()

    source = st.radio(
        "Data source",
        ["Upload data", "Paste text", "Generate sample sales data", "Use local file path"],
        index=2,
    )

    uploaded = None
    pasted_text = ""
    local_path = ""
    rows = SAMPLE_ROWS

    if source == "Upload data":
        uploaded = st.file_uploader("CSV, Excel, JSON, or TXT", type=["csv", "xlsx", "xls", "json", "txt"])
        if uploaded is not None:
            st.info(f"Selected `{uploaded.name}`. Click Train agent to learn it.")
    elif source == "Paste text":
        pasted_text = st.text_area("Paste notes, documents, or raw text", height=180)
    elif source == "Generate sample sales data":
        rows = st.slider("Rows", 1_000, 50_000, SAMPLE_ROWS, step=1_000)
    else:
        local_path = st.text_input("File path", value="")

    st.divider()
    st.markdown("### AI provider")
    provider = st.selectbox(
        "Chat engine",
        ["Offline ML", "Groq", "xAI Grok", "OpenRouter", "Custom"],
        index=["Offline ML", "Groq", "xAI Grok", "OpenRouter", "Custom"].index(
            st.session_state.get("api_provider", "Offline ML")
        ),
    )
    st.session_state.api_provider = provider

    if provider != "Offline ML":
        default_base, default_model = provider_settings(
            provider,
            st.session_state.get("api_base_url", ""),
            st.session_state.get("api_model", ""),
        )
        st.session_state.api_key = st.text_input(
            "API key",
            value=st.session_state.get("api_key", ""),
            type="password",
            placeholder="Paste provider key",
        )
        st.session_state.api_model = st.text_input("Model", value=st.session_state.get("api_model") or default_model)
        if provider == "Custom":
            st.session_state.api_base_url = st.text_input(
                "Chat completions URL",
                value=st.session_state.get("api_base_url") or default_base,
                placeholder="https://api.example.com/v1/chat/completions",
            )
        else:
            st.caption(default_base)

    st.divider()
    st.markdown("### Training")
    target_request = st.text_input("Target column or outcome", value=st.session_state.target_col or "")
    tree_depth = st.slider("Reasoning depth", 2, 12, 6)
    n_estimators = st.slider("Forest trees", 50, 500, 250, step=50)

    train_clicked = st.button("Train agent", use_container_width=True)
    reset_clicked = st.button("Reset chat", use_container_width=True)

    if reset_clicked:
        st.session_state.chat = []
        st.rerun()

    if st.session_state.train_result:
        st.divider()
        st.markdown("### Active model")
        st.write(f"Target: `{st.session_state.target_col}`")
        st.write(f"Random Forest: `{st.session_state.train_result['rf_accuracy']:.1%}`")
        st.write(f"Decision Tree: `{st.session_state.train_result['dt_accuracy']:.1%}`")


if train_clicked:
    try:
        with st.spinner("Reading data and fitting the local ML agent..."):
            if source == "Upload data":
                if uploaded is None:
                    st.error("Upload a file first.")
                    st.stop()
                raw_df = read_uploaded_file(uploaded)
            elif source == "Paste text":
                if not pasted_text.strip():
                    st.error("Paste some text first.")
                    st.stop()
                raw_df = text_to_frame(pasted_text)
            elif source == "Generate sample sales data":
                raw_df = dl.generate_synthetic_data(n_rows=rows)
            else:
                if not local_path.strip():
                    st.error("Enter a local file path first.")
                    st.stop()
                raw_df = read_local_file(local_path.strip())

            train_agent(raw_df, target_request, tree_depth, n_estimators)
        st.success("Agent trained and ready to chat.")
        st.rerun()
    except Exception as exc:
        st.error(f"Training failed: {exc}")


trained = st.session_state.train_result is not None
status = "Model trained" if trained else "No data required to start"
target_text = st.session_state.target_col or "auto-selected after training"
data_text = (
    f"{st.session_state.data_profile['rows']:,} rows active"
    if st.session_state.data_profile
    else "upload, paste, or generate data"
)

st.markdown(
    f"""
<section class="app-head">
    <div class="app-head-row">
        <div class="app-title">Analyst Chatbot</div>
        <div class="status-row">
            <div class="status-pill">{status}</div>
            <div class="status-pill">Target: {target_text}</div>
            <div class="status-pill">{data_text}</div>
            <div class="status-pill">{st.session_state.get("api_provider", "Offline ML")}</div>
        </div>
    </div>
</section>
""",
    unsafe_allow_html=True,
)


if trained:
    render_metrics()


st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

if not st.session_state.chat:
    st.session_state.chat = [
        {
            "role": "assistant",
            "content": (
                "Hi. Upload or generate data from the sidebar, train me, then ask anything. "
                "You can also chat now if you only need analyst guidance."
            ),
        }
    ]

for message in st.session_state.chat:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown("#### Pinned")
pin_prompts = [
    "Summarize",
    "Top drivers",
    "Missing values",
    "Why high?",
    "Predict",
]
pin_cols = st.columns(len(pin_prompts))
pin_map = {
    "Summarize": "Summarize this dataset",
    "Top drivers": "What are the top drivers?",
    "Missing values": "Show missing values",
    "Why high?": "Why is the high outcome happening?",
    "Predict": "I want to make a prediction from this data",
}
for idx, label in enumerate(pin_prompts):
    with pin_cols[idx]:
        if st.button(label, key=f"pin_{idx}", use_container_width=True):
            pinned_prompt = pin_map[label]
            st.session_state.chat.append({"role": "user", "content": pinned_prompt})
            st.session_state.chat.append({"role": "assistant", "content": generate_response(pinned_prompt)})
            st.rerun()

prompt = st.chat_input("Message the analyst chatbot...")
if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    st.session_state.chat.append({"role": "assistant", "content": generate_response(prompt)})
    st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.df_raw is not None:
    with st.expander("Data and model details", expanded=False):
        tabs = st.tabs(["Preview", "Drivers", "Profile", "Cleaning"])
        with tabs[0]:
            st.dataframe(st.session_state.df_raw.head(30), use_container_width=True, height=360)
        with tabs[1]:
            if st.session_state.train_result and Path("feature_importances.pkl").exists():
                imp = trainer.load_importances()["importances"]
                ranked_imp = pd.DataFrame(
                    sorted(imp.items(), key=lambda item: item[1], reverse=True)[:15],
                    columns=["feature", "importance"],
                )
                st.bar_chart(ranked_imp.set_index("feature"))
            else:
                st.info("Train the model to see drivers.")
        with tabs[2]:
            st.json(st.session_state.data_profile or {})
        with tabs[3]:
            report = st.session_state.get("cleaning_report")
            if report:
                st.caption(
                    "Audit trail of every action `data_loader.clean_data` performed. "
                    "No LLM was involved in preparing this data."
                )
                st.json(report)
            else:
                st.info("Train the model to see the cleaning report.")
