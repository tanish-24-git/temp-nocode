# services/ops/nlp_ops.py
from typing import Dict, Any, List, Tuple, Callable, Optional
import re
import logging
import json

import pandas as pd

logger = logging.getLogger(__name__)

NlpOpFunc = Callable[..., pd.DataFrame]

# -----------------------------
# Basic text cleaning
# -----------------------------

def text_lowercase(df: pd.DataFrame, field: str = "text") -> pd.DataFrame:
    df = df.copy()
    if field in df.columns:
        df[field] = df[field].astype(str).str.lower()
    return df


def text_uppercase(df: pd.DataFrame, field: str = "text") -> pd.DataFrame:
    df = df.copy()
    if field in df.columns:
        df[field] = df[field].astype(str).str.upper()
    return df


def remove_punctuation(df: pd.DataFrame, field: str = "text") -> pd.DataFrame:
    df = df.copy()
    if field in df.columns:
        df[field] = df[field].astype(str).str.replace(r"[^\w\s]", "", regex=True)
    return df


def remove_stopwords(df: pd.DataFrame, field: str = "text") -> pd.DataFrame:
    """
    Lite implementation: simple English stopword set.
    For production you may plug in NLTK/spacy etc.
    """
    df = df.copy()
    if field not in df.columns:
        return df
    stopwords = {
        "the", "a", "an", "and", "or", "is", "are", "was", "were",
        "this", "that", "to", "of", "in", "on", "for", "with", "at",
    }
    def drop_sw(text: str) -> str:
        tokens = str(text).split()
        return " ".join([t for t in tokens if t.lower() not in stopwords])
    df[field] = df[field].apply(drop_sw)
    return df


def remove_urls(df: pd.DataFrame, field: str = "text") -> pd.DataFrame:
    pattern = re.compile(r"http[s]?://\S+")
    df = df.copy()
    if field in df.columns:
        df[field] = df[field].astype(str).apply(lambda x: pattern.sub("", x))
    return df


def strip_html(df: pd.DataFrame, field: str = "text") -> pd.DataFrame:
    pattern = re.compile(r"<.*?>")
    df = df.copy()
    if field in df.columns:
        df[field] = df[field].astype(str).apply(lambda x: pattern.sub("", x))
    return df


# -----------------------------
# Tokenization-aware utilities
# -----------------------------

def estimate_token_count(df: pd.DataFrame, field: str = "text", out_field: str = "token_count") -> pd.DataFrame:
    """
    Rough token estimate: whitespace-based word count.
    You can replace this later with a real tokenizer.
    """
    df = df.copy()
    if field in df.columns:
        df[out_field] = df[field].astype(str).apply(lambda x: len(str(x).split()))
    return df


def truncate_by_tokens(df: pd.DataFrame, field: str = "text", max_tokens: int = 512) -> pd.DataFrame:
    df = df.copy()
    if field in df.columns:
        def trunc(text: str) -> str:
            tokens = str(text).split()
            if len(tokens) <= max_tokens:
                return text
            return " ".join(tokens[:max_tokens])
        df[field] = df[field].apply(trunc)
    return df


# -----------------------------
# Prompt/completion construction
# -----------------------------

def build_prompt_completion(
    df: pd.DataFrame,
    prompt_template: str,
    completion_template: str,
    prompt_field: str = "prompt",
    completion_field: str = "completion",
) -> pd.DataFrame:
    """
    Render prompt & completion using Python format strings and row values.

    Example:
      prompt_template = "Question: {question}"
      completion_template = "Answer: {answer}"
    """
    df = df.copy()

    def render_row(row: pd.Series, tmpl: str) -> str:
        try:
            return tmpl.format(**row.to_dict())
        except Exception:
            # fallback: nothing substituted
            return tmpl

    df[prompt_field] = df.apply(lambda r: render_row(r, prompt_template), axis=1)
    df[completion_field] = df.apply(lambda r: render_row(r, completion_template), axis=1)
    return df


# -----------------------------
# Chunking (simplified)
# -----------------------------

def chunk_by_tokens(
    df: pd.DataFrame,
    field: str = "text",
    max_tokens: int = 256,
    stride: int = 256,
) -> pd.DataFrame:
    """
    Very simple chunking based on the rough tokenization above.
    Each row in the input may be expanded to multiple rows in the output.
    """
    if field not in df.columns:
        return df

    new_rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        text = str(row[field])
        tokens = text.split()
        if not tokens:
            new_rows.append(row.to_dict())
            continue

        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i : i + max_tokens]
            new_row = row.to_dict()
            new_row[field] = " ".join(chunk_tokens)
            new_rows.append(new_row)
            i += stride

    return pd.DataFrame(new_rows)


# -----------------------------
# PII & safety (regex-based)
# -----------------------------

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_PATTERN = re.compile(r"\+?\d[\d\-\s]{7,}")
IP_PATTERN = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")

def mask_emails(df: pd.DataFrame, field: str = "text", mask: str = "<EMAIL>") -> pd.DataFrame:
    df = df.copy()
    if field in df.columns:
        df[field] = df[field].astype(str).apply(lambda x: EMAIL_PATTERN.sub(mask, x))
    return df


def mask_phone_numbers(df: pd.DataFrame, field: str = "text", mask: str = "<PHONE>") -> pd.DataFrame:
    df = df.copy()
    if field in df.columns:
        df[field] = df[field].astype(str).apply(lambda x: PHONE_PATTERN.sub(mask, x))
    return df


def mask_ip_addresses(df: pd.DataFrame, field: str = "text", mask: str = "<IP>") -> pd.DataFrame:
    df = df.copy()
    if field in df.columns:
        df[field] = df[field].astype(str).apply(lambda x: IP_PATTERN.sub(mask, x))
    return df


def regex_pii_mask(
    df: pd.DataFrame,
    field: str,
    pattern: str,
    mask: str = "<PII>",
) -> pd.DataFrame:
    df = df.copy()
    if field in df.columns:
        compiled = re.compile(pattern)
        df[field] = df[field].astype(str).apply(lambda x: compiled.sub(mask, x))
    return df


# -----------------------------
# JSONL preparation
# -----------------------------

def to_jsonl_records(
    df: pd.DataFrame,
    input_fields: List[str],
    mode: str = "prompt_completion",
) -> pd.DataFrame:
    """
    Converts fields into a single column of JSON-serializable dicts
    that will later be written as JSONL.

    mode:
      - "prompt_completion": expects `prompt`, `completion`
      - "chat": expects `messages` column or builds a messages list
    """
    df = df.copy()
    records: List[Dict[str, Any]] = []

    if mode == "prompt_completion":
        for _, row in df.iterrows():
            rec = {
                "prompt": row.get("prompt"),
                "completion": row.get("completion"),
            }
            records.append({"jsonl": rec})
    else:
        # generic: store selected fields
        for _, row in df.iterrows():
            rec = {f: row.get(f) for f in input_fields}
            records.append({"jsonl": rec})

    return pd.DataFrame(records)


# -----------------------------
# Metadata generation
# -----------------------------

def generate_nlp_metadata(
    df: pd.DataFrame,
    text_field: str = "text",
    tokenizer_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Simple metadata: sample count, avg approximate tokens.
    """
    n = len(df)
    if text_field in df.columns:
        avg_tokens = df[text_field].astype(str).apply(lambda x: len(str(x).split())).mean()
    else:
        avg_tokens = None

    return {
        "sample_count": int(n),
        "avg_tokens": float(avg_tokens) if avg_tokens is not None else None,
        "tokenizer_name": tokenizer_name,
    }

# -----------------------------
# OP REGISTRY + dispatcher
# -----------------------------

OP_REGISTRY: Dict[str, NlpOpFunc] = {
    # Cleaning
    "text_lowercase": text_lowercase,
    "text_uppercase": text_uppercase,
    "remove_punctuation": remove_punctuation,
    "remove_stopwords": remove_stopwords,
    "remove_urls": remove_urls,
    "strip_html": strip_html,
    # Token
    "estimate_token_count": estimate_token_count,
    "truncate_by_tokens": truncate_by_tokens,
    # Prompt/Completion
    "build_prompt_completion": build_prompt_completion,
    # Chunking
    "chunk_by_tokens": chunk_by_tokens,
    # PII
    "mask_emails": mask_emails,
    "mask_phone_numbers": mask_phone_numbers,
    "mask_ip_addresses": mask_ip_addresses,
    "regex_pii_mask": regex_pii_mask,
    # JSONL prep (we keep it simple – actual writing is done in engine)
    "to_jsonl": to_jsonl_records,
}

def list_ops() -> List[str]:
    return sorted(list(OP_REGISTRY.keys()))


def apply_op(
    df: pd.DataFrame,
    op_name: str,
    func: NlpOpFunc,
    step_config: Dict[str, Any],
    nlp_config: Dict[str, Any],
    meta_store: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    """
    Generic dispatcher used by PreprocessingEngine._run_nlp().
    It maps step_config into kwargs for the function.
    """
    warnings: List[str] = []

    kwargs = {k: v for k, v in step_config.items() if k != "op"}

    # Many ops work over a 'field', default from nlp_config.text_column if present
    if "field" not in kwargs and "text_column" in nlp_config:
        kwargs["field"] = nlp_config["text_column"]

    # Special case: build_prompt_completion needs templates
    if op_name == "build_prompt_completion":
        if "prompt_template" not in kwargs or "completion_template" not in kwargs:
            warnings.append("build_prompt_completion needs prompt_template & completion_template; skipped")
            return df, meta_store, warnings

    # Special case: to_jsonl – we don't write here, only pack JSON-able payloads
    if op_name == "to_jsonl":
        input_fields = kwargs.get("input_fields") or ["prompt", "completion"]
        mode = kwargs.get("output_mode", "prompt_completion")
        new_df = func(df, input_fields=input_fields, mode=mode)
        meta_store.setdefault("jsonl", {})
        meta_store["jsonl"]["mode"] = mode
        meta_store["jsonl"]["fields"] = input_fields
        return new_df, meta_store, warnings

    # Generic call
    new_df = func(df, **kwargs)

    # Example: update basic NLP metadata once if requested
    if step_config.get("emit_metadata"):
        text_field = kwargs.get("field") or nlp_config.get("text_column", "text")
        meta = generate_nlp_metadata(new_df, text_field=text_field, tokenizer_name=nlp_config.get("tokenizer_name"))
        meta_store.setdefault("nlp_metadata", meta)

    return new_df, meta_store, warnings
