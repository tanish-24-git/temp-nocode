# services/ops/tabular_ops.py
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
import json
import logging

logger = logging.getLogger(__name__)

# -------------------------
# Basic transforms (pure-ish)
# -------------------------

def impute_mean(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    df = df.copy()
    for c in cols:
        try:
            df[c] = df[c].fillna(df[c].mean())
        except Exception as e:
            logger.warning("impute_mean failed on %s: %s", c, e)
    return df

def impute_median(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    df = df.copy()
    for c in cols:
        try:
            df[c] = df[c].fillna(df[c].median())
        except Exception as e:
            logger.warning("impute_median failed on %s: %s", c, e)
    return df

def impute_mode(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    cols = columns or df.columns.tolist()
    df = df.copy()
    for c in cols:
        try:
            modes = df[c].mode()
            if len(modes) > 0:
                df[c] = df[c].fillna(modes.iloc[0])
        except Exception as e:
            logger.warning("impute_mode failed on %s: %s", c, e)
    return df

def drop_missing(df: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
    # axis=0 drop rows, axis=1 drop columns
    return df.dropna(axis=axis)

# -------------------------
# Cleaning utils
# -------------------------
def strip_whitespace(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    df = df.copy()
    cols = columns or df.select_dtypes(include=['object', 'category']).columns.tolist()
    for c in cols:
        try:
            df[c] = df[c].astype(str).str.strip()
        except Exception as e:
            logger.warning("strip_whitespace failed on %s: %s", c, e)
    return df

def lower_case(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    df = df.copy()
    cols = columns or df.select_dtypes(include=['object', 'category']).columns.tolist()
    for c in cols:
        try:
            df[c] = df[c].astype(str).str.lower()
        except Exception as e:
            logger.warning("lower_case failed on %s: %s", c, e)
    return df

# -------------------------
# Encoders & scalers
# -------------------------
def label_encode_columns(df: pd.DataFrame, columns: List[str], save_meta: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies LabelEncoder to each column listed and stores classes in save_meta.
    Returns transformed df and meta updates.
    """
    df = df.copy()
    meta = {}
    for c in columns:
        try:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str).fillna("##MISSING##"))
            meta[c] = {"type": "label", "classes": le.classes_.tolist()}
        except Exception as e:
            logger.warning("label_encode_columns failed on %s: %s", c, e)
    save_meta.update(meta)
    return df, save_meta

def onehot_encode_columns(df: pd.DataFrame, columns: List[str], save_meta: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    OneHotEncode specified columns (drop='first' to avoid collinearity).
    Returns df with new columns and meta listing the produced feature names.
    """
    df = df.copy()
    produced = []
    try:
        ohe = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
        arr = ohe.fit_transform(df[columns].astype(str).fillna("##MISSING##"))
        out_cols = ohe.get_feature_names_out(columns).tolist()
        produced = out_cols
        df = df.drop(columns=columns)
        df_out = pd.DataFrame(arr, columns=out_cols, index=df.index)
        df = pd.concat([df, df_out], axis=1)
        save_meta["onehot"] = {"columns": columns, "produced": out_cols}
    except Exception as e:
        # fallback: keep original columns and record warning
        logger.warning("onehot_encode_columns failed for %s: %s", columns, e)
        save_meta.setdefault("warnings", []).append(f"onehot failed for {columns}: {str(e)}")
    return df, save_meta

def apply_scaler(df: pd.DataFrame, columns: List[str], method: str = "standard", save_meta: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    method: 'standard' | 'minmax' | 'passthrough'
    Stores scaler params in save_meta if provided.
    """
    df = df.copy()
    meta = {}
    if method == "passthrough":
        return df, save_meta or {}
    try:
        if method == "standard":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        arr = scaler.fit_transform(df[columns].astype(float))
        df[columns] = arr
        if save_meta is not None:
            meta = {"scaler": {"method": method}}
            save_meta.update(meta)
    except Exception as e:
        logger.warning("apply_scaler failed for %s: %s", columns, e)
        if save_meta is not None:
            save_meta.setdefault("warnings", []).append(f"scaler failed for {columns}: {str(e)}")
    return df, save_meta or {}

# -------------------------
# Operation registry (name -> function)
# -------------------------
OP_REGISTRY = {
    # imputers
    "impute_mean": impute_mean,
    "impute_median": impute_median,
    "impute_mode": impute_mode,
    "drop_missing": drop_missing,
    # cleaning
    "strip_whitespace": strip_whitespace,
    "lower_case": lower_case,
    # encoding/scaling are special-cased (need meta)
    "label_encode": label_encode_columns,
    "onehot_encode": onehot_encode_columns,
    "scale": apply_scaler,
}

def list_ops() -> List[str]:
    return sorted(list(OP_REGISTRY.keys()))
