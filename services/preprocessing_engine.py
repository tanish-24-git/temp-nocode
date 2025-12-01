# services/preprocessing_engine.py
import os
import json
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import structlog
import traceback
from services.ops import tabular_ops

logger = structlog.get_logger()

DEFAULT_ORDER = ["cleaning", "impute", "encoding", "scaling", "feature_engineering"]

class PreprocessingEngine:
    def __init__(self, upload_dir: str):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def _atomic_write(self, df: pd.DataFrame, dest: Path):
        partial = dest.with_suffix(dest.suffix + ".partial")
        # write to temp then move
        df.to_csv(partial, index=False)
        partial.replace(dest)

    def _write_sidecar(self, dest: Path, sidecar: Dict[str, Any]):
        sidecar_path = dest.with_suffix(dest.suffix + ".preprocess.json")
        with open(sidecar_path, "w", encoding="utf-8") as fh:
            json.dump(sidecar, fh, indent=2)

    def _safe_read(self, file_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
        # robust read wrapper
        read_kwargs = {"encoding": "utf-8", "engine": "python", "on_bad_lines": "skip"}
        if nrows:
            read_kwargs["nrows"] = nrows
        return pd.read_csv(file_path, **read_kwargs)

    def run_tabular(self, file_path: str, plan: Dict[str, Any], provenance: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        run a conservative tabular plan.
        plan shape: global_defaults, columns (per-col overrides), order (optional), feature_engineering (optional)
        """
        provenance = provenance or {}
        t0 = time.time()
        warnings = []
        meta_store = {}
        try:
            df = self._safe_read(file_path)
            original_name = Path(file_path).name
            # combine defaults
            global_defaults = plan.get("global_defaults", {})
            col_plan = plan.get("columns", {}) or {}
            order = plan.get("order", DEFAULT_ORDER)

            # Step: cleaning (global & per-column)
            if "cleaning" in order:
                # apply typical lightweight cleaning like trim/lower if requested
                # global cleaning
                if global_defaults.get("strip_whitespace", True):
                    df = tabular_ops.strip_whitespace(df)
                if global_defaults.get("lower_case", False):
                    df = tabular_ops.lower_case(df)
                # per-column cleaning
                for col, cfg in col_plan.items():
                    if not isinstance(cfg, dict):
                        continue
                    if cfg.get("action") == "drop":
                        if col in df.columns:
                            df = df.drop(columns=[col])
                            warnings.append(f"dropped column {col} by plan")
                            meta_store.setdefault("dropped_columns", []).append(col)
                            continue
                    if cfg.get("strip_whitespace", False):
                        df = tabular_ops.strip_whitespace(df, [col])
                    if cfg.get("lower_case", False):
                        df = tabular_ops.lower_case(df, [col])

            # Step: imputation
            if "impute" in order:
                # global missing strategy
                missing = global_defaults.get("missing_strategy", "mean")
                # apply per-column overrides first
                for col, cfg in col_plan.items():
                    if isinstance(cfg, dict) and cfg.get("missing_strategy"):
                        strat = cfg.get("missing_strategy")
                        if strat == "mean":
                            df = tabular_ops.impute_mean(df, [col])
                        elif strat == "median":
                            df = tabular_ops.impute_median(df, [col])
                        elif strat == "mode":
                            df = tabular_ops.impute_mode(df, [col])
                        elif strat == "drop":
                            df = df.dropna(subset=[col])
                        else:
                            warnings.append(f"unknown impute strategy {strat} for column {col}")
                # then apply global to remaining missing numeric columns if not handled
                if missing == "mean":
                    df = tabular_ops.impute_mean(df)
                elif missing == "median":
                    df = tabular_ops.impute_median(df)
                elif missing == "mode":
                    df = tabular_ops.impute_mode(df)
                elif missing == "drop":
                    df = tabular_ops.drop_missing(df, axis=0)

            # Step: encoding
            # Collect columns that need label/onehot/passthrough
            label_cols = []
            onehot_cols = []
            # respect per-column flags, otherwise fallback to global encoding
            global_encoding = global_defaults.get("encoding", "onehot")
            for c in df.select_dtypes(include=['object', 'category']).columns:
                cfg = col_plan.get(c, {})
                enc = cfg.get("encoding", None) or global_encoding
                if enc == "label":
                    label_cols.append(c)
                elif enc == "onehot":
                    onehot_cols.append(c)
                elif enc in (None, "passthrough", "keep"):
                    # do nothing
                    continue
                else:
                    # unsupported encoding -> record
                    warnings.append(f"unsupported encoding {enc} for column {c}, skipping")

            # apply label encode (store classes)
            if label_cols:
                df, meta_store = tabular_ops.label_encode_columns(df, label_cols, meta_store)

            # apply onehot (may produce many columns)
            if onehot_cols:
                df, meta_store = tabular_ops.onehot_encode_columns(df, onehot_cols, meta_store)

            # Step: scaling
            if "scaling" in order:
                scaling_default = global_defaults.get("scaling", True)
                if scaling_default:
                    # choose scaler method if provided
                    scaler_method = global_defaults.get("scaler_method", "standard")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    # allow per-column override to disable scaling
                    cols_to_scale = []
                    for col in numeric_cols:
                        cfg = col_plan.get(col, {})
                        if cfg.get("scaling") is False:
                            continue
                        cols_to_scale.append(col)
                    if cols_to_scale:
                        df, meta_store = tabular_ops.apply_scaler(df, cols_to_scale, method=scaler_method, save_meta=meta_store)

            # Feature engineering placeholder (skipped for phase1 but reserve metadata)
            # if "feature_engineering" in order: ... (left for phase2)
            # finalize outputs
            preprocessed_name = f"preprocessed_{original_name}"
            preprocessed_path = self.upload_dir / preprocessed_name

            # atomic write
            self._atomic_write(df, preprocessed_path)

            # sidecar metadata
            sidecar = {
                "plan": plan,
                "created_at": time.time(),
                "duration_seconds": time.time() - t0,
                "warnings": warnings,
                "transformer_meta": meta_store,
                "rows": int(len(df)),
                "columns": list(df.columns),
                "provenance": provenance or {}
            }
            self._write_sidecar(preprocessed_path, sidecar)

            # preview
            preview = df.head(10).to_dict(orient="records")

            return {
                "preprocessed_path": str(preprocessed_path),
                "sidecar_path": str(preprocessed_path.with_suffix(preprocessed_path.suffix + ".preprocess.json")),
                "preview": preview,
                "meta": sidecar
            }

        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Preprocessing failed", error=str(e), traceback=tb)
            raise

