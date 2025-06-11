#!/usr/bin/env python3
# coding: utf-8
"""
Generic CVE-severity (or description) experiment runner.
Drop this file in `exp_runner.py` and import `Experiment`.
"""

from __future__ import annotations
import json, os, shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Any, Dict

import pandas as pd
import numpy as np

# ─── optional metrics libs ────────────────────────────────────────────────
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
try:
    from sacrebleu.metrics import BLEU
except ImportError:  # optional
    BLEU = None


# ──────────────────────────────────────────────────────────────────────────
# metric helpers
# ──────────────────────────────────────────────────────────────────────────
def classification_metrics(
    y_true: List[str], y_pred: List[str], labels: List[str]
) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(acc),
        "per_class": {lbl: {"P": float(p[i]), "R": float(r[i]), "F1": float(f[i]), "n": int(s[i])}
                      for i, lbl in enumerate(labels)}
    }


def bleu_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    if BLEU is None:
        raise RuntimeError("pip install sacrebleu")
    bleu = BLEU()
    score = bleu.corpus_score(y_pred, [y_true]).score
    return {"BLEU": score}


# ──────────────────────────────────────────────────────────────────────────
#  experiment runner
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Experiment:
    df: pd.DataFrame
    model: Any                          # must have predict(list[str]) -> list[Any]
    target_col: str
    text_col: str
    metric_fn: Callable[[List[Any], List[Any]], Dict[str, Any]]
    run_name: str
    batch_size: int = 32
    out_dir: Path = Path("runs")
    preprocess_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    postprocess_fn: Callable[[List[Any]], List[Any]] | None = None
    _run_path: Path = field(init=False, repr=False)

    # ---------------------------------------------------------------------
    @classmethod
    def from_csv(
        cls, csv_path: Path, *args, **kwargs
    ) -> "Experiment":  # noqa: D403
        df = pd.read_csv(csv_path)
        return cls(df=df, *args, **kwargs)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, *args, **kwargs) -> "Experiment":
        return cls(df=df.copy(), *args, **kwargs)

    # ---------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._run_path = self.out_dir / f"{self.run_name}_{ts}"
        self._run_path.mkdir(parents=True, exist_ok=True)

        df_work = self.preprocess_fn(self.df.copy()) if self.preprocess_fn else self.df

        texts = df_work[self.text_col].astype(str).tolist()
        preds: List[Any] = []
        for i in range(0, len(texts), self.batch_size):
            preds.extend(self.model.predict(texts[i : i + self.batch_size]))

        if self.postprocess_fn:
            preds = self.postprocess_fn(preds)

        df_work["pred"] = preds
        metrics = self.metric_fn(df_work[self.target_col].tolist(), preds)

        # -------- persist --------
        df_work.to_parquet(self._run_path / "results.parquet")
        with open(self._run_path / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)

        print(f"✓ run saved to {self._run_path}")
        return metrics

    # ---------------------------------------------------------------------
    def clean_up(self) -> None:
        """Remove the run folder – handy when iterating."""
        if hasattr(self, "_run_path") and self._run_path.exists():
            shutil.rmtree(self._run_path)
            print(f"Deleted {self._run_path}")


# ──────────────────────────────────────────────────────────────────────────
#  example usage
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. fake dataframe
    demo = pd.DataFrame(
        {
            "text": ["dummy vuln text"] * 4,
            "target": ["HIGH", "LOW", "HIGH", "MEDIUM"],
        }
    )

    # 2. dummy model that echoes the first token → terrible on purpose
    class EchoModel:
        def predict(self, texts: List[str]) -> List[str]:
            return ["HIGH" for _ in texts]

    exp = Experiment.from_dataframe(
        demo,
        model=EchoModel(),
        target_col="target",
        text_col="text",
        metric_fn=lambda y, yhat: classification_metrics(y, yhat, labels=["LOW","MEDIUM","HIGH","CRITICAL"]),
        run_name="echo_demo",
    )
    print(exp.run())
