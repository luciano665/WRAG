from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from typing import Iterable, List, Optional, Sequence, Tuple


# Auto install hook if not already installed
def _ensure_datsets_installed(auto_install: bool):
    try:
        import datasets
    except Exception as e:
        if not auto_install:
            raise RuntimeError(
                "Python package `datasets` not found.\n"
                "Install with: pip install datasets\n"
                "Or call get_ragbench_questions(..., auto_install=True)."
            ) from e
        print("[ragbench_questions] Installing `datasets` ...", file=sys.stderr)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        import datasets

def _load_dotenv_if_available():
    """Load .env if python-dotenv is present; otherwise ignore silently."""
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        # python-dotenv not installed or failed — environment may already have the vars
        pass

def _get_hf_token() -> Optional[str]:
    """
    Read token from env (after .env load).
    """
    _load_dotenv_if_available()
    token = os.getenv("HF_TOKEN")
    return token

DEFAULT_SUBSETS: Tuple[str, ...] = (
    "covidqa",
    "cuad",
    "delucionqa",
    "emanual",
    "expertqa",
    "finqa",
    "hagrid",
    "hotpotqa",
    "msmarco",
    "pubmedqa",
    "tatqa",
    "techqa",
)
DEFAULT_SPLITS: Tuple[str, ...] = ("train", "validation", "test")


def _try_load(dataset_name: str, subset: str, split: str, hf_token: Optional[str]):
    from datasets import load_dataset
    if hf_token:
        return load_dataset(dataset_name, subset, split=split, token=hf_token)
    return load_dataset(dataset_name, subset, split=split)

def _load_with_owner_fallback(subset: str, split: str, hf_token: Optional[str]):
    for ds_name in ("galileo-ai/ragbench", "rungalileo/ragbench"):
        try:
            return _try_load(ds_name, subset, split, hf_token)
        except Exception:
            continue
    raise RuntimeError(
        f"Could not load RAGBench for subset='{subset}', split='{split}' "
        f"from either galileo-ai/ragbench or rungalileo/ragbench."
    )

def get_ragbench_questions(
    subsets: Optional[Sequence[str]] = None,
    splits: Sequence[str] = DEFAULT_SPLITS,
    include_ids: bool = False,
    auto_install: bool = False,
) -> List[str]:
    """
    Return a flat list of questions across requested subsets & splits.

    Args:
        subsets: list like ["hotpotqa"]. If None, uses DEFAULT_SUBSETS.
        splits: iterable of split names; default: ("train","validation","test")
        include_ids: if True -> JSON strings with {"id", "question"}. Else raw question strings.
        auto_install: if True, tries `pip install datasets` if missing.
    """
    _ensure_datsets_installed(auto_install=auto_install)
    hf_token = _get_hf_token()

    if subsets is None:
        subsets = DEFAULT_SUBSETS

    out: List[str] = []
    for subset in subsets:
        for split in splits:
            try:
                ds = _load_with_owner_fallback(subset, split, hf_token)
            except Exception:
                continue

            if "question" not in ds.column_names:
                continue

            if include_ids:
                ids = ds["id"] if "id" in ds.column_names else [None] * len(ds)
                for _id, q in zip(ids, ds["question"]):
                    out.append(json.dumps({"id": _id, "question": q}, ensure_ascii=False))
            else:
                out.extend(ds["question"])
    return out