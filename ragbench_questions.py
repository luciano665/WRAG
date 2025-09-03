from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from typing import Iterable, List, Optional, Sequence, Tuple


# Auto install hook if not already installed
def _ensure_datasets_installed(auto_install: bool):
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
    include_response: bool = False,
    auto_install: bool = False,
    limit: Optional[int] = None,
) -> List[str]:
    """
    Return a flat list of questions across requested subsets & splits.

    Args:
        subsets: list like ["hotpotqa"]. If None, uses DEFAULT_SUBSETS.
        splits: iterable of split names; default: ("train","validation","test")
        include_ids: if True -> JSON strings with {"id", "question"}. Else raw question strings.
        auto_install: if True, tries `pip install datasets` if missing.
    
    If include_response=True:
        - Returns JSON strings with at least {"question","response"}.
        - If include_ids=True and 'id' exists -> include "id".
        - Skips rows/splits missing the 'response' column (since needed for eval).
    """
    _ensure_datasets_installed(auto_install=auto_install)
    hf_token = _get_hf_token()

    if subsets is None:
        subsets = DEFAULT_SUBSETS

    out: List[str] = []
    count = 0
    for subset in subsets:
        for split in splits:
            if limit is not None and count >= limit:
                break
            try:
                ds = _load_with_owner_fallback(subset, split, hf_token)
            except Exception:
                continue

            if "question" not in ds.column_names:
                continue
            if include_response:
                if "response" not in ds.column_names:
                    # If response need for eval, skip this split
                    continue
                have_id = include_ids and ("id" in ds.column_names)
                for i in range(len(ds)):
                    if limit is not None and count >= limit:
                        break
                    record = {
                        "question": ds["question"][i],
                        "response": ds["response"][i],
                    }
                    if have_id:
                        record["id"] = ds["id"][i]
                    out.append(json.dumps(record, ensure_ascii=False))
                    count += 1
            else:
                if include_ids:
                    have_id = "id" in ds.column_names
                    ids = ds["id"] if have_id else [None] * len(ds)
                    for i in range(len(ds)):
                        if limit is not None and count >= limit:   
                            break
                        out.append(json.dumps({"id": ids[i], "question": ds["question"][i]}, ensure_ascii=False))
                        count += 1
                else:
                    for q in ds["question"]:
                        if limit is not None and count >= limit:   
                            break
                        out.append(q)
                        count += 1

        if limit is not None and count >= limit:   
            break
    return out

def get_ragbench_questions_df(
    subsets: Optional[Sequence[str]] = None,
    splits: Sequence[str] = DEFAULT_SPLITS,
    auto_install: bool = False,
):
    """
    Return a pandas.DataFrame with columns [subset, split, id, question].
    """
    _ensure_datasets_installed(auto_install=auto_install)
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError(
            "`pandas` is required for get_ragbench_questions_df. Install via: pip install pandas"
        ) from e
    
    hf_token = _get_hf_token()
    if subsets is None:
        subsets = DEFAULT_SUBSETS
    
    rows = []
    for subset in subsets:
        for split in splits:
            try:
                ds = _load_with_owner_fallback(subset, split, hf_token)
            except Exception:
                continue
            if "question" not in ds.column_names:
                continue

            has_id = "id" in ds.column_names
            has_resp = "response" in ds.column_names
            n = len(ds)

            for i in range(n):
                row = {
                    "subset": subset,
                    "split": split,
                    "id": ds["id"][i] if has_id else None,
                    "question": ds["question"][i],
                }
                if has_resp:
                    row["response"] = ds["response"][i]
                rows.append(row)
    # Coluns includes response if any row had it
    columns = ["subset", "split", "id", "question"]
    if any("response" in r for r in rows):
        columns.append("response")
    return pd.DataFrame(rows, columns=columns)

# CLI
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export RAGBench questions")
    p.add_argument("--subset", action="append", help="Subset name (repeatable), e.g. --subset hotpotqa")
    p.add_argument("--split", action="append", help="Split(s) to include (repeatable). Default: train,validation,test")
    p.add_argument("--include-ids", action="store_true", help="Emit JSONL {id, question} lines")
    p.add_argument("--include-response", action="store_true", help="Emit JSONL with {'question','response',('id')}")
    p.add_argument("--out", type=str, default="-", help="Output file path or '-' for stdout (default)")
    p.add_argument("--limit", type=int, default=None, help="Maximum number of rows to emit")
    p.add_argument("--auto-install", action="store_true", help="Auto-install `datasets` if missing")
    return p.parse_args()

def main_cli():
    args = _parse_args()
    splits = tuple(args.split) if args.split else DEFAULT_SPLITS
    subsets = args.subset or None

    lines = get_ragbench_questions(
        subsets=subsets,
        splits=splits,
        include_ids=args.include_ids,
        include_response=args.include_response,
        auto_install=args.auto_install,
        limit=args.limit,
    )

    if args.out == "-":
        for line in lines:
            print(line if isinstance(line, str) else str(line))
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            for line in lines:
                f.write((line if isinstance(line, str) else str(line)) + "\n")
        print(f"Wrote {len(lines)} lines to {args.out}")

if __name__ == "__main__":
    main_cli()