"""Re-derive labels on existing data from each record's stored latent theta, under
the *current* personas calibration — no LLM calls. The transcripts depend only on
the theta *bands* (which don't change with the weights), so re-labeling yields a
valid dataset under a recalibrated generator without regenerating any calls.

    python -m ml.relabel data/raw/train.jsonl data/raw/test.jsonl data/raw/ood.jsonl
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .config import GLOBAL_SEED
from .personas import sample_label

def relabel_records(records, rng):
    for r in records:
        r["label"] = sample_label(r["theta"], rng)
    return records

def _read(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]

def _write(path, records):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                          encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", help="JSONL files to re-label in place")
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    for f in args.files:
        recs = relabel_records(_read(f), rng)
        _write(f, recs)
        pos = sum(r["label"] for r in recs)
        print(f"relabeled {len(recs)} (pos {pos}) -> {f}")

if __name__ == "__main__":
    main()
