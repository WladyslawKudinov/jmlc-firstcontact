import argparse
import json
import os
import uuid
from pathlib import Path

import numpy as np

from .config import load_scenarios, RAW, GLOBAL_SEED, field_ids
from .personas import sample_latent, sample_label, behavior_brief
from .llm import build_messages, parse_generation, make_complete_fn, LLMError

def transcript_text(transcript):
    return "\n".join(
        f"{'Лид' if t['role'] == 'user' else 'Софья'}: {t['text']}" for t in transcript
    )

def make_record(scenario, theta, label, transcript, fields, generator):
    return {
        "call_id": str(uuid.uuid4()),
        "scenario_id": scenario["id"],
        "transcript": transcript,
        "transcript_text": transcript_text(transcript),
        "fields": fields,
        "theta": theta,
        "label": int(label),
        "generator": generator,
    }

def _generate_one(scenario, rng, complete_fn, temperature, generator):
    theta = sample_latent(rng)
    label = sample_label(theta, rng)
    system, user = build_messages(scenario, behavior_brief(theta))
    raw = complete_fn(system=system, user=user, temperature=temperature)
    transcript, fields = parse_generation(raw, field_ids(scenario))
    return make_record(scenario, theta, label, transcript, fields, generator)

def generate_dataset(scenarios, n, rng, complete_fn, temperature, generator):
    return [_generate_one(scenarios[i % len(scenarios)], rng, complete_fn, temperature, generator)
            for i in range(n)]

def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _count_lines(path):
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

def _read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]

def generate_to_jsonl(scenarios, n, rng, complete_fn, temperature, generator, out_path,
                      *, resume=True, max_consecutive_skips=10, log=None):
    """Generate up to n records, appending each to out_path the moment it's produced.

    Resumable: with resume=True, rows already in out_path are counted and only the
    remainder is generated, so a crashed run re-runs cheaply and a completed run is
    idempotent. A record whose generation raises LLMError (retries already exhausted
    inside complete_fn, or an unparseable response) is skipped rather than aborting
    the whole batch. A *systemic* failure (bad model/key/endpoint failing every call)
    aborts after max_consecutive_skips consecutive failures instead of looping forever.
    Returns (made, skipped).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume and out_path.exists():
        out_path.unlink()
    made = _count_lines(out_path) if out_path.exists() else 0
    skipped = consecutive = 0
    log = log or (lambda msg: None)
    with out_path.open("a", encoding="utf-8") as f:
        while made < n:
            scenario = scenarios[(made + skipped) % len(scenarios)]
            try:
                rec = _generate_one(scenario, rng, complete_fn, temperature, generator)
            except LLMError as e:
                skipped += 1
                consecutive += 1
                log(f"[skip] generation failed ({skipped}): {e}")
                if consecutive >= max_consecutive_skips:
                    raise LLMError(
                        f"aborting after {consecutive} consecutive generation failures "
                        f"(likely a bad model/key/endpoint); last error: {e}"
                    ) from e
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            made += 1
            consecutive = 0
    return made, skipped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200, help="gen-A total (train+test)")
    ap.add_argument("--n-ood", type=int, default=300, help="gen-B (distribution shift)")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--out", default=str(RAW))
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore existing partial output and start fresh")
    args = ap.parse_args()

    out = Path(args.out)
    scenarios = load_scenarios()
    rng = np.random.default_rng(GLOBAL_SEED)
    resume = not args.no_resume

    # gen-A pool, written incrementally so a crash never loses progress (resumable)
    pool = out / "gen_a.jsonl"
    complete_a = make_complete_fn()
    made_a, skip_a = generate_to_jsonl(scenarios, args.n, rng, complete_a, 0.9, "A",
                                       pool, resume=resume, log=print)

    # gen-B OOD — optionally a different model (GEN_MODEL_OOD) for a stronger shift
    ood_model = os.environ.get("GEN_MODEL_OOD") or os.environ.get("GEN_MODEL")
    ood_temp = float(os.environ.get("GEN_TEMPERATURE_OOD", "1.1"))
    complete_b = make_complete_fn(model=ood_model)
    made_b, skip_b = generate_to_jsonl(scenarios, args.n_ood, rng, complete_b, ood_temp, "B",
                                       out / "ood.jsonl", resume=resume, log=print)

    # deterministic train/test split of the gen-A pool (cheap, re-runnable)
    a = _read_jsonl(pool)
    np.random.default_rng(GLOBAL_SEED).shuffle(a)
    n_test = int(len(a) * args.test_frac)
    _write_jsonl(out / "test.jsonl", a[:n_test])
    _write_jsonl(out / "train.jsonl", a[n_test:])

    print(f"gen_a={made_a} (skipped {skip_a}) -> train={len(a) - n_test} test={n_test} | "
          f"ood={made_b} (skipped {skip_b}) -> {out}")

if __name__ == "__main__":
    main()
