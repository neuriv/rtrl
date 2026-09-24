"""Reuse complete initial greedy outputs, rescoring with the current verifier."""

import hashlib
import json
import math
from pathlib import Path


def load_initial_evaluation(path, config, rows, scorer):
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    records = (json.loads(line) for line in raw.splitlines() if line.strip())
    manifest = next(records, {})
    if manifest.get("type") != "manifest":
        raise ValueError("Initial evaluation source requires a leading manifest")
    for key in ("model", "revision", "eval_sha256", "max_tokens", "precision", "versions", "attention_backend"):
        if key not in config or key not in manifest or config[key] != manifest[key]:
            raise ValueError(f"Initial evaluation configuration differs: {key}")
    evaluation = next((record for record in records if record.get("type") == "eval"), None)
    if (evaluation is None or evaluation.get("optimizer_steps") != 0
            or evaluation.get("training_s") != 0):
        raise ValueError("Source must contain an initial evaluation before any training")
    samples = evaluation.get("results", [])
    expected = [row["id"] for row in rows]
    found = [sample.get("id") for sample in samples]
    if (not expected or len(set(expected)) != len(expected) or len(found) != len(expected)
            or len(set(found)) != len(found) or set(found) != set(expected)):
        raise ValueError("Initial evaluation must contain every held-out ID exactly once")
    by_id = {sample["id"]: sample for sample in samples}
    results = []
    for row in rows:
        sample = dict(by_id[row["id"]])
        if (not isinstance(sample.get("text"), str)
                or sample.get("finish_reason") not in ("stop", "length")):
            raise ValueError("Initial evaluation requires complete text and finish reasons")
        sample["reward"] = float(scorer(row, sample))
        if not math.isfinite(sample["reward"]):
            raise ValueError("Initial evaluation scorer returned a nonfinite reward")
        results.append(sample)
    source_s = evaluation.get("source_evaluation_s", evaluation.get("evaluation_s"))
    if not isinstance(source_s, (int, float)) or not math.isfinite(source_s) or source_s < 0:
        raise ValueError("Initial evaluation requires a finite source duration")
    return {"accuracy": sum(sample["reward"] for sample in results) / len(results),
            "truncation_fraction": sum(sample["finish_reason"] == "length" for sample in results) / len(results),
            "evaluation_s": 0.0, "source_evaluation_s": source_s, "results": results,
            "initial_evaluation_source": {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                "code_commit": manifest.get("code_commit"), "run_url": manifest.get("run_url")}}
