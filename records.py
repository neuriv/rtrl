"""One manifest, complete group records, then a completion marker; JSONL on disk."""

import hashlib
import json
import math
from pathlib import Path


def write_record(handle, record):
    handle.write(json.dumps(record, allow_nan=False) + "\n")
    handle.flush()


def external_output(path):
    path = Path(path).expanduser().resolve()
    for base in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for parent in (base, *base.parents):
            if (parent / ".git").exists():
                if path.is_relative_to(parent):
                    raise ValueError("Write traces/reports outside the Git repository")
                break
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("x")


def read_prompts(path):
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Prompt IDs must be unique; input must be nonempty")
    for row in rows:
        if not isinstance(row["id"], str) or not isinstance(row["prompt"], str) or not row["prompt"]:
            raise ValueError("Each prompt needs a string id and nonempty prompt text")
    return rows, hashlib.sha256(raw).hexdigest()


def validate_groups(groups):
    """Reject silent truncation, mixed prompt tokens, and fabricated retry gaps."""
    seen, trials, prompts = set(), {}, {}
    for group in groups:
        key = (group["prompt_id"], group["trial"])
        if group["id"] in seen or group["attempt"] < 0 or group["trial"] < 0:
            raise ValueError("Duplicate group ID or invalid trial/attempt")
        seen.add(group["id"])
        trials.setdefault(key, []).append(group)
        original = prompts.setdefault(group["prompt_id"], group["prompt_token_ids"])
        if original != group["prompt_token_ids"]:
            raise ValueError("The same prompt ID must preserve its tokens across every trial")
        if not group["prompt_token_ids"] or len(group["samples"]) < 2:
            raise ValueError("Groups need prompt tokens and at least two siblings")
        for sample in group["samples"]:
            for clock in ("generation_s", "ready_s"):
                if not math.isfinite(sample[clock]) or sample[clock] < 0:
                    raise ValueError("Completion times must be finite and nonnegative")
            if sample["ready_s"] < sample["generation_s"]:
                raise ValueError("Reward readiness cannot precede generation")
            if len(sample["token_ids"]) != len(sample["logprobs"]):
                raise ValueError("Token IDs and behavior log probabilities must align")
            if not all(math.isfinite(x) for x in sample["logprobs"]):
                raise ValueError("Nonfinite behavior log probability")
            if sample["status"] == "complete":
                if not sample["token_ids"] or not isinstance(sample["reward"], (float, int)) or not math.isfinite(sample["reward"]):
                    raise ValueError("Complete responses need tokens and a finite reward")
            elif sample["status"] not in ("error", "truncated") or sample["reward"] is not None:
                raise ValueError("Failed/truncated responses have no task reward")
    for siblings in trials.values():
        if sorted(g["attempt"] for g in siblings) != list(range(len(siblings))):
            raise ValueError("Retry attempts must be unique and contiguous from zero")
        first = siblings[0]
        if any(g["prompt_token_ids"] != first["prompt_token_ids"] or len(g["samples"]) != len(first["samples"]) for g in siblings):
            raise ValueError("Retries must preserve prompt tokens and group size")
    return trials


def read_run(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        records = []
        for line in handle:
            digest.update(line)
            if line.strip():
                records.append(json.loads(line))
    if len(records) < 3 or records[0].get("type") != "manifest" or records[-1].get("type") != "end":
        raise ValueError("Incomplete run: manifest and final completion marker required")
    manifest, groups = records[0], records[1:-1]
    if manifest["schema"] != 1 or any(g.get("type") != "group" for g in groups):
        raise ValueError("Unsupported trace schema")
    if len(groups) != manifest["expected_groups"] or records[-1]["groups"] != len(groups):
        raise ValueError("Incomplete group collection")
    trials = validate_groups(groups)
    expected = {(p, t) for p in manifest["prompt_ids"] for t in range(manifest["trials"])}
    if set(trials) != expected or any(len(gs) != manifest["attempts"] for gs in trials.values()):
        raise ValueError("Missing prompt/trial/attempt coverage")
    if any(len(g["samples"]) != manifest["group_size"] for g in groups):
        raise ValueError("Group size differs from the manifest")
    allowed_finishes = {"stop", "length"} if manifest["accept_length"] else {"stop"}
    for group in groups:
        for sample in group["samples"]:
            if sample["status"] == "complete" and sample["finish_reason"] not in allowed_finishes:
                raise ValueError("Completed response has a finish reason excluded by the task budget")
    manifest["trace_sha256"] = digest.hexdigest()
    return manifest, groups
