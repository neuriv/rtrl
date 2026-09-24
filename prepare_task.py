"""Prepare immutable official GSM8K inputs outside Git; no generated task data."""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from records import external_output, write_record

REPOSITORY = "openai/grade-school-math"
PROMPT_SUFFIX = "\n\nShow concise reasoning. Put only the final integer answer on the final line."


def download(url):
    with urlopen(Request(url, headers={"User-Agent": "rtrl-task-preparation"}), timeout=60) as response:
        return response.read()


def final_answer(answer):
    """Only accept the official answer delimiter followed by a valid integer."""
    reasoning, separator, value = answer.rpartition("####")
    value = value.strip()
    if not separator or not re.fullmatch(r"-?(?:\d+|\d{1,3}(?:,\d{3})+)", value):
        raise ValueError(f"Invalid GSM8K final integer: {value!r}")
    return str(int(value.replace(",", "")))


def prepare_rows(train, test, eval_size, seed):
    if not 1 <= eval_size <= len(test):
        raise ValueError("eval-size must be between 1 and the official test split size")
    train_questions = {" ".join(row["question"].split()) for row in train}
    if any(" ".join(row["question"].split()) in train_questions for row in test):
        raise ValueError("Official train/test questions overlap")
    indices = {"train": list(range(len(train))),
               "eval": sorted(random.Random(seed).sample(range(len(test)), eval_size))}

    def convert(rows, split, chosen):
        return [{"id": f"gsm8k-{split}-{i}",
                 "prompt": rows[i]["question"].strip() + PROMPT_SUFFIX,
                 "reference": final_answer(rows[i]["answer"])} for i in chosen]

    return (convert(train, "train", indices["train"]),
            convert(test, "test", indices["eval"]), indices)


def prepare(output_dir, eval_size=128, seed=2718, revision="master"):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        revision = json.loads(download(
            f"https://api.github.com/repos/{REPOSITORY}/commits/{quote(revision, safe='')}"))["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Dataset revision must resolve to an immutable commit SHA")
    urls = {split: f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}/"
            f"grade_school_math/data/{split}.jsonl" for split in ("train", "test")}
    originals = {split: download(url) for split, url in urls.items()}
    source_rows = {split: [json.loads(line) for line in raw.splitlines() if line.strip()]
                   for split, raw in originals.items()}
    train, evaluation, indices = prepare_rows(
        source_rows["train"], source_rows["test"], eval_size, seed)
    output_dir = Path(output_dir).expanduser().resolve()
    manifest = {"dataset": "GSM8K", "repository": f"https://github.com/{REPOSITORY}",
                "revision": revision, "seed": seed, "eval_size": eval_size,
                "prompt_suffix": PROMPT_SUFFIX, "reward": "rewards:arithmetic",
                "row_indices_zero_based": indices, "sources": {}, "outputs": {}}
    for split, raw in originals.items():
        filename = f"original_{split}.jsonl"
        with external_output(output_dir / filename) as handle:
            handle.buffer.write(raw)
        manifest["sources"][split] = {"url": urls[split], "file": filename,
                                      "sha256": hashlib.sha256(raw).hexdigest(),
                                      "rows": len(source_rows[split])}
    for split, rows in (("train", train), ("eval", evaluation)):
        path = output_dir / f"{split}.jsonl"
        with external_output(path) as handle:
            for row in rows:
                write_record(handle, row)
        manifest["outputs"][split] = {"file": path.name, "rows": len(rows),
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with external_output(output_dir / "provenance.json") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--revision", default="master")
    args = parser.parse_args()
    manifest = prepare(**vars(args))
    print(json.dumps({"revision": manifest["revision"], "outputs": manifest["outputs"]}))


if __name__ == "__main__":
    main()
