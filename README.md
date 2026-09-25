# RTRL

Single-H100 GRPO training and whole-group rollout-selection experiments. The shared harness lives on `main`; separate experiment PRs add executable JSON recipes. Research conclusions, generated data, plots, logs and checkpoints stay outside Git.

## Layout

- `rtrl/`: training, live rollout cancellation, rewards, frozen-policy collection/replay and analysis tools.
- `tests/`: CPU unit and protocol tests; no model download or GPU training required.
- `environment/`: separate pinned training and legacy vLLM environments.
- `experiments/`: versioned recipes contributed by experiment PRs.

## Environment

On Linux x86_64 with one H100, NVIDIA driver 570+ and Ubuntu 22.04 or newer:

```sh
git pull --ff-only
CUDA_VISIBLE_DEVICES=0 bash environment/setup_training.sh
.venv-training/bin/python -m rtrl train --help
```

The installer creates managed Python 3.12, installs the CUDA 12.8 lock, checks imports/device availability and runs CPU tests. It does not run a standalone model smoke experiment. W&B defaults to online; authenticate with `.venv-training/bin/wandb login` before online runs. Offline W&B with local plots or TensorBoard is also supported: use `--wandb-mode offline`, preserve its saved directory, and optionally sync later with `.venv-training/bin/wandb sync`. Offline logging preserves the same local training and evaluation records.

For CPU development, install editable with `python -m pip install -e '.[test,analysis]'`, then run `python -m pytest -q`. Training uses the exact `environment/requirements-training.txt` lock; CPU development is not a timing benchmark. The separate `environment/setup_h100.sh` and `requirements-h100.txt` retain the legacy vLLM collector and its stronger platform requirements; do not combine the two environments.

## Reproduction

Choose external directories and create one persistent experiment log. Every training launch reads this log and requires a clean, committed checkout.

```sh
mkdir -p ../rtrl-results
test -e ../rtrl-results/experiment-log.md || printf '%s\n' 'Question: does same-prompt whole-group replacement improve learning per training second?' > ../rtrl-results/experiment-log.md
.venv-training/bin/python -m rtrl.prepare_task \
  --output-dir ../rtrl-results/gsm8k --eval-size 128 --seed 2718 \
  --revision 3101c7d5072418e28b9008a6636bde82a006892c
```

Use a new output directory for each run. Run a baseline directly from `main`:

```sh
CUDA_VISIBLE_DEVICES=0 .venv-training/bin/python -m rtrl train \
  --train-prompts ../rtrl-results/gsm8k/train.jsonl \
  --eval-prompts ../rtrl-results/gsm8k/eval.jsonl \
  --output-dir ../rtrl-results/baseline-s17 \
  --experiment-log ../rtrl-results/experiment-log.md \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --mode baseline --seed 17 --seconds 1800 \
  --group-size 4 --groups-per-update 4 --max-tokens 512 \
  --learning-rate 1e-6 --eval-every 10 --eval-batch-size 16 \
  --wandb-mode offline
```

Choose `--wandb-mode online` for a live dashboard. For an experiment PR containing a recipe:

```sh
.venv-training/bin/python -m rtrl.experiment experiments/EXPERIMENT.json \
  --data-dir ../rtrl-results/gsm8k --output-dir ../rtrl-results/EXPERIMENT \
  --experiment-log ../rtrl-results/experiment-log.md --print-config
```

Remove `--print-config` to train. Dependent conditions also supply `--baseline ../rtrl-results/BASELINE`. A recipe contains `arguments`, a mapping of existing `rtrl train` option names to scalar values. Pin `model`, its immutable `revision`, seed and every comparative setting. Optional `deadline_quantile` (`p50`, `p80`, `p90`) uses the completed matching baseline and the existing calibration rule: exclude its first four groups, then take the nearest-rank timing quantile. Optional `reuse_initial_evaluation: true` reuses and validates the baseline's initial greedy outputs. Absolute machine paths belong in runtime arguments, not recipes. The resolved settings and recipe hash are recorded with each run.

Historical experiment recipes set `arguments.deadline` to the exact recorded cutoff. Use `deadline_quantile` instead when calibrating a new cutoff from a fresh matching baseline; these are distinct reproduction choices.

Common training defaults are four answers per prompt, four groups per update, 512 generated tokens, learning rate `1e-6`, evaluation every ten actual optimizer steps, FP32 parameters/AdamW state and BF16 forward computation. Training uses sequence-mean GRPO with one gradient pass and no KL. Token-capped answers receive zero reward. All-equal-reward batches skip optimizer updates. cuDNN SDPA is disabled to avoid observed runtime-compilation stalls; the other SDPA backends remain enabled.

Baseline lets groups finish. Deadline mode discards the entire original group at a token boundary and generates one independent same-prompt replacement with no deadline. Random mode finishes originals and makes replacement decisions independently of completion time. Cache settings remain identical. Training seconds include originals, discarded work, retries and optimization; evaluation/checkpoint time is reported separately. The last batch may overrun the time budget. Compare recorded curves and common optimizer counts; recorded evaluation points need not occur at identical elapsed times. Cached initial evaluation can change startup execution and should be matched or explicitly reported.

## Analysis and records

```sh
.venv-training/bin/python -m rtrl.analyze_training \
  --runs ../rtrl-results/BASELINE ../rtrl-results/EXPERIMENT \
  --output ../rtrl-results/comparison.json
```

Install the `analysis` extra to render the report or inspect local TensorBoard curves:

```sh
.venv-tools/uv pip install --python .venv-training/bin/python -e '.[analysis]'
.venv-training/bin/python -m rtrl.plot_training \
  --report ../rtrl-results/comparison.json --output-dir ../rtrl-results/plots
.venv-training/bin/python -m rtrl.export_tensorboard \
  --runs ../rtrl-results/BASELINE ../rtrl-results/EXPERIMENT \
  --output-dir ../rtrl-results/tensorboard
.venv-training/bin/tensorboard --logdir ../rtrl-results/tensorboard --host 127.0.0.1
```

Frozen-policy commands remain available through `python -m rtrl`: `collect`, `local`, `replay`, `audit`, `diagnose`, and `probe`. Built-in reward specifications now use `rtrl.rewards:FUNCTION`.

Before releasing a GPU host, copy and hash-verify run configurations, event records, environment/provenance files, the experiment log, W&B files, final checkpoints and useful matched-update checkpoints. Check that model/optimizer state can be loaded and that recorded steps match. Checkpoint references in W&B point to files on the originating host; references alone do not preserve those files.

Commit source, tests, environment locks and recipes only. Keep each experiment PR to one coherent recipe/config commit based on the shared foundation. Preserve the historical training commit in records; package reorganization does not retroactively change it. No result or improvement claim follows from a successful setup or test suite.
