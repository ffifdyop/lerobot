# Validation

- Ran `check` with the existing `/data/DualDirection-RSI/.venv-lerobot/bin/python` interpreter
  and this repository's source via PYTHONPATH. No environment sync was performed.
- Checked all 17,281 frames across 22 episodes: zero blank subtasks or task prompts.
- Used all 2,732 curation-excluded frames and applied no interval/episode exclusions.
- Decoded 132 camera frames from episode boundaries and checked 50-by-12 action chunks.
- Preserved checkpoint FAST offset 128; adapted state/action to 12 dimensions and view order
  to overhead, left wrist, right wrist. The native JSON round-trip and actual `@parser.wrap`
  configuration-file path both preserved features, pretrained source, recipe, and full-data settings.
- A temporary one-frame dataset with an empty subtask failed coverage validation before decoding.
  An existing native training directory was refused. Check mode created no native train/ directory.
- Bash syntax, repository-configured Ruff checks/formatting, `git diff --check`, and
  `openspec validate add-fineart-so101-full-data-training --strict --no-interactive` passed.
- Outputs: `outputs/fineart_so101/preflight_verified/{train_config,preflight}.json` (ignored artifacts).
- Not performed: environment setup, tokenizer fitting, model weight loading, GPU forward/backward,
  smoke training, formal training, or robot evaluation. GPU memory feasibility and learning quality
  are therefore not established by these checks.
- Added the W&B extra with an offline lock refresh, preserving every existing package version.
  A pre-update lock backup is `/tmp/lerobot-fineart-uv-before-wandb.lock`.

## Dual DDP and W&B revision

- Read-only GPU inventory confirmed GPU 0 and GPU 2 are the two 96 GB RTX PRO 6000 cards.
- The new preflight report is `outputs/fineart_so101/preflight_ddp_wandb/preflight.json`.
  It records DDP with two workers, batch 4 per GPU/global 8, W&B online configuration, and
  all 17,281 frames. No W&B run is initialized by check mode.
- W&B logger, existing Trackio/legacy-migration, and distributed config tests: 64 passed
  (mocked W&B, no network logging).
- Native CLI parsing resolved the new config to DDP=2, per-rank batch=4/global batch=8,
  online W&B with a saved run ID, SO101 features, and no data exclusions.
- `uv lock --check --offline` passed. Comparing old/new package identity/version/source sets
  confirmed no package versions changed.
- The full data preflight still decoded 132 camera frames; no tokenizer fitting or GPU training ran.
- Ruff checking uses `--ignore UP042` on the touched source set because the existing
  `CheckpointFormat(str, Enum)` predates this change; that unrelated enum is unchanged.

## W&B reference configuration revision (2026-11-05)

- Located the supplied reference at
  `/home/gsp/Dev/Manipulation/lerobot/examples/pi05/train_so101_bimanual_folding_cloth.sh`.
  Its current version uses Trackio; read its prior W&B settings from Git commit `6c8e56fe`.
- Reused entity `2943872339-zhejiang-university`; retained project `fineart-so101` and
  job name `fineart_so101_full_<mode>`. Explicitly set `disable_artifact=True` in the example.
  No credentials were copied into the repository.
- Re-ran the full data check: 17,281 frames, zero blank labels, all 2,732 excluded-marker
  frames included, 132 decoded camera frames. Output:
  `outputs/fineart_so101/preflight_wandb_reference/{train_config,preflight}.json`.
- Verified the generated configuration through the native `@parser.wrap()` CLI path:
  reference entity, this project's project/job, DDP=2, and artifacts disabled survive parsing.
- All 11 W&B logger tests passed with a mocked SDK. Bash syntax, Ruff check/format,
  `git diff --check`, and strict OpenSpec validation passed.
- No training, FAST fitting, model loading, or live W&B initialization was performed.

## Editable training parameters and epoch budget (2026-11-05)

- Added one editable shell parameter block, environment overrides, and CLI overrides for batch,
  LR/end LR/warmup, epochs/steps, train ratio, workers/seed, and save/log/eval settings.
- Default `check` on the actual dataset resolved five epochs to 10,805 steps (2,161 per epoch,
  17,281 training frames, batch 4 per rank, no held-out episodes).
- A real check with train ratio 0.8 reused native per-task episode holdout: 16 training episodes /
  12,951 frames and 6 eval episodes / 4,330 frames. Sets are disjoint and cover all source frames.
  Batch 2 resolved five epochs to 16,190 steps; batch 4 resolved to 8,095 steps.
- Conflicting environment/CLI batch, LR, ratio, and epoch values verified that CLI overrides win.
  The resulting native `@parser.wrap()` config round-trip retained batch 4, LR 1e-5, 8,095 steps,
  decay horizon 8,095, warmup 100, eval every 1,000 steps, and W&B artifact uploads disabled.
- All three actual checks still verified full annotation coverage and decoded 132 camera frames.
  Reports are under `outputs/fineart_so101/preflight_editable_{defaults,split,cli}/`.
- Added 23 passing CPU tests, including actual Accelerate batch-sharding lengths, explicit-step
  precedence, invalid parameter rejection, native episode holdout integration, and forwarding
  only training episodes to the mocked FAST fitting resolver. No tokenizer was actually fitted.
- Normalization remains the native full-source `meta/stats.json`; the report and guide document it.
- No native `train/` directory was created; no model loading, GPU training, or W&B run was started.

## Trainability, LoRA, and explicit LR controls (2026-11-05)

- Exposed gradient checkpointing, native PEFT LoRA with rank/alpha, whole-VLM freezing,
  vision-only freezing, and explicit MAX_LR/FINAL_LR with LR/MIN_LR compatibility aliases.
- Frozen VLM selects native train_expert_only, text/FAST weights zero, no tokenizer fitting,
  and a 100% low-level subtask-conditioned action recipe. Native FineART config validation
  accepts this combination; its text/FAST incompatibility guard remains unchanged.
- Real checks generated `preflight_controls_lora` (GC off, vision frozen, LoRA, warmup 200,
  peak 2e-5/final 2e-6) and `preflight_controls_frozen` (GC on, frozen VLM, expert LoRA,
  warmup 50, peak 3e-5/final 1e-6). Both preserve 10,805 steps and passed native CLI round-trip.
- Read safetensors keys only, without loading weights: language+expert regex matches 166
  Linear modules (126 VLM language + 40 action side); expert-only regex matches 40.
  Neither matches vision or LM-head modules.
- Recipe CPU suite: 36 passed, covering native config invariants, adapter scope, skipping FAST
  fitting when disabled, and real warmup/cosine scheduler initial/final rates (warmup 0 and 100).
- Explicit setup now includes the existing peft extra; setup was not run. The borrowed audit
  interpreter has no peft package, so adapter injection, checkpoint loading, GPU forward/backward,
  and training remain unvalidated. No model weights were loaded and no W&B run was initialized.
- Bash syntax, Ruff check/format, diff whitespace, and strict OpenSpec validation passed.

## Native dual-GPU FSDP2 (2026-11-05)

- Added the native backend selector (default fsdp, optional ddp), reshard-after-forward, and CPU
  offload controls. FSDP2 uses dp_replicate=1/dp_shard=2 with explicit SiglipEncoderLayer units;
  the joint decoder/expert/LM head remain in the root unit. No custom sharding runtime was added.
- Preserved all previous shell assignments, including the user's batch 12, 4000 steps, peak/final
  LR 4e-5/4e-6, and disabled gradient checkpointing. A before/after diff of
  `/tmp/train_fineart_so101.before-fsdp.sh` shows only added backend controls/arguments.
- Real full-data checks: `preflight_fsdp_defaults`, `preflight_fsdp_ddp_fallback`, and
  `preflight_fsdp_frozen_lora`, under `outputs/fineart_so101/`. Each checked 17,281 frames,
  zero blank subtasks/tasks, all 2,732 curation-excluded frames, and 132 decoded camera frames.
  Global batch is 24 and each epoch is 721 steps; 4000 steps is about 5.55 epochs.
- The third configuration combines FSDP2, LoRA, frozen VLM, policy gradient checkpointing,
  reshard-after-forward=false and CPU offload=true. Native validation accepts the configuration
  and keeps only low-level flow supervision. No adapters or model weights were actually loaded.
- All three configs passed the actual native `@parser.wrap()` config-file path and topology
  resolution for world size 2. Default FSDP2 plugin construction and class lookup were also
  exercised on CPU. DDP CLI selection overrode the shell's fsdp environment selection.
- 83 CPU tests passed across recipe, native parallelism/factory, and distributed train-config
  suites, including real Accelerate class selection on a small SigLIP layer and unchanged
  data-sharding epoch budgets in both backends. Bash syntax, Ruff check/format and whitespace
  validation passed.
- W&B retains the established entity/project, persisted run ID, and artifacts disabled. No
  live W&B run, native train/ directory, model loading, FAST fitting, or GPU training was started.
  FSDP2 forward/backward, save/resume, LoRA injection, and batch-12 memory feasibility remain
  unverified for this FineART recipe. The guide uses explicit --steps 20 for a future smoke run.

## FSDP group size and replica-group count (2026-11-05)

- Exposed DP_SHARD/--dp-shard (cards per group) and DP_REPLICATE/--dp-replicate (number of
  groups). Empty overrides preserve FSDP=2x1 and DDP=1x2 defaults. Positive degrees are
  required; FSDP requires shard >= 2 and DDP requires shard=1.
- Replaced the fixed two-worker count with their product for visible-device validation,
  torchrun local workers, CUDA checks, global batch, and epoch/frame-pass calculations.
  Missing/repeated GPUs or count mismatch fail before checkpoint and data loading.
- Full checks produced `outputs/fineart_so101/preflight_groups_defaults` (2x1) and
  `preflight_groups_hsdp_2x2` (2x2), retaining all 17,281 frames, zero blank labels,
  2,732 curation-excluded frames, and 132 decoded boundary-camera frames.
- The 2x2 check generated --nproc-per-node=4, batch 12 per GPU/global 48, and 361 steps per
  epoch. Default 2x1 remains global 24 and 721 steps per epoch. The user's explicit 4000-step
  override and 4e-5/4e-6 LR remain unchanged in both configurations.
- 125 CPU tests passed across recipe, parallelism/factory, and distributed train-config suites.
  Tests compare actual Accelerate batch-sharding budgets up to 8 workers, construct its native
  2x2 parallelism configuration, and reject conflicting degrees or invalid visible-device lists.
- This is single-node launch/config validation. No four-GPU allocation, model loading,
  tokenizer fitting, HSDP forward/backward, or W&B initialization was performed.

## Project environment dependency repair (2026-11-05)

- Reproduced missing datasets, av, accelerate, liger-kernel, wandb and peft in the project's
  Python 3.13.2 `.venv`. Earlier checks used an explicitly selected external audit interpreter.
- Ran the existing setup mode with UV_PROJECT_ENVIRONMENT=/data/lerobot-fineart/.venv,
  installing the locked training/fineart_vla/fineart_vla_kernels/wandb/peft extras. Network
  download required running outside the restricted network sandbox. No training was launched.
- Actual imports now pass: datasets 4.8.5, av 15.1.0, accelerate 1.15.0, transformers 5.5.4,
  liger-kernel 0.8.5, wandb 0.31.0 and peft 0.21.2. FineARTVLAPolicy import and native FSDP2
  plugin construction also pass. PyTorch remains 2.11.0+cu128 and uv.lock is byte-for-byte unchanged.
- The launcher now passes its full check using the project's own interpreter. Report:
  `outputs/fineart_so101/preflight_project_environment/{train_config,preflight}.json`.
  No weights, FAST fitting, GPU forward/backward, or live W&B initialization were performed.

## Launcher usage (2026-11-05)

- Added usage comments at the script header, including setup, default check, explicit train,
  smoke --steps 20, GPU/group/batch overrides, precedence, and the explicit STEPS override.
- Added --help/-h/help before dependency or environment setup. Bash syntax and both help flags
  pass; --help also works with PATH=/usr/bin:/bin, without uv on PATH. An unknown mode prints
  usage and exits 2 before any Python process or training launch.
- Training parameter defaults and default check behavior remain unchanged.
