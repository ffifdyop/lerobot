## Decisions

Use native TrainPipelineConfig and lerobot_train instead of a second training loop. Load the
checkpoint config before replacing input/output features with dataset-derived SO101 features.
Retain the checkpoint's FAST offset and use its bundled tokenizer as the fitting source.
Rebuild recipe processors with source dataset statistics. Use the existing 30/70 subtask recipe.

The default check mode performs CPU data/config validation and serializes artifacts separately
from the future native train/ directory. Smoke/train first prepare FAST in the parent, disable
worker auto-fitting, and delegate to two native workers via torch.distributed.run.
No interval markers or stale readiness flags control sampling. By default all episodes train;
an explicitly requested train ratio delegates the episode holdout to the native dataset factory.
The preflight report explicitly excludes claims about model loading, fitting, GPU loss, and success.

The launcher reuses the local uv environment without implicit sync; setup is a separate mode.
An explicitly selected existing interpreter is supported without changing its installation.

Default to FSDP2 with dp_replicate=1 and dp_shard=2 on GPU IDs 0,2; retain selectable
DDP with dp_replicate=2 and dp_shard=1. The current user-edited shell defaults are batch 12
per rank (global 24), 4000 steps, peak/final LR 4e-5/4e-6, and gradient checkpointing disabled;
preserve these settings when adding the backend selector. W&B is an opt-in native backend
(`wandb_logging`) so old `wandb` checkpoint sections
continue their existing Trackio migration. Only rank 0 creates the run, and native cross-rank
averaged metrics feed the logger. Persist run IDs in configs/checkpoints for online resume.
Explicitly disable model artifact uploads in the example. Reuse the PI05 folding recipe's W&B
entity `2943872339-zhejiang-university` (commit `6c8e56fe`, before its Trackio migration), while
keeping this project's `fineart-so101` project and `fineart_so101_full_<mode>` job name.
Use existing W&B authentication; never copy API keys into the recipe. Install W&B through its optional extra.

Keep common settings in one editable shell section. Command-line flags override environment
settings, which override shell defaults. When explicit steps are absent, default training/check
to five epochs derived from actual post-split training frames and even-batch padding in either
backend. Explicit steps take precedence, including the user's current 4000-step override;
document smoke with explicit --steps 20. Expose batch, LR/end LR/warmup, epochs/steps, train ratio,
workers, seed, and save/log/eval frequencies. Reuse native per-task episode tail holdouts, and
report requested versus actual proportions and both episode lists. Fit FAST on training episodes
only, so the existing episode-aware tokenizer signature separates split caches. Preserve native
source-dataset normalization statistics and document that they include the held-out episodes.

Map gradient checkpointing and vision freezing to native policy flags. Map whole-VLM freezing
to train_expert_only, explicitly zero text/FAST losses, disable tokenizer fitting, and use only
the low-level action recipe so no high-level-only batches have an empty loss. LoRA delegates
to native PeftConfig/wrap_with_peft; select language+expert targets unless VLM freezing requests
expert-only targets. Exclude vision/multimodal/LM-head targets, document that base parameters
are frozen by PEFT, and expose rank/alpha. Add peft to explicit setup only.
Name the existing native LR controls MAX_LR, FINAL_LR and WARMUP_STEPS, retain LR/MIN_LR
aliases, and preserve the native scheduler's warmup start and global-step cosine semantics.

Use only native FSDP2 configuration. Explicitly wrap SiglipEncoderLayer children, keeping the
joint Gemma decoder, expert, and LM head in the root FSDP unit because joint attention accesses
decoder internals without decoder.forward hooks. Default reshard_after_forward=true and
cpu_offload=false, exposing both controls. Keep policy-level gradient checkpointing; do not use
the unimplemented accelerator activation-checkpointing field. Keep native safetensors model
checkpoints with DCP optimizer state. Root gathers and saves can still create memory peaks.
Record backend, topology, wrap classes, resharding, offload, and checkpoint format in preflight.

Expose native dp_shard (GPUs per group) and dp_replicate (number of groups) as shell/CLI
controls. Defaults remain 2x1 for FSDP and 1x2 for DDP. Two nontrivial dimensions select HSDP
through the existing native FSDP2 runtime. Require positive degrees and an exact distinct visible
GPU count equal to their product before loading data. Derive torchrun local workers, global
batch, frame passes, and epoch budgets from that product. Keep this launcher single-node.

## Validation

Run the actual check against the supplied dataset and cached checkpoint. Round-trip its JSON
through native config parsing, verify checkpoint/feature/sampling settings, and exercise a
missing-subtask failure. Validate shell/Python syntax, formatting, and this OpenSpec change.
Do not run setup, fit a tokenizer, or launch smoke/training for this request.
