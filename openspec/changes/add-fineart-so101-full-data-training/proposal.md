## Why

SO101 Flora language annotations now cover every frame, including intervals previously
marked for exclusion. The user wants a full-data FineART baseline without interval filtering.
The released checkpoint has ALOHA features and a FAST token layout that must survive adaptation.

## What Changes

- Add a shell launcher with setup, check, smoke, and train modes.
- Generate native training configurations from the checkpoint and actual SO101 features.
- Check actual language coverage and video decoding; record full-data usage and source hashes.
- Document conservative BF16 settings and the limits of preflight evidence.
- Launch two native workers on the two 96 GB GPUs (default IDs 0,2), with batch size per GPU.
- Default to native FSDP2 with a DDP fallback and expose resharding/CPU-offload controls.
- Expose FSDP shard-group size and replica-group count; derive single-node worker count and
  training budgets from their product and validate the selected GPU list.
- Add optional native W&B logging with a persisted run ID and a separate dependency extra.
- Fit FAST once in the parent before workers read the completed tokenizer artifacts.
- Expose editable batch/LR/epochs/train-ratio and logging/save/eval controls; default to five
  training epochs and reuse native per-task episode holdouts when requested.
- Expose gradient checkpointing, LoRA rank/scaling, VLM/vision freezing, and explicit
  peak/final LR names. Frozen VLM mode uses only the action recipe and skips text/FAST supervision.

## Impact

Changes affect the example, launcher, optional logger configuration/backend, trainer logger
selection, and dependency metadata. Default Trackio behavior and legacy checkpoint migration
remain supported. No source data changes, automatic training launch, hardware control, or interval filtering.
