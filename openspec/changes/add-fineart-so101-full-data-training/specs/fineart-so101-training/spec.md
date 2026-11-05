## ADDED Requirements

### Requirement: Full-data training configuration
The example SHALL default to every episode and interval, resolve SO101 features from the dataset,
retain the base checkpoint's FAST token offset, and delegate training to the native trainer.

#### Scenario: Prepare a full-data run
- **WHEN** check mode runs against the supplied annotated dataset and local checkpoint
- **THEN** the generated config contains 12-dimensional SO101 features, no interval or episode
  exclusions, the checkpoint's FAST offset, and the subtask training recipe
- **AND** no model weights are loaded and no training or tokenizer fitting is started

### Requirement: User-configurable training budget and split
The launcher SHALL expose batch size, learning rates, warmup, epochs/steps, train ratio,
workers, seed, and save/log/eval controls in its editable parameter block and command line.

#### Scenario: Five epochs after changing the batch or split
- **WHEN** no explicit steps are supplied
- **THEN** check and train default to five epochs computed from the actual training frames
- **AND** the calculation includes even-batch padding across all FSDP2 or DDP workers
- **AND** explicit steps override epochs and smoke defaults to 20 steps

#### Scenario: Hold out complete episodes
- **WHEN** train ratio is between zero and one
- **THEN** the native per-task episode holdout is reused and interval flags remain ignored
- **AND** the report includes disjoint train/eval episode lists, frame counts, and actual proportions
- **AND** parent FAST fitting receives only training episodes, excluding held-out episodes

### Requirement: Configurable trainability and learning-rate schedule
The launcher SHALL expose gradient checkpointing, LoRA rank/alpha, VLM/vision freezing,
warmup steps, maximum LR and final LR, and pass these controls to native training configuration.

#### Scenario: Frozen VLM action training
- **WHEN** whole-VLM freezing is requested
- **THEN** the VLM is frozen with train_expert_only, text/FAST supervision and FAST fitting are disabled
- **AND** all batches use the low-level subtask action recipe
- **AND** any LoRA adapters target only the action expert and action/time projections

#### Scenario: LoRA with language supervision
- **WHEN** LoRA is enabled and whole-VLM freezing is not requested
- **THEN** native PEFT receives rank/alpha and language+expert targets, excluding vision and LM head
- **AND** text/FAST supervision remains enabled to train the language adapters

#### Scenario: Explicit LR schedule
- **WHEN** maximum/final LR and warmup steps are specified
- **THEN** native optimizer/scheduler configs retain those values and decay ends at total steps
- **AND** old LR/MIN_LR aliases remain supported

### Requirement: Inspectable preflight and explicit launch
The launcher SHALL check real language coverage and video decoding, save a preflight report,
and start the native trainer only in smoke or train mode.

#### Scenario: Discover launcher usage
- **WHEN** a user reads the script header or invokes --help or -h
- **THEN** usage explains setup/check/smoke/train, the default check mode, and parameter overrides
- **AND** shell help returns before invoking uv, Python, data loading, or training
- **AND** unknown modes print usage and exit with status 2

#### Scenario: Missing subtask
- **WHEN** any frame resolves to a missing or empty subtask
- **THEN** preparation fails before training and reports the language coverage failure

#### Scenario: Existing training output
- **WHEN** the requested train/ directory already exists
- **THEN** preparation refuses to overwrite that run

### Requirement: Dual GPU launch and W&B logging
The example SHALL default to two native workers and enable W&B logging without duplicate runs.

#### Scenario: Select native FSDP2
- **WHEN** the default FSDP backend is selected
- **THEN** the topology is dp_replicate=1 and dp_shard=2 with BF16
- **AND** SiglipEncoderLayer children form separate units while joint decoder parameters remain in the root
- **AND** reshard-after-forward and CPU offload are configurable
- **AND** existing trainability, LR, W&B, dataset, and batch-budget controls remain available

#### Scenario: Select DDP fallback
- **WHEN** the DDP backend is explicitly selected
- **THEN** the topology is dp_replicate=2 and dp_shard=1
- **AND** the global batch and epoch budget match FSDP2 for the same per-GPU batch and training frames

#### Scenario: Configure FSDP groups
- **WHEN** dp_shard and dp_replicate are explicitly set to positive degrees
- **THEN** they control cards per shard group and number of replica groups respectively
- **AND** the visible GPU count and native worker count equal their product
- **AND** global batch and epoch budgets use all data-parallel workers
- **AND** two nontrivial dimensions use native HSDP with unchanged FineART wrap units

#### Scenario: Reject inconsistent GPU selection
- **WHEN** visible GPUs are missing, repeated, or differ in count from dp_shard times dp_replicate
- **THEN** the launcher fails before checkpoint resolution, data loading, or training

#### Scenario: Reuse the established W&B team
- **WHEN** the example generates its default training configuration
- **THEN** W&B uses entity `2943872339-zhejiang-university` from the PI05 folding recipe
- **AND** the project is `fineart-so101` and the job name is `fineart_so101_full_<mode>`
- **AND** `wandb_logging.disable_artifact` is explicitly true

#### Scenario: Launch dual GPU training
- **WHEN** smoke or train mode is selected with the default topology
- **THEN** the parent prepares the FAST tokenizer once before launching two workers
- **AND** both workers read the completed tokenizer with auto-fitting disabled
- **AND** batch size is interpreted per GPU and only rank 0 logs the reduced native metrics

#### Scenario: Resume logged training
- **WHEN** an online W&B run resumes from its saved native checkpoint configuration
- **THEN** its saved W&B run ID is reused with server resume enabled
- **AND** the legacy wandb checkpoint-section migration and default Trackio behavior remain supported
