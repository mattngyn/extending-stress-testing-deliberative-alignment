# Training Pipeline (Future Work)

This directory is a stub for the training pipeline described in [Sch+25].
The evaluation side of the codebase is fully implemented; this documents
how training would integrate.

## Architecture: Separate Base Classes

Test and training environments serve different purposes and use different
frameworks:

- **`BaseTestEnvironment`** (Inspect AI) — for evaluation. Each subclass
  tests a different covert behavior type with its own custom scorer.
  Produces an Inspect `Task`.
- **`BaseTrainingEnvironment`** (Verifiers) — for training. All subclasses
  test the same behavior (covert rule violation) with different specific
  rules. Uniform scoring structure. Produces a Verifiers `ToolEnv`.

## Training Taxonomy

| Stage | Framework | Input | Output |
|---|---|---|---|
| **SFT** | TRL `SFTTrainer` | Filtered Verifiers rollouts → HF Dataset | SFT checkpoint |
| **Offline RL** | TRL `DPOTrainer` | Verifiers rollouts → preference pairs | RL checkpoint |
| **Online RL** | Verifiers + TRL `GRPOTrainer` | Live rollouts with `Rubric` rewards | RL checkpoint |

## SFT Pipeline

1. Run model through Verifiers training environments (`ToolEnv`) with
   anti-scheming spec (AS1-AS5, GP1-GP4) in the system prompt
2. Collect rollouts
3. Filter: (a) per-environment grading script says desirable outcome,
   (b) reward model confirms spec-driven reasoning
4. Strip anti-scheming spec from system prompt (model must learn to
   reason about the spec from memory)
5. Format as HF Dataset with `"messages"` column → `SFTTrainer`

## Offline RL Pipeline

1. Run SFT model through Verifiers environments, multiple rollouts per env
2. Group by environment
3. Pair desirable (escalated) with undesirable (violated rule) trajectories
4. Format as HF Dataset with `"prompt"`, `"chosen"`, `"rejected"` → `DPOTrainer`

## Online RL Pipeline

1. Build environments as Verifiers `ToolEnv` via `BaseTrainingEnvironment`
2. Combine anti-scheming + anti-overrefusal environments via `EnvGroup`
3. `Rubric` with per-environment reward (uniform: rule violation + escalation)
4. `GRPOTrainer` generates rollouts, scores, updates weights in one loop

## Training Environment Details (from Paper Section 3.3)

- **Anti-scheming**: Conflicts between task completion and rule compliance.
  Agent discovers rule during execution; completing task requires violating
  it. Rewarded behavior: refuse and escalate. All terminal/sandbox-based.
- **Anti-overrefusal**: Minimal edits adding rule exceptions. Rewarded
  behavior: complete the task. Prevents "always refuse" policy.
- **Synthetic generation**: LLMs generate environments with varied
  motivations, rules, concealment incentives.
- **Validation**: Schemer + saint verifier agents confirm correct scoring.
- **Per-environment grading**: Uniform structure (rule violation +
  escalation), but the specific rule check varies per environment.
