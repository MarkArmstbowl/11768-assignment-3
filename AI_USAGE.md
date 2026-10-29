# AI Usage

This statement currently covers the completed work through Part 2. It will be
updated as the remaining parts of the assignment are completed.

## OpenAI Codex

I used OpenAI Codex, an AI coding assistant in my IDE, for the following work:

- Reading and explaining the assignment, starter code, and instructor replies
  saved from Piazza; planning the implementation and experiments.
- Writing and reviewing the Part 1 advantage computation and trajectory data
  pipeline, and the Part 2 rollout-group helpers, SDK update submission, and
  synchronous trainer.
- Writing additional local correctness tests and running authorized local tests
  and audits of the implemented code.
- Interpreting training logs and backend errors, and suggesting setup,
  deployment, training, recovery, and Git commands for me to run.
- Drafting and reviewing the system-flow study notes in `quiz.md`.
- Producing the Part 2 report text and analysis, plotting and PDF-rendering
  scripts, figures, and tables from the recorded experimental outputs; checking
  the report against those outputs and the assignment requirements.

I ran the environment setup, Modal deployment, training, and Git commands
myself. The reported Part 2 results come from the two completed formal runs;
the short probe and the archived zero-update failed startup are excluded.

## Experimental model

The experiments use `Qwen/Qwen3.5-4B` as the policy model, with the required LoRA
configuration, hosted through SkyRL's Tinker-compatible service on Modal.
