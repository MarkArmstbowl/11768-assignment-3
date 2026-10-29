# AI Usage

This statement covers work completed so far through Part 4, including
implementation, experiments, and report sections. It will be updated as the
assignment continues.

## OpenAI Codex

I used OpenAI Codex, an AI coding assistant in my IDE, for the following work:

- Reading and explaining the assignment, starter code, and instructor replies
  saved from Piazza; planning the implementation and experiments.
- Writing and reviewing the Part 1 advantage computation and trajectory data
  pipeline, and the Part 2 rollout-group helpers, SDK update submission, and
  synchronous trainer.
- Writing and reviewing the Part 3 asynchronous producer-consumer trainer,
  including policy-version capture, stale-group handling, background evaluation,
  and failure/cancellation cleanup, together with additional local tests.
- Writing additional local correctness tests and running authorized local tests
  and audits of the implemented code.
- Interpreting training logs and backend errors, and suggesting setup,
  deployment, training, recovery, and Git commands for me to run.
- Drafting and reviewing the system-flow study notes in `quiz.md`.
- Producing the Part 2, Part 3, and Part 4 report text and analysis,
  plotting and PDF-rendering scripts, figures, and tables from the recorded
  experimental outputs; checking the reports against those outputs and the
  assignment requirements, including the group-size study, the reused
  synchronous/asynchronous timing comparison, the higher-staleness
  importance-weighting study, and selected rollout analysis.

I ran the environment setup, Modal deployment, training, and Git commands
myself. The reported results come from the completed formal p2a, p2b, p3, p41,
and both p43 runs; short probes and the archived zero-update failed startup
are excluded.

## Experimental model

The experiments use `Qwen/Qwen3.5-4B` as the policy model, with the required LoRA
configuration, hosted through SkyRL's Tinker-compatible service on Modal.
