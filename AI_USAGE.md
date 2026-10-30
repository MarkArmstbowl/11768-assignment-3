# AI Usage

I used OpenAI Codex in my IDE, with GPT models, as a coding and analysis
collaborator throughout this assignment. I set the direction and constraints
for the work. At my request, Codex wrote most of the student implementation,
additional tests, report prose, figures, and plotting/rendering code.

I directed the work and kept responsibility for the experiments and
submission. I read and repeatedly referred to the assignment and the
instructor replies saved from Piazza, asked Codex to follow the required
parts and unchanged experiment configurations, and challenged its suggested
commands, time estimates, error interpretations, and claims of completeness.
I also asked for explanations of the system mechanisms: token alignment,
advantage computation, SDK call ordering, policy versions, stale data,
retries, and cancellation. Codex helped me maintain study notes in
`quiz.md`; I used these explanations and code walkthroughs to study how
the system works.

I ran the environment setup, account authentication, Modal deployments,
training, all four final TextCraft evaluations, Hugging Face upload, and
Git commands myself. I checked Modal activity and credit usage, reviewed
terminal output and generated reports, and brought unexpected warnings,
errors, and results back for analysis. I ran local tests and also authorized
Codex to run local checks at specific points. I required setup and remote
operations to remain under my control.

Codex assisted with:

- Implementing and reviewing the six student files: advantages, trajectory
  stitching and masking, rollout-group/batch helpers, Tinker update
  submission, and the synchronous and asynchronous trainers.
- Writing additional local tests with fake policies, environments, SDK
  futures, and worker failures; reviewing edge cases and hardening
  failure/cancellation cleanup.
- Explaining the writeup and Piazza guidance, planning experiments, and
  suggesting commands for setup, deployment, training, recovery, export,
  evaluation, and Git operations for me to execute.
- Investigating local artifacts and the logs I supplied, consulting
  official Modal and Hugging Face documentation, and distinguishing model
  mistakes, parser failures, and infrastructure errors.
- Writing the optional external evaluation-progress wrapper and its
  offline tests. I used it for the two base-model evaluations. It observes
  completed episodes while calling the supplied evaluator, retaining its
  task set, model settings, concurrency, scoring, and final JSON, with no
  additional sampling requests or retries.
- Drafting and checking the reports through Part 5, selecting and analyzing
  recorded rollouts, producing diagnostic plots and tables, and preparing
  the combined PDF and final artifact/requirement audits.

The reported experiments use the seven required formal runs with their
provided configurations. Short probes and the archived failed p2b startup
that made no optimizer updates are excluded. Part 5 completed 50 updates
without resuming; its final adapter is `tinker://model_71bf8a90/v50`, exported
to the public repository `Marstbowl/scottyrl-a3-p5_textcraft_grpo`. The four
full-split evaluations completed. Their abandoned-episode counts of 18, 37,
0, and 2 (fine-tuned medium/hard, then base medium/hard) remain counted as
failures and are disclosed rather than removed from the reported results.

The experimental policy model was `Qwen/Qwen3.5-4B` with LoRA rank 32,
trained through the provided SkyRL Tinker-compatible service on Modal.
Final evaluation used the provided standalone vLLM serving route. Codex's
GPT models provided coding and writing assistance; Qwen was the model
whose agent behavior was trained and evaluated.
