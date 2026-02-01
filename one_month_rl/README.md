# One-Month Reinforcement Learning Path (Tailored)

This plan is based on your current interests and experience: hands-on work with Stable-Baselines3 (`PPO`, `MlpPolicy`), Gymnasium environments, TensorBoard, and a trading environment. It focuses on building solid RL fundamentals, deeply understanding PPO, structuring experiments, and applying them to a trading task. Progress is gated: you must pass each chapter’s multiple-choice test to continue (>= 80%).

## Structure
- Duration: 4 weeks
- Chapters: 8 (2 per week)
- Each chapter: 1–2h theory, 1–2h practice, MCQ test
- Pass criterion: 8/10 correct in the corresponding test (see `tests.md`)

## Week 1 — RL Foundations & Tabular Methods

### Chapter 1: Core RL Concepts & Gymnasium Basics
- Description: Foundations of RL framed as decision-making under uncertainty; how Gymnasium formalizes environments; how SB3 interacts via rollouts.
- You will learn:
  - MDPs: states, actions, rewards, transitions, discounting
  - Policies, value functions, Bellman equations (high-level)
  - Gymnasium API: `Env`, observation and action spaces, steps, resets
  - SB3 agent lifecycles at a glance
- Objectives:
  - Explain an MDP and why discounting matters
  - Identify Gymnasium observation/action spaces for a simple env
  - Describe how an RL algorithm collects rollouts
- Theory (1–2h):
  - Read: Spinning Up RL — Intro (https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
  - Watch: Stanford CS234 Lecture 1 — RL Overview (Playlist: https://www.youtube.com/playlist?list=PLoROMvodv4rN4wG6Nk6sNpTEbuOSosZdX)
  - Read: Gymnasium docs — Environment basics (https://gymnasium.farama.org/tutorials/gymnasium_basics/)
- Practice (1–2h):
  - Implement a tiny Gridworld (or use `FrozenLake-v1`) with Gymnasium
  - Interact with env using a random policy and log transitions
- Test: See `tests.md` → Chapter 1

### Chapter 2: Tabular RL — DP, MC, TD, Q-Learning, SARSA
- Description: Classic algorithms to build intuition before deep RL.
- You will learn:
  - Policy evaluation/improvement, Monte Carlo vs TD learning
  - Q-learning vs SARSA; on/off-policy distinctions
- Objectives:
  - Implement and tune tabular Q-learning
  - Explain difference between MC and TD methods
- Theory (1–2h):
  - Read: Sutton & Barto (ch. 4–6) — Official book (http://incompleteideas.net/book/RLbook2020.pdf)
  - Watch: CS234 — Tabular methods segment (Playlist: https://www.youtube.com/playlist?list=PLoROMvodv4rN4wG6Nk6sNpTEbuOSosZdX)
- Practice (1–2h):
  - Train tabular Q-learning on `FrozenLake-v1` with slippery vs non-slippery
  - Plot learning curves; compare SARSA vs Q-learning
- Test: See `tests.md` → Chapter 2

## Week 2 — Policy Gradients & SB3/PPO Internals

### Chapter 3: Policy Gradient Basics (REINFORCE)
- Description: Move from value-based methods to optimizing policies directly.
- You will learn:
  - REINFORCE objective, gradients via log-prob, variance reduction
  - Advantage intuition leading to GAE (prelude to PPO)
- Objectives:
  - Implement a simple REINFORCE agent (PyTorch) on `CartPole-v1`
  - Explain why baselines reduce variance
- Theory (1–2h):
  - Read: Spinning Up — Policy Gradient (VPG) (https://spinningup.openai.com/en/latest/algorithms/vpg.html)
  - Watch: CS234 — Policy Gradient segment (Playlist: https://www.youtube.com/playlist?list=PLoROMvodv4rN4wG6Nk6sNpTEbuOSosZdX)
- Practice (1–2h):
  - Train REINFORCE; add baseline; compare returns and variance
- Test: See `tests.md` → Chapter 3

### Chapter 4: Stable-Baselines3 + Gymnasium Stack Deep Dive
- Description: Understand SB3’s training loop and how PPO integrates with `MlpPolicy`.
- You will learn:
  - PPO rollouts, advantages (GAE), clipping, value/entropy losses
  - Key hyperparameters: `n_steps`, `batch_size`, `gamma`, `gae_lambda`, `clip_range`, `vf_coef`, `ent_coef`, `learning_rate`
  - Callbacks, logging, evaluation APIs; TensorBoard interpretation (entropy, value loss, explained variance)
- Objectives:
  - Configure PPO agents for `CartPole-v1` and `LunarLander-v2`
  - Interpret TensorBoard metrics and diagnose instability
- Theory (1–2h):
  - Read: SB3 PPO docs (https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#mlppolicy)
  - Watch: Johnny Code — Getting Started with SB3 (https://www.youtube.com/watch?v=OqvXHi_QtT0) and Custom Gymnasium Env + Q-Learning/SB3 (https://www.youtube.com/watch?v=AoGRjPt-vms)
- Practice (1–2h):
  - Train PPO with 2 configs; log with TensorBoard; write a callback to record episode rewards
- Test: See `tests.md` → Chapter 4

## Week 3 — Custom Environments & PPO Tuning

### Chapter 5: Custom Gymnasium Environment — Trading Focus
- Description: Build a minimal trading env that matches your EUR/BTC interest.
- You will learn:
  - Specifying observation/action spaces for trading
  - Reward shaping pitfalls; episode termination; data handling
- Objectives:
  - Implement `gymnasium.Env` with `reset`/`step` for synthetic price series
  - Log trades, rewards, and positions per step
- Theory (1–2h):
  - Read: Gymnasium — Creating Custom Envs (https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
  - Review: RL reward shaping tips (SB3 Tips & Tricks: https://stable-baselines3.readthedocs.io/en/stable/guide/rl_tips.html)
- Practice (1–2h):
  - Implement env; run random policy; sanity-check reward distribution
- Test: See `tests.md` → Chapter 5

### Chapter 6: PPO Hyperparameters, GAE, and Stability
- Description: Practical PPO tuning and common pitfalls.
- You will learn:
  - GAE intuition; bias-variance tradeoff with `gae_lambda`
  - PPO clipping and its effect on updates; normalization; time limits
- Objectives:
  - Execute a small hyperparameter sweep on your trading/env or `LunarLander-v2`
  - Diagnose failures via logs; apply normalization and reward scaling
- Theory (1–2h):
  - Read: PPO paper (https://arxiv.org/abs/1707.06347) & Spinning Up — PPO (https://spinningup.openai.com/en/latest/algorithms/ppo.html)
  - Read: SB3 best practices (https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
- Practice (1–2h):
  - Sweep `clip_range`, `n_steps`, `gae_lambda`; compare mean returns and stability
- Test: See `tests.md` → Chapter 6

## Week 4 — Experiment Design, Evaluation, and Capstone

### Chapter 7: Reproducibility, Evaluation, and Experiment Tracking
- Description: Make experiments reliable and comparable.
- You will learn:
  - Seeding, evaluation protocols, off-policy evaluation caveats
  - Tools: `evaluate_policy`, `Monitor`, TensorBoard; optional: Weights & Biases
- Objectives:
  - Establish a standard experiment template with config + seed list
  - Produce reproducible plots and tables
- Theory (1–2h):
  - Read: Reproducibility in RL — SB3 (https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#reproducibility)
  - Review: SB3 evaluation utilities (https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html)
- Practice (1–2h):
  - Run 3 seeds × 2 configs; aggregate returns; compute confidence intervals
- Test: See `tests.md` → Chapter 7

### Chapter 8: Capstone — EUR/BTC Trading Agent
- Description: Integrate everything into a documented, reproducible pipeline.
- You will learn:
  - Data prep for crypto; risk metrics (max drawdown, Sharpe)
  - Baselines vs PPO policy comparison; generalization cautions
- Objectives:
  - Train PPO on your env with clear logs, checkpoints, and metrics
  - Compare vs a simple baseline (e.g., buy-and-hold or rule-based)
- Theory (1–2h):
  - Read: Trading evaluation metrics — Max Drawdown (https://en.wikipedia.org/wiki/Drawdown_(economics)), Sharpe Ratio (https://en.wikipedia.org/wiki/Sharpe_ratio)
  - Review: prior chapters’ notes
- Practice (1–2h):
  - Full run + report: setup, configs, training curves, evaluation, conclusions
- Test: See `tests.md` → Chapter 8

## Gating & Schedule
- Gate: Proceed only if you score ≥ 8/10 in the chapter test.
- Remediation: If you fail, revisit the theory links, re-run practice with smaller scope, and retake.
- Suggested cadence:
  - Mon/Tue: Chapters 1–2
  - Wed/Thu: Chapters 3–4
  - Fri: Chapter 5
  - Mon: Chapter 6
  - Tue/Wed: Chapter 7
  - Thu/Fri: Chapter 8 + capstone write-up

## Resources
- Spinning Up RL by OpenAI (Policy Gradient, PPO, Key Concepts)
- Stable-Baselines3 Docs (PPO, Policies, Callbacks, Logging)
- Gymnasium Docs (Envs, Spaces, Custom Envs)
- CS234 Lectures (Emma Brunskill) — selected segments
- Johnny Code videos — SB3 + Gymnasium

## Deliverables
- Practice artifacts: code, logs, TensorBoard screenshots
- Tests: answers tracked; gating enforced
- Capstone report: assumptions, configs, results, evaluation, next steps

See tests: `tests.md`
See practice checklists: `practice_checklists.md`
