# Practice Checklists — One-Month RL Path

Use these as completion criteria for each chapter’s practice.

## Chapter 1
- [ ] Run a Gymnasium env (`FrozenLake-v1` or `CartPole-v1`) with random policy
- [ ] Log transitions (obs, action, reward, terminated/truncated) for 100 steps
- [ ] Explain MDP components for the chosen env

## Chapter 2
- [ ] Implement tabular Q-learning (with `epsilon`-greedy)
- [ ] Compare SARSA vs Q-learning returns over 5 seeds
- [ ] Plot learning curves; interpret stability differences

## Chapter 3
- [ ] Implement REINFORCE on `CartPole-v1`
- [ ] Add baseline; compare variance (std of returns across episodes)
- [ ] Save training curves; summarize outcome

## Chapter 4
- [ ] Train PPO (`MlpPolicy`) on `CartPole-v1` and `LunarLander-v2`
- [ ] Use TensorBoard; record `entropy`, `value_loss`, `explained_variance`
- [ ] Implement a simple SB3 callback that logs episode reward every N steps

## Chapter 5
- [ ] Implement minimal trading `gymnasium.Env` with synthetic price series
- [ ] Validate `observation_space` and `action_space`; run random policy
- [ ] Log per-step: position, trade, reward, PnL; sanity-check reward distribution

## Chapter 6
- [ ] Sweep PPO hyperparams (`clip_range`, `n_steps`, `gae_lambda`)
- [ ] Apply observation/reward normalization; explain effects
- [ ] Diagnose instability via logs; propose fixes

## Chapter 7
- [ ] Create experiment template: config file + seed list
- [ ] Run 3 seeds × 2 configs; aggregate returns; compute 95% CI
- [ ] Use `Monitor` + `evaluate_policy`; store metrics and plots

## Chapter 8
- [ ] Train PPO on trading env with clear configs & checkpoints
- [ ] Evaluate vs baseline (buy-and-hold or rule-based)
- [ ] Compute risk metrics: volatility, max drawdown, Sharpe; write capstone report
