# Chapter Tests — One-Month RL Path

Pass criterion for each chapter: ≥ 8/10 correct. Answers at end of each chapter section.

## Chapter 1 — Core RL Concepts & Gymnasium Basics
1. In an MDP, which component defines the immediate scalar feedback? 
   A) Policy  B) Reward  C) Transition model  D) Value function
2. The discount factor γ primarily controls: 
   A) Exploration  B) Reward scaling  C) Future reward weighting  D) Action space size
3. In Gymnasium, `env.step(action)` returns: 
   A) `state, reward`  B) `obs, reward, terminated, truncated, info`  C) `obs, reward, done, info`  D) `reward, obs`
4. Observation space describes: 
   A) Legal actions  B) Possible states/observations  C) Reward distribution  D) Episode length
5. A rollout is: 
   A) A single step update  B) A batch of transitions collected by interacting with the env  C) A policy initialization  D) A replay buffer
6. The Bellman equation relates: 
   A) Policy and reward only  B) Value of a state to expected returns from its successors  C) Action probabilities to gradients  D) Entropy to clipping
7. TensorBoard in SB3 typically logs: 
   A) Source code diffs  B) Episode reward, entropy, losses  C) Only rewards  D) Transitions count only
8. The `info` dict from Gymnasium is useful for: 
   A) Storing custom metrics/diagnostics  B) Changing action spaces  C) Defining rewards  D) Computing gradients
9. For `CartPole`, the action space is typically: 
   A) Continuous  B) Discrete with 2 actions  C) Discrete with 4 actions  D) Binary observation
10. The term “policy” refers to: 
   A) The loss function  B) Mapping from states to actions or distributions over actions  C) The value function  D) The environment reset function

Answers: 1-B, 2-C, 3-B, 4-B, 5-B, 6-B, 7-B, 8-A, 9-B, 10-B

References:
- Spinning Up — RL Intro: https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
- Gymnasium — Basics: https://gymnasium.farama.org/tutorials/gymnasium_basics/
- Stanford CS234 Playlist: https://www.youtube.com/playlist?list=PLoROMvodv4rN4wG6Nk6sNpTEbuOSosZdX

---

## Chapter 2 — Tabular RL: DP, MC, TD, Q-Learning, SARSA
1. Monte Carlo methods estimate values using: 
   A) One-step bootstrapping  B) Full returns from complete episodes  C) Gradients of policies  D) Entropy regularization
2. TD learning combines: 
   A) MC with supervised learning  B) Sampling with bootstrapping  C) Bootstrapping with gradients  D) Policy gradients with clipping
3. Q-learning is: 
   A) On-policy  B) Off-policy  C) Model-based  D) Purely Monte Carlo
4. SARSA updates use: 
   A) Next state’s max action  B) Next action from current policy  C) Only terminal rewards  D) A baseline
5. Policy evaluation computes: 
   A) The optimal policy directly  B) Value function under a fixed policy  C) Advantage function only  D) Entropy of the policy
6. In `FrozenLake-v1` with slippery tiles, the environment introduces: 
   A) Deterministic transitions  B) Stochastic transitions  C) Continuous actions  D) Continuous states
7. A typical Q-learning update uses: 
   A) Target = reward + γ * max_a' Q(s', a')  B) reward only  C) advantage estimate  D) entropy term
8. MC vs TD: MC tends to have ____ variance and ____ bias compared to TD. 
   A) Lower, higher  B) Higher, lower  C) Higher, higher  D) Lower, lower
9. Exploration can be handled by: 
   A) `epsilon`-greedy  B) Always choosing max-Q  C) Deterministic policies only  D) Noisy rewards
10. Convergence of tabular Q-learning requires (among others): 
   A) Constant large learning rate  B) GLIE and sufficient exploration  C) Zero exploration  D) No bootstrapping

Answers: 1-B, 2-B, 3-B, 4-B, 5-B, 6-B, 7-A, 8-B, 9-A, 10-B

References:
- Sutton & Barto (2020) Book (Ch. 4–6): http://incompleteideas.net/book/RLbook2020.pdf
- Stanford CS234 Playlist: https://www.youtube.com/playlist?list=PLoROMvodv4rN4wG6Nk6sNpTEbuOSosZdX

---

## Chapter 3 — Policy Gradient Basics (REINFORCE)
1. REINFORCE uses the gradient of: 
   A) Q-values  B) Expected return via log-prob of actions  C) Entropy only  D) Value function loss
2. A baseline in policy gradients: 
   A) Changes the optimum  B) Reduces variance without changing expected gradient  C) Increases bias  D) Removes need for sampling
3. The log-prob trick relates gradients to: 
   A) Bellman backup  B) Probability of actions under the policy  C) Advantage normalization only  D) Clipping
4. Advantage functions are used to: 
   A) Increase variance  B) Reduce variance by centering returns  C) Eliminate entropy  D) Change action spaces
5. REINFORCE requires: 
   A) Differentiable policies  B) Tabular Q-values only  C) Model-based planning  D) Replay buffers
6. Variance in REINFORCE can be reduced by: 
   A) Larger batch sizes  B) Baselines  C) Normalizing advantages  D) All of the above
7. Policy gradient methods optimize: 
   A) Value function directly  B) Expected return under the policy  C) Transition model  D) Entropy only
8. For `CartPole-v1`, a simple policy network outputs: 
   A) State transitions  B) Action logits/probabilities  C) Rewards  D) Gradients
9. Entropy regularization encourages: 
   A) Determinism  B) Exploration/diversity in actions  C) Lower returns  D) No learning
10. REINFORCE updates are typically computed: 
   A) Per episode  B) With bootstrapped targets  C) With target networks  D) Via Q-learning

Answers: 1-B, 2-B, 3-B, 4-B, 5-A, 6-D, 7-B, 8-B, 9-B, 10-A

References:
- Spinning Up — Policy Gradient (VPG): https://spinningup.openai.com/en/latest/algorithms/vpg.html
- Stanford CS234 Playlist: https://www.youtube.com/playlist?list=PLoROMvodv4rN4wG6Nk6sNpTEbuOSosZdX

---

## Chapter 4 — SB3 + PPO Internals
1. PPO’s clipped objective prevents: 
   A) Any gradient  B) Too large policy updates  C) Entropy computation  D) Value learning
2. GAE trades off: 
   A) Bias and variance via `lambda`  B) Entropy and clipping  C) Learning rate and batch size  D) Replay and targets
3. `n_steps` controls: 
   A) Total training steps  B) Rollout length before an update  C) Action space size  D) Evaluation frequency
4. `ent_coef` scales: 
   A) Value loss  B) Policy gradient  C) Entropy bonus  D) Advantage normalization
5. Explained variance near 1 suggests: 
   A) Poor value fit  B) Good value fit  C) High entropy  D) Divergence
6. `batch_size` in PPO affects: 
   A) How many transitions per optimizer step  B) Replay buffer size  C) Action space  D) Episode length
7. `clip_range` too small often causes: 
   A) No learning  B) Overly conservative updates  C) Exploding gradients  D) Increased entropy only
8. `MlpPolicy` typically includes: 
   A) CNN layers only  B) Fully connected layers with non-linearities  C) RNN layers only  D) No layers
9. TensorBoard `entropy` trend dropping too fast may indicate: 
   A) Healthy exploration  B) Early premature convergence  C) Great value fit  D) Better clipping
10. SB3 callbacks can: 
   A) Modify PyTorch directly only  B) Log custom metrics, save checkpoints, stop training  C) Change Gymnasium API  D) Only print

Answers: 1-B, 2-A, 3-B, 4-C, 5-B, 6-A, 7-B, 8-B, 9-B, 10-B

References:
- SB3 — PPO docs (`MlpPolicy` anchor): https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#mlppolicy
- Spinning Up — PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html
- Johnny Code — SB3 Intro: https://www.youtube.com/watch?v=OqvXHi_QtT0
- Johnny Code — Custom Env + Q-Learning/SB3: https://www.youtube.com/watch?v=AoGRjPt-vms

---

## Chapter 5 — Custom Env (Trading)
1. `observation_space` must: 
   A) Be arbitrary  B) Match the shape/type of returned observations  C) Be continuous only  D) Be discrete only
2. Reward shaping in trading often risks: 
   A) Encouraging random actions  B) Misaligned incentives (e.g., overtrading)  C) Fixed episode length  D) Deterministic transitions
3. Episode termination should occur: 
   A) Only on profit  B) At dataset end or hitting constraints  C) Randomly  D) Never
4. The `info` dict is best for: 
   A) Passing back diagnostics like PnL, drawdown  B) Defining actions  C) Computing gradients  D) Setting seed
5. Action space for a simple trading env might be: 
   A) Continuous actions only  B) Discrete: buy/hold/sell  C) Text commands  D) Image pixels
6. A sanity check for env rewards includes: 
   A) Plotting reward distribution under random policy  B) Reading code only  C) Training PPO immediately  D) Ignoring tensors
7. Data handling should ensure: 
   A) Leakage across train/test  B) No normalization  C) Chronological splits and normalization where appropriate  D) Random shuffling across time
8. A position limit protects against: 
   A) Too high entropy  B) Unbounded position growth  C) Gradient clipping failures  D) TensorBoard issues
9. Observation construction should avoid: 
   A) Fixed shape  B) Non-stationarity  C) Direct future data leakage  D) Reward scaling
10. Logging trades per step helps: 
   A) Diagnose behavior  B) Slow training only  C) Increase entropy  D) Reduce rewards

Answers: 1-B, 2-B, 3-B, 4-A, 5-B, 6-A, 7-C, 8-B, 9-C, 10-A

References:
- Gymnasium — Creating Custom Envs: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
- SB3 — Tips & Tricks: https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html

---

## Chapter 6 — PPO Tuning & Stability
1. Increasing `gae_lambda` towards 1.0 tends to: 
   A) Increase bias  B) Decrease variance  C) Decrease bias, increase variance  D) Increase entropy only
2. Large `n_steps` with small `batch_size` can: 
   A) Improve stability always  B) Hurt optimization per epoch  C) Change observation shapes  D) Disable clipping
3. Reward normalization often: 
   A) Stabilizes training  B) Prevents learning entirely  C) Is irrelevant  D) Changes action space
4. Time-limit truncation handling is important because: 
   A) It affects advantage estimates near episode ends  B) It sets `ent_coef`  C) It sets `vf_coef`  D) It changes seed
5. Too high `clip_range` tends to: 
   A) Aggressive updates and instability  B) No learning  C) Perfect value fit  D) Better entropy
6. Gradient explosion can be mitigated by: 
   A) Lower learning rate  B) Gradient clipping  C) Smaller networks  D) All of the above
7. Normalizing observations is useful because: 
   A) It simplifies action spaces  B) It helps network training across scales  C) It sets rewards  D) It reduces episodes
8. PPO instability symptoms include: 
   A) Highly volatile returns, collapsing entropy, poor value fit  B) Smooth returns, stable entropy  C) Perfect explained variance  D) Faster evaluation
9. A sweep across seeds helps you: 
   A) Overfit  B) Evaluate robustness  C) Increase entropy  D) Ignore variance
10. GAE balances: 
   A) Replay and sampling  B) Bias and variance via temporal credit assignment  C) Policy and value networks  D) Entropy and clipping

Answers: 1-C, 2-B, 3-A, 4-A, 5-A, 6-D, 7-B, 8-A, 9-B, 10-B

References:
- PPO Paper (Schulman et al., 2017): https://arxiv.org/abs/1707.06347
- Spinning Up — PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html
- SB3 — Tips & Tricks: https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html

---

## Chapter 7 — Reproducibility & Evaluation
1. Setting seeds ensures: 
   A) Identical results across all libraries always  B) Reduced randomness where supported  C) Increased exploration  D) No variance
2. `evaluate_policy` in SB3 computes: 
   A) Entropy  B) Average returns over episodes  C) Gradients  D) Clipping values
3. Confidence intervals over returns help: 
   A) Summarize variability and reliability  B) Change the policy  C) Reduce entropy  D) Set learning rate
4. A fair baseline in trading might be: 
   A) Buy-and-hold  B) Perfect foresight  C) Leverage 100x  D) Random walk reward
5. `Monitor` wrapper logs: 
   A) Episode rewards/lengths  B) Gradients  C) Entropy  D) Seeds only
6. Off-policy evaluation is tricky because: 
   A) It needs log-probs under the behavior policy  B) It always works  C) It sets PPO hyperparameters  D) It removes variance
7. Reproducible experiments require tracking: 
   A) Configs, versions, seeds, data splits  B) GPUs only  C) Entropy only  D) Value loss only
8. Aggregating over multiple seeds primarily: 
   A) Overfits to randomness  B) Measures central tendency and spread  C) Changes gym API  D) Disables clipping
9. Without proper evaluation, one risks: 
   A) False confidence in results  B) Better value fits  C) Lower entropy  D) Faster training
10. Weights & Biases or TensorBoard primarily provide: 
   A) Experiment tracking and visualization  B) Action spaces  C) Rewards  D) Transitions

Answers: 1-B, 2-B, 3-A, 4-A, 5-A, 6-A, 7-A, 8-B, 9-A, 10-A

References:
- SB3 — Reproducibility (section): https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#reproducibility
- SB3 — Evaluation utilities: https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html
- SB3 — Examples (Monitor usage): https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
- Gymnasium — Wrappers index: https://gymnasium.farama.org/api/wrappers/

---

## Chapter 8 — Capstone (EUR/BTC)
1. Max drawdown measures: 
   A) Average profit  B) Largest peak-to-trough loss  C) Entropy  D) Reward variance
2. Sharpe ratio measures: 
   A) Return adjusted by volatility  B) Entropy  C) Clipping quality  D) Value fit
3. A simple baseline for crypto might be: 
   A) Buy-and-hold  B) PPO with perfect reward  C) Future-aware policy  D) Gradient-only action
4. Train/test split in time series should be: 
   A) Random across full history  B) Chronological (older train, newer test)  C) Reversed  D) By day of week only
5. Overfitting risk in RL trading is increased by: 
   A) Using test data in training  B) Proper splits  C) Normalization  D) Logging
6. Evaluation beyond returns should include: 
   A) Risk metrics (volatility, drawdown), turnover, costs  B) Only episode length  C) Entropy  D) Clipping
7. Transaction costs affect: 
   A) Rewards and optimal policies  B) Only entropy  C) Only value loss  D) Only advantages
8. A checkpoint strategy helps: 
   A) Recover best-performing policies  B) Increase entropy  C) Randomize seeds  D) Remove variance
9. Comparing against a rule-based baseline helps you: 
   A) Inflate results  B) Get a realistic benchmark  C) Ignore metrics  D) Reduce variance
10. A thorough capstone report includes: 
   A) Assumptions, data prep, configs, results, evaluation, conclusions  B) Only returns  C) Only code  D) Only TensorBoard

Answers: 1-B, 2-A, 3-A, 4-B, 5-A, 6-A, 7-A, 8-A, 9-B, 10-A

References:
- Drawdown (economics): https://en.wikipedia.org/wiki/Drawdown_(economics)
- Sharpe ratio: https://en.wikipedia.org/wiki/Sharpe_ratio
