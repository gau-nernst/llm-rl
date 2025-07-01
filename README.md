# Reinforcement Learning for LLMs

This repo serves as a workspace for my learning of RL in the context of modern LLMs.

Resources:
- https://spinningup.openai.com/

## Basic concepts

Term | Explanation
---|---
Agent | Us
Environment | The world
Action | Environment changes upon an agent's action, but it can also change on its own. Current action is acted **after**/**on** current state.
State | Complete view of the world.
Observation | What agent can see. Partial or full state of the world.
Policy | Agent's rule to decide which action to take. It can be probabilistic / stochastic. Usually it's parameterized so that we can optimize it.
Trajectory (aka episodes, rollouts) | Sequence of states and actions
State transition | What happens to the world in the next time step. Depends on the laws of the environment, and only the current action. It can be stochastic.
Reward | Response from the environment to the agent. Only depends on **current state, current action, and next state**. It can be simplified to only depend on current state, or current state and current action.
Return | Cumulative reward over a trajectory. The agent's goal is to maximize this. For infinite-horizon, we can include a discount factor.
Expected return | Since policy and transition can be stochastic, trajectory is stochastic as well. Hence, expected return is computed over the distribution of trajectories. The **optimal policy** maximizes expected return.
Value function | Expected return at a particular state, following a particular policy. **Action-value function** (or Q-function) additionally conditioned on a particular (current) action. It follows that Value function is expected value of Action-value function over policy (action distribution). **Optimal action** corresponds to optimal Q-function.
Advantage function | The difference between Q-function and Value function i.e. Q - V -> how much a particular action is better than average, given a policy.

## Model-free vs Model-based

Model-based: Agent learns to model the environment. This allows the agent to plan ahead, but the learned model might not generalise well.

### Model-free RL

Learning | Description
---|---
Policy optimization | Models the policy. Optimize via stochastic gradient ascent on (approximate) Expected return. Typically **on-policy** - use data according to latest policy. Usually more stable. e.g. A2C/A3C, PPO.
Q-Learning | Models the Q-function. Optimize based on Bellman equation. Typically **off-policy** - use any data, not necessarily with the latest policy. The corresponding policy is choosing the action with the largest Q-function value. e.g. DQN.

## Policy gradient

**Policy gradient** is the gradient of Expected return $J(\pi_\theta)$ (our objective) with respect to Policy's parameters $\theta$. We can derive a nice form for it.

```math
\nabla_\theta J(\pi_\theta) = \nabla_\theta \mathbb{E}_{\tau\sim \pi_\theta}[R(\tau)]
```
```math
= \nabla_\theta \int_\tau P(\tau|\theta) R(\tau)
```
```math
= \int_\tau \nabla_\theta P(\tau|\theta) R(\tau)
```
```math
= \int_\tau P(\tau|\theta) \nabla_\theta\log P(\tau|\theta) R(\tau)
```
```math
= \mathbb{E}_{\tau\sim\pi_\theta}\left[\nabla_\theta\log P(\tau|\theta) R(\tau)\right]
```
```math
= \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^T \nabla_\theta\log\pi_\theta(a_t|s_t) R(\tau)\right]
```

Note: A trajectory's log-prob is equal to log-prob of initial state, plus the log-prob sum of (action + state transition). Since initial state and state transition are properties of the environment, they do not depend on policy's parameters. Therefore, the derivative of trajectory's log-prob (wrt policy's parameters) is equal to the sum of log-policy's derivatives.

```math
\nabla_\theta\log P(\tau|\theta) = \sum_{t=0}^T \nabla_\theta\log\pi_\theta(a_t|s_t)
```

In the sampling form (estimator)

```math
\hat g = \frac{1}{|D|} \sum_{\tau\in D}\sum_{t=0}^T \nabla_\theta\log\pi_\theta(a_t|s_t) R(\tau)
```

Notice that once we sample the trajectories (using current policy), we can bring the gradient operator out of the sum. We can define the objective function:

```math
L(\theta) = \frac{1}{|D|} \sum_{\tau\in D}\sum_{t=0}^T \log\pi_\theta(a_t|s_t) R(\tau)
```

Please note that this loss function is only defined to simplify the computation of policy gradient using an autograd engine. **THERE IS NO MEANING** behind this loss function. Do not try to interpret its value.

**Other forms of Policy gradient** It turns out that we can replace $R(\tau)$ in Policy gradient equation with other terms that result in the same expected value. These alternatives can offer a tighter variance of Policy gradient estimator, making training more data-efficient. Here are some alternative forms (proofs won't be covered).

Using on-policy Q-function

```math
\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^T \nabla_\theta\log\pi_\theta(a_t|s_t) Q^{\pi_\theta}(s_t,a_t)\right]
```

Using Advantage function - this is the most common since there are many ways to estimate the advantage function and it results in lowest variance.

```math
\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^T \nabla_\theta\log\pi_\theta(a_t|s_t) A^{\pi_\theta}(s_t,a_t)\right]
```

### Trust Region Policy Estimation (TRPO)

https://arxiv.org/abs/1502.05477

Impose an additional constraint not to deviate too much from the old policy -> avoid big update that can collapse training.

**Surrogate advantage** is a measure of how policy $\pi_\theta$ performs relative to the old policy $\pi_{\theta_{old}}$, using data from the old policy.

```math
L(\theta_{old},\theta) = \mathbb{E}_{s,a\sim\pi_{\theta_{old}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}A^{\pi_{\theta_{old}}}(s,a)\right]
```

**Average KL-divergence** over states visited by the old policy

```math
\bar D_{KL}(\theta|\theta_{old}) = \mathbb{E}_{s\sim\pi_{\theta_{old}}}\left[D_{KL}(\pi_\theta|\pi_{\theta_{old}})\right]
```

Theoretical TRPO update

```math
\theta_{k+1} = \arg\max_\theta L(\theta_k,\theta), \bar D_{KL}(\theta||\theta_k)\leq\delta
```

Note that the gradient of surrogate advantage function (wrt policy's parameters) is still equal to policy gradient.

### Proximal Policy Optimization (PPO)

https://arxiv.org/abs/1707.06347

**PPO-clip** Loss function

```math
L_{PPO-Clip} = \mathbb{E}_{s,a\sim\pi_{\theta_{old}}}\left[\min\left(\frac{\pi_\theta}{\pi_{\theta_{old}}}A^{\pi_{\theta_{old}}},\mathrm{clip}\left(\frac{\pi_\theta}{\pi_{\theta_{old}}},1-\epsilon,1+\epsilon\right)A^{\pi_{\theta_{old}}}\right)\right]
```

Note that since Advantage $A^{\pi_{\theta_{old}}}$ can be either positive or negative, we can't factor out the Advantage term.

**PPO-Penalty** Loss function

```math
L_{PPO-Penalty} = \mathbb{E}_{s,a\sim\pi_{\theta_{old}}}\left[\frac{\pi_\theta}{\pi_{\theta_{old}}}A^{\pi_{\theta_{old}}}-\beta D_{KL}(\pi_{\theta_{old}}|\pi_\theta)\right]
```

Where the penalty coefficient $\beta$ is adjusted automatically over the course of training.

### Generalized Advantage Estimator (GAE)

https://arxiv.org/abs/1506.02438
