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
= \mathbb{E}_{\tau\sim\pi_\theta}[\nabla_\theta\log P(\tau|\theta) R(\tau)]
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

Notice that once we sample the trajectories (using current policy), we can bring the gradient operator out of the sum. We can define the loss function

```math
L = -\frac{1}{|D|} \sum_{\tau\in D}\sum_{t=0}^T \log\pi_\theta(a_t|s_t) R(\tau)
```

Please note that this loss function is only defined to simplify the computation of policy gradient using an autograd engine. **THERE IS NO MEANING** behind this loss function. Do not try to interpret its value.
