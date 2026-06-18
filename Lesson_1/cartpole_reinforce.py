#!/usr/bin/env python3
"""Simple REINFORCE implementation for CartPole.

Gym's classic-control renderer uses Pygame when the environment is created with
render_mode="human".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

try:
    import gym
except ImportError:  # Gymnasium keeps the same API for this example.
    import gymnasium as gym


CHECKPOINT = Path(__file__).with_name("cartpole_policy.pt")


class Policy(nn.Module):
    def __init__(self, state_dim: int = 4, hidden_dim: int = 128, action_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def reset(env):
    result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def step(env, action: int):
    result = env.step(action)
    if len(result) == 5:
        state, reward, terminated, truncated, info = result
        return state, reward, terminated or truncated, info
    return result


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    returns = []
    total = 0.0
    for reward in reversed(rewards):
        total = reward + gamma * total
        returns.append(total)

    returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)
    return (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)


def run_episode(env, policy: Policy, gamma: float, deterministic: bool = False):
    state = reset(env)
    done = False
    rewards: list[float] = []
    log_probs: list[torch.Tensor] = []

    while not done:
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        logits = policy(state_tensor)
        dist = Categorical(logits=logits)

        action = torch.argmax(logits) if deterministic else dist.sample()
        state, reward, done, _ = step(env, int(action.item()))

        rewards.append(float(reward))
        log_probs.append(dist.log_prob(action))

    returns = discounted_returns(rewards, gamma)
    loss = -torch.stack([log_prob * value for log_prob, value in zip(log_probs, returns)]).sum()
    return loss, sum(rewards)


def train(episodes: int, gamma: float, lr: float, seed: int) -> Policy:
    torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    best_score = 0.0
    try:
        for episode in range(1, episodes + 1):
            loss, score = run_episode(env, policy, gamma)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            best_score = max(best_score, score)
            if episode == 1 or episode % 25 == 0:
                print(f"episode={episode:04d} score={score:6.1f} best={best_score:6.1f}")

            if score >= 500:
                print(f"Solved at episode {episode}.")
                break
    finally:
        env.close()

    torch.save(policy.state_dict(), CHECKPOINT)
    print(f"Saved policy to {CHECKPOINT}")
    return policy


def watch(policy: Policy, episodes: int, gamma: float) -> None:
    env = gym.make("CartPole-v1", render_mode="human")
    try:
        for episode in range(1, episodes + 1):
            _, score = run_episode(env, policy, gamma, deterministic=True)
            print(f"watch_episode={episode} score={score:.1f}")
    finally:
        env.close()


def load_policy() -> Policy:
    policy = Policy()
    policy.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    policy.eval()
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and animate a CartPole policy with Gym, Torch, and Pygame.")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--watch", type=int, default=3, help="Pygame-rendered episodes after training.")
    parser.add_argument("--watch-only", action="store_true", help="Load the saved policy and only animate it.")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    policy = load_policy() if args.watch_only else train(args.episodes, args.gamma, args.lr, args.seed)
    if args.watch:
        watch(policy, args.watch, args.gamma)


if __name__ == "__main__":
    main()
