"""Demonstrate a trained Taxi-v3 Q-Learning agent."""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register, registry
from gymnasium.spaces import Discrete


ENV_NAME = "Taxi-v3"
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_Q_TABLE_PATH = PROJECT_DIR / "q_table.npy"

ACTION_NAMES = {
    0: "south / вниз",
    1: "north / вверх",
    2: "east / вправо",
    3: "west / влево",
    4: "pickup / посадить пассажира",
    5: "dropoff / высадить пассажира",
}


def ensure_taxi_v3_registered() -> None:
    """Register Taxi-v3 for Gymnasium versions that expose only Taxi-v4."""

    if ENV_NAME in registry:
        return

    register(
        id=ENV_NAME,
        entry_point="gymnasium.envs.toy_text.taxi:TaxiEnv",
        reward_threshold=8,
        max_episode_steps=200,
    )


def make_taxi_env(render_mode: str | None = None) -> gym.Env[Any, Any]:
    """Create Taxi-v3 and hide the deprecation warning in newer Gymnasium."""

    ensure_taxi_v3_registered()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=f".*{ENV_NAME}.*",
            category=DeprecationWarning,
        )
        return gym.make(ENV_NAME, render_mode=render_mode)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for demonstration mode."""

    parser = argparse.ArgumentParser(
        description="Run a trained Q-Learning agent in Taxi-v3."
    )
    parser.add_argument("--q-table-path", type=Path, default=DEFAULT_Q_TABLE_PATH)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps-per-episode", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--render-mode",
        choices=("ansi", "human", "none"),
        default="ansi",
        help="Use ansi for terminal output, human for a window, none for metrics only.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Pause between rendered steps in seconds.",
    )
    return parser.parse_args()


def load_q_table(q_table_path: Path) -> np.ndarray:
    """Load the saved Q-table from disk."""

    if not q_table_path.exists():
        raise FileNotFoundError(
            f"Q-table not found: {q_table_path}. "
            "Run `python train.py` before `python test.py`."
        )
    return np.load(q_table_path)


def validate_environment_and_q_table(
    env: gym.Env[Any, Any],
    q_table: np.ndarray,
) -> None:
    """Ensure that the Q-table shape matches the Taxi-v3 spaces."""

    if not isinstance(env.observation_space, Discrete):
        raise TypeError("Taxi-v3 is expected to have a discrete observation space.")
    if not isinstance(env.action_space, Discrete):
        raise TypeError("Taxi-v3 is expected to have a discrete action space.")

    expected_shape = (int(env.observation_space.n), int(env.action_space.n))
    if q_table.shape != expected_shape:
        raise ValueError(
            f"Q-table shape is {q_table.shape}, expected {expected_shape}."
        )


def render_frame(env: gym.Env[Any, Any], render_mode: str) -> None:
    """Render one frame in the selected mode."""

    if render_mode == "none":
        return

    rendered_frame = env.render()
    if render_mode == "ansi" and rendered_frame is not None:
        print(rendered_frame)


def run_episode(
    env: gym.Env[Any, Any],
    q_table: np.ndarray,
    max_steps_per_episode: int,
    render_mode: str,
    sleep_seconds: float,
    seed: int,
) -> tuple[float, int, bool]:
    """Run one greedy-policy episode and return reward, steps and success flag."""

    state, _ = env.reset(seed=seed)
    state = int(state)
    total_reward = 0.0
    steps_taken = max_steps_per_episode
    successful_delivery = False

    render_frame(env, render_mode)

    for step in range(1, max_steps_per_episode + 1):
        action = int(np.argmax(q_table[state]))
        next_state, reward, terminated, truncated, _ = env.step(action)
        action_name = ACTION_NAMES.get(action, str(action))

        total_reward += float(reward)
        state = int(next_state)
        steps_taken = step

        if render_mode != "none":
            print(
                f"Step {step:>3}: action={action} ({action_name}), "
                f"reward={reward}"
            )
            render_frame(env, render_mode)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        if terminated or truncated:
            successful_delivery = bool(terminated)
            break

    return total_reward, steps_taken, successful_delivery


def main() -> None:
    """Entry point for testing a trained agent."""

    args = parse_args()
    q_table = load_q_table(args.q_table_path)

    render_mode = None if args.render_mode == "none" else args.render_mode
    env = make_taxi_env(render_mode=render_mode)
    env.action_space.seed(args.seed)
    validate_environment_and_q_table(env, q_table)

    total_rewards: list[float] = []
    episode_steps: list[int] = []
    successful_deliveries = 0
    verbose_episode_output = args.render_mode != "none" or args.episodes <= 10

    if not verbose_episode_output:
        print(f"Running {args.episodes} evaluation episodes without rendering...")

    for episode in range(1, args.episodes + 1):
        if verbose_episode_output:
            print(f"\nEpisode {episode}/{args.episodes}")

        reward, steps, success = run_episode(
            env=env,
            q_table=q_table,
            max_steps_per_episode=args.max_steps_per_episode,
            render_mode=args.render_mode,
            sleep_seconds=args.sleep,
            seed=args.seed + episode,
        )

        total_rewards.append(reward)
        episode_steps.append(steps)
        successful_deliveries += int(success)

        if verbose_episode_output:
            status = "success" if success else "not delivered"
            print(f"Episode result: {status}, reward={reward:.2f}, steps={steps}")

    env.close()

    print("\nEvaluation summary")
    print(f"Mean reward: {np.mean(total_rewards):.2f}")
    print(f"Successful deliveries: {successful_deliveries / args.episodes * 100:.2f}%")
    print(f"Mean steps: {np.mean(episode_steps):.2f}")


if __name__ == "__main__":
    main()
