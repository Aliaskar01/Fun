"""Train a tabular Q-Learning agent in the Gymnasium Taxi-v3 environment."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib
import numpy as np
from gymnasium.envs.registration import register, registry
from gymnasium.spaces import Discrete

# Backend Agg позволяет сохранять графики без открытия GUI-окна.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ENV_NAME = "Taxi-v3"
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_Q_TABLE_PATH = PROJECT_DIR / "q_table.npy"
DEFAULT_PLOT_PATH = PROJECT_DIR / "training_rewards.png"


@dataclass(frozen=True)
class TrainConfig:
    """Гиперпараметры и пути, необходимые для обучения агента."""

    episodes: int = 20_000
    max_steps_per_episode: int = 200
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 1.0
    min_epsilon: float = 0.05
    epsilon_decay: float = 0.9995
    seed: int = 42
    average_window: int = 100
    evaluation_episodes: int = 100
    q_table_path: Path = DEFAULT_Q_TABLE_PATH
    plot_path: Path = DEFAULT_PLOT_PATH


def parse_args() -> TrainConfig:
    """Parse command-line arguments and build the training config."""

    parser = argparse.ArgumentParser(
        description="Train a Q-Learning agent on Taxi-v3 with Gymnasium."
    )
    parser.add_argument("--episodes", type=int, default=TrainConfig.episodes)
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=TrainConfig.max_steps_per_episode,
    )
    parser.add_argument("--alpha", type=float, default=TrainConfig.alpha)
    parser.add_argument("--gamma", type=float, default=TrainConfig.gamma)
    parser.add_argument("--epsilon", type=float, default=TrainConfig.epsilon)
    parser.add_argument(
        "--min-epsilon",
        type=float,
        default=TrainConfig.min_epsilon,
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=TrainConfig.epsilon_decay,
    )
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument(
        "--average-window",
        type=int,
        default=TrainConfig.average_window,
    )
    parser.add_argument(
        "--evaluation-episodes",
        type=int,
        default=TrainConfig.evaluation_episodes,
    )
    parser.add_argument(
        "--q-table-path",
        type=Path,
        default=DEFAULT_Q_TABLE_PATH,
        help="Where to save the learned Q-table.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Where to save the training reward plot.",
    )

    args = parser.parse_args()
    config = TrainConfig(
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        min_epsilon=args.min_epsilon,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
        average_window=args.average_window,
        evaluation_episodes=args.evaluation_episodes,
        q_table_path=args.q_table_path,
        plot_path=args.plot_path,
    )
    validate_config(config)
    return config


def validate_config(config: TrainConfig) -> None:
    """Fail fast if hyperparameters are outside meaningful ranges."""

    if config.episodes <= 0:
        raise ValueError("episodes must be greater than 0.")
    if config.max_steps_per_episode <= 0:
        raise ValueError("max_steps_per_episode must be greater than 0.")
    if config.average_window <= 0:
        raise ValueError("average_window must be greater than 0.")
    if config.evaluation_episodes <= 0:
        raise ValueError("evaluation_episodes must be greater than 0.")
    if not 0 < config.alpha <= 1:
        raise ValueError("alpha must be in the interval (0, 1].")
    if not 0 <= config.gamma <= 1:
        raise ValueError("gamma must be in the interval [0, 1].")
    if not 0 <= config.epsilon <= 1:
        raise ValueError("epsilon must be in the interval [0, 1].")
    if not 0 <= config.min_epsilon <= config.epsilon:
        raise ValueError("min_epsilon must be between 0 and epsilon.")
    if not 0 < config.epsilon_decay <= 1:
        raise ValueError("epsilon_decay must be in the interval (0, 1].")


def ensure_taxi_v3_registered() -> None:
    """Register Taxi-v3 for new Gymnasium versions where only Taxi-v4 is listed.

    Gymnasium 1.3+ deprecates the public Taxi-v3 id in favor of Taxi-v4. The
    assignment explicitly requires Taxi-v3, so we keep that environment id while
    using Gymnasium's built-in TaxiEnv implementation with deterministic default
    settings.
    """

    if ENV_NAME in registry:
        return

    register(
        id=ENV_NAME,
        entry_point="gymnasium.envs.toy_text.taxi:TaxiEnv",
        reward_threshold=8,
        max_episode_steps=200,
    )


def make_taxi_env(render_mode: str | None = None) -> gym.Env[Any, Any]:
    """Create Taxi-v3 while staying compatible with current Gymnasium releases."""

    ensure_taxi_v3_registered()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=f".*{ENV_NAME}.*",
            category=DeprecationWarning,
        )
        return gym.make(ENV_NAME, render_mode=render_mode)


def get_discrete_space_sizes(env: gym.Env[Any, Any]) -> tuple[int, int]:
    """Return the number of states and actions for a discrete environment."""

    if not isinstance(env.observation_space, Discrete):
        raise TypeError("Taxi-v3 is expected to have a discrete observation space.")
    if not isinstance(env.action_space, Discrete):
        raise TypeError("Taxi-v3 is expected to have a discrete action space.")

    return int(env.observation_space.n), int(env.action_space.n)


def select_action(
    state: int,
    q_table: np.ndarray,
    epsilon: float,
    env: gym.Env[Any, Any],
    rng: np.random.Generator,
) -> int:
    """Choose an action using epsilon-greedy exploration.

    With probability epsilon the agent explores a random action. Otherwise,
    it exploits the current Q-table and chooses the action with the highest
    estimated value for the current state.
    """

    if rng.random() < epsilon:
        return int(env.action_space.sample())
    return int(np.argmax(q_table[state]))


def update_q_value(
    q_table: np.ndarray,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    terminated: bool,
    config: TrainConfig,
) -> None:
    """Apply the Q-Learning update rule for one transition."""

    old_q_value = q_table[state, action]

    # Если эпизод завершен успехом, будущей награды уже нет.
    next_best_q_value = 0.0 if terminated else np.max(q_table[next_state])

    td_target = reward + config.gamma * next_best_q_value
    td_error = td_target - old_q_value
    q_table[state, action] = old_q_value + config.alpha * td_error


def train_agent(config: TrainConfig) -> tuple[np.ndarray, np.ndarray]:
    """Train the Q-table and return reward history for plotting."""

    env = make_taxi_env()
    env.action_space.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    n_states, n_actions = get_discrete_space_sizes(env)

    # Q-таблица хранит оценку качества каждого действия в каждом состоянии.
    q_table = np.zeros((n_states, n_actions), dtype=np.float64)
    episode_rewards = np.zeros(config.episodes, dtype=np.float64)
    epsilon = config.epsilon

    for episode in range(1, config.episodes + 1):
        if episode == 1:
            state, _ = env.reset(seed=config.seed)
        else:
            state, _ = env.reset()

        state = int(state)
        total_reward = 0.0

        for _ in range(config.max_steps_per_episode):
            action = select_action(state, q_table, epsilon, env, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = int(next_state)

            update_q_value(
                q_table=q_table,
                state=state,
                action=action,
                reward=float(reward),
                next_state=next_state,
                terminated=terminated,
                config=config,
            )

            state = next_state
            total_reward += float(reward)

            if terminated or truncated:
                break

        episode_rewards[episode - 1] = total_reward
        epsilon = max(config.min_epsilon, epsilon * config.epsilon_decay)

        if episode % config.average_window == 0 or episode == config.episodes:
            start = max(0, episode - config.average_window)
            average_reward = float(np.mean(episode_rewards[start:episode]))
            print(
                f"Episode {episode:>6}/{config.episodes}: "
                f"mean reward last {episode - start:>3} = {average_reward:>7.2f}, "
                f"epsilon = {epsilon:.4f}"
            )

    env.close()
    return q_table, episode_rewards


def moving_average(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate a simple moving average and matching episode indexes."""

    if len(values) < window:
        indexes = np.arange(1, len(values) + 1)
        return indexes, values.copy()

    weights = np.ones(window, dtype=np.float64) / window
    averaged_values = np.convolve(values, weights, mode="valid")
    indexes = np.arange(window, len(values) + 1)
    return indexes, averaged_values


def plot_rewards(
    episode_rewards: np.ndarray,
    average_window: int,
    plot_path: Path,
) -> None:
    """Save a readable Reward vs Episode chart."""

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")

    episodes = np.arange(1, len(episode_rewards) + 1)
    average_indexes, average_rewards = moving_average(
        episode_rewards,
        average_window,
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        episodes,
        episode_rewards,
        color="#94a3b8",
        linewidth=0.8,
        alpha=0.35,
        label="Reward per episode",
    )
    ax.plot(
        average_indexes,
        average_rewards,
        color="#2563eb",
        linewidth=2.4,
        label=f"Mean reward over {average_window} episodes",
    )
    ax.axhline(0, color="#111827", linewidth=1.0, alpha=0.45)
    ax.set_title("Taxi-v3 Q-Learning Training Progress", fontsize=15, pad=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True, alpha=0.35)
    fig.tight_layout()

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def evaluate_agent(
    q_table: np.ndarray,
    episodes: int,
    max_steps_per_episode: int,
    seed: int,
) -> dict[str, float]:
    """Evaluate the greedy policy learned by the agent."""

    env = make_taxi_env()
    env.action_space.seed(seed)

    total_rewards: list[float] = []
    episode_steps: list[int] = []
    successful_deliveries = 0

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        state = int(state)
        total_reward = 0.0
        steps_taken = max_steps_per_episode

        for step in range(1, max_steps_per_episode + 1):
            action = int(np.argmax(q_table[state]))
            next_state, reward, terminated, truncated, _ = env.step(action)

            state = int(next_state)
            total_reward += float(reward)
            steps_taken = step

            if terminated or truncated:
                successful_deliveries += int(terminated)
                break

        total_rewards.append(total_reward)
        episode_steps.append(steps_taken)

    env.close()

    return {
        "mean_reward": float(np.mean(total_rewards)),
        "success_rate": successful_deliveries / episodes * 100.0,
        "mean_steps": float(np.mean(episode_steps)),
    }


def save_q_table(q_table: np.ndarray, q_table_path: Path) -> None:
    """Persist the trained Q-table as a NumPy binary file."""

    q_table_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(q_table_path, q_table)


def main() -> None:
    """Entry point for training and evaluation."""

    config = parse_args()
    print(f"Training Q-Learning agent in {ENV_NAME}")
    print(
        "Hyperparameters: "
        f"alpha={config.alpha}, gamma={config.gamma}, "
        f"epsilon={config.epsilon}, epsilon_decay={config.epsilon_decay}, "
        f"min_epsilon={config.min_epsilon}"
    )

    q_table, episode_rewards = train_agent(config)
    save_q_table(q_table, config.q_table_path)
    plot_rewards(episode_rewards, config.average_window, config.plot_path)

    metrics = evaluate_agent(
        q_table=q_table,
        episodes=config.evaluation_episodes,
        max_steps_per_episode=config.max_steps_per_episode,
        seed=config.seed + 10_000,
    )

    print("\nTraining finished.")
    print(f"Q-table saved to: {config.q_table_path}")
    print(f"Reward plot saved to: {config.plot_path}")
    print("\nEvaluation on greedy policy:")
    print(f"Mean reward: {metrics['mean_reward']:.2f}")
    print(f"Successful deliveries: {metrics['success_rate']:.2f}%")
    print(f"Mean steps: {metrics['mean_steps']:.2f}")


if __name__ == "__main__":
    main()
