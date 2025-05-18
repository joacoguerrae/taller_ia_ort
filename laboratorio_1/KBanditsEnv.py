import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
from typing import Optional, Callable, List, Dict, Tuple

# source: https://github.com/BRJoaquin/10-armed-bandit-gymnasium


class BanditEnv(gym.Env):
    """
    Base class for Bandit environments using Gymnasium.
      - k: the number of arms.
      - r_dist: a list (of length k) of callables (functions)
                each returning a float reward when called.
      - render_mode: must be either None or "human".

    We track:
      - current_step: how many actions have been taken.
      - balance: total cumulative reward so far.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        k: int,
        r_dist: List[Callable[[], float]],
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        # validation
        assert isinstance(k, int) and k > 0, "k must be a positive integer"
        assert len(r_dist) == k, "r_dist must be a list of callables of length k"
        assert render_mode is None or render_mode in self.metadata["render_modes"], (
            "render_mode must be either None or 'human'"
        )

        self.k = k
        self.r_dist = r_dist
        self.render_mode = render_mode

        # Gym spaces
        # The action space is a Discrete space of size k (use each arm)
        self.action_space = spaces.Discrete(self.k)
        # For bandits, we typically have a trivial or dummy observation. Use a discrete space of size 1.
        self.observation_space = spaces.Discrete(1)

        # Internal states
        self.current_step = 0
        self.balance = 0.0

    def _get_obs(self) -> int:
        """
        In a bandit, there's typically no meaningful state.
        Return a dummy observation (0).
        """
        return 0

    def _get_info(self) -> Dict[str, float | int]:
        """
        Provide additional info: the current step count and the total balance (accumulated reward).
        """
        return {"step": self.current_step, "balance": self.balance}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[int, Dict[str, float | int]]:
        """
        Reset the environment state. For a bandit, this usually just resets counters
        because there's no environment state to reset otherwise.
        """
        super().reset(seed=seed)
        np.random.seed(seed)
        self.current_step = 0
        self.balance = 0.0
        if self.render_mode == "human":
            print("Environment reset.")
        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> Tuple[int, float, bool, bool, Dict[str, float | int]]:
        """
        1) Executes the chosen action (which arm to pull).
        2) Samples reward by calling the corresponding function in r_dist.
        3) Updates balance and step.
        4) Returns observation, reward, terminated, truncated, info.
        """
        reward = self.r_dist[action]()  # Call the function for this arm
        self.balance += reward
        self.current_step += 1

        observation = self._get_obs()
        info = self._get_info()

        terminated = False
        truncated = False

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """
        If render_mode == 'human', print out current step and balance so far.
        """
        if self.render_mode == "human":
            print(f"[Render] Step: {self.current_step} | Balance: {self.balance}:.2f")

    def close(self) -> None:
        """
        Close any open resources (files, windows, etc.).
        Nothing special for a basic bandit environment.
        """
        pass


class BanditTenArmedGaussian(BanditEnv):
    """
    A 10-armed Gaussian bandit environment based on Reinforcement Learning: An Introduction (Sutton and Barto) book.
      - k = 10
      - Each arm i has a true value q*(i) ~ N(0, 1).
      - Actual reward ~ N(q*(i), 1) each time it is pulled.

    The r_dist is thus a list of lambdas/functions,
    each returning np.random.normal(mean_i, 1).
    """

    def __init__(self, print_true_values: bool = False, render_mode: Optional[str] = None) -> None:
        k = 10  # 10-armed
        means = np.random.normal(loc=0.0, scale=1.0, size=k)
        
        if print_true_values:
            print("True means (q*):", means)

        # Create a list of callables, each generating a reward ~ N(mean_i, 1).
        r_dist: List[Callable[[], float]] = [
            (lambda mean=mean_i: float(np.random.normal(loc=mean, scale=1.0)))
            for mean_i in means
        ]

        self.true_means = means
        super().__init__(k=k, r_dist=r_dist, render_mode=render_mode)
        
        
# Register the environments with OpenAI Gym
register(
    id="k_bandits_env/KBandits-v0",
    entry_point=BanditEnv,
)
register(
    id="k_bandits_env/KBanditsGaussian-v0",
    entry_point=BanditTenArmedGaussian,
)