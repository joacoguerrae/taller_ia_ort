# based on https://gymnasium.farama.org/introduction/create_custom_env/
from enum import Enum
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, size=5, max_steps=100, stochastic=False):
        super().__init__()
        self.size = size  
        self.steps = 0
        self.max_steps = max_steps
        self.window_size = 512  
        self.stochastic = stochastic  # Flag para modo estocástico

        self.observation_space = spaces.Dict(
            {
                "pos": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
        self._agent_location = np.array([-1, -1], dtype=int)

        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            Actions.RIGHT.value: np.array([1, 0]),
            Actions.UP.value: np.array([0, -1]),
            Actions.LEFT.value: np.array([-1, 0]),
            Actions.DOWN.value: np.array([0, 1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self._terminal_states = [np.array([0, 0]), np.array([self.size - 1, self.size - 1])]

        # Inicializar dinámicas del ambiente
        self._init_transition_dynamics()

    def _init_transition_dynamics(self):
        """Define las probabilidades de transición para los dos modos (determinista o estocástico)."""
        self.p = {}
        for x in range(self.size):
            for y in range(self.size):
                for a in range(4):  
                    if (x, y) in [(0, 0), (self.size - 1, self.size - 1)]:  
                        self.p[((x, y), a)] = [(1.0, (x, y), 0.0)]
                        continue
                    
                    delta = self._action_to_direction[a]
                    new_x = np.clip(x + delta[0], 0, self.size - 1).item()
                    new_y = np.clip(y + delta[1], 0, self.size - 1).item()
                    
                    if not self.stochastic:
                        # Determinista: 100% probabilidad de moverse a la dirección elegida
                        self.p[((x, y), a)] = [(1.0, (new_x, new_y), -1.0)]
                    else:
                        # Estocástico: 80% acción elegida, 10% arriba, 10% abajo (si aplica)
                        alt_actions = [Actions.UP.value, Actions.DOWN.value]
                        alt_transitions = []
                        
                        for alt_a in alt_actions:
                            alt_dx, alt_dy = self._action_to_direction[alt_a]
                            alt_x = np.clip(x + alt_dx, 0, self.size - 1).item()
                            alt_y = np.clip(y + alt_dy, 0, self.size - 1).item()
                            alt_transitions.append((0.10, (alt_x, alt_y), -1.0))
                        
                        self.p[((x, y), a)] = [(0.80, (new_x, new_y), -1.0)] + alt_transitions

    def step(self, action):
        self.steps += 1

        transitions = self.p[(tuple(self._agent_location), action)]
        probabilities, next_states, rewards = zip(*[(prob, state, reward) for prob, state, reward in transitions])

        choice_idx = np.random.choice(len(next_states), p=probabilities)
        self._agent_location = np.array(next_states[choice_idx])

        terminated = any(np.array_equal(self._agent_location, t) for t in self._terminal_states)
        truncated = self.steps >= self.max_steps
        reward = rewards[choice_idx]
        observation = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        return {"pos": self._agent_location}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        while True:
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            if not any(np.array_equal(self._agent_location, t) for t in self._terminal_states):
                break

        observation = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Draw terminal states
        for target in self._terminal_states:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * target,
                    (pix_square_size, pix_square_size),
                ),
            )
        
        # Draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Add gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
            
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


register(
    id="gymnasium_env/GridWorld-v0",  # Versión Determinista
    entry_point=GridWorldEnv,
    kwargs={"stochastic": False}
)

register(
    id="gymnasium_env/GridWorld_stochastic-v0",  # Versión Estocástica
    entry_point=GridWorldEnv,
    kwargs={"stochastic": True}
)

