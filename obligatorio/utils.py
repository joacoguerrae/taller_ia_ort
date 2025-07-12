import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.wrappers import (
    TransformReward,
    RecordVideo,
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
    AtariPreprocessing,
    FrameStackObservation,
    RecordEpisodeStatistics,
)


def process_state(obs, device="cuda"):
    """
    Preprocess the state to be used as input for the model (transform to tensor).
    """
    return torch.tensor(obs, dtype=torch.float32, device=device) / 255.0


def show_observation(observation):
    dimension = observation.shape
    if len(dimension) == 3:
        if dimension[2] == 3:
            plt.imshow(observation)
        elif dimension[2] == 1:
            plt.imshow(observation[:, :, 0], cmap="gray")
    elif len(dimension) == 2:
        plt.imshow(observation, cmap="gray")
    else:
        raise ValueError("Invalid observation shape")
    plt.show()


def show_observation_stack(observation):
    frames = observation.shape[0]
    for i in range(frames):
        show_observation(observation[i])


class FireOnLifeLostWrapper(gymnasium.Wrapper):
    """Presiona FIRE automáticamente tras reset y tras cada pérdida de vida."""

    def __init__(self, env):
        super().__init__(env)
        self._prev_lives = None

    def reset(self, **kwargs):
        # 1) Reset normal
        obs, info = self.env.reset(**kwargs)
        # 2) Inyectar FIRE para arrancar la partida
        obs, _, terminated, truncated, info = self.env.step(1)
        # Si por alguna razón el juego acabó (raro), reinicia otra vez
        if terminated or truncated:
            return self.reset(**kwargs)
        # 3) Guarda el número de vidas inicial
        self._prev_lives = info.get("lives")
        return obs, info

    def step(self, action):
        # 1) Paso normal del agente
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 2) Detecta pérdida de vida
        current_lives = info.get("lives", self._prev_lives)
        if (current_lives < self._prev_lives) and not (terminated or truncated):
            # 3) Inyecta FIRE para reanudar tras perder vida
            obs, fire_reward, terminated, truncated, info = self.env.step(1)
            reward += fire_reward  # opcional: sumar recompensa de FIRE
        # 4) Actualiza contador de vidas
        self._prev_lives = current_lives
        return obs, reward, terminated, truncated, info


def make_env(
    env_name: str,
    render_mode: str = "rgb_array",
    # Video
    video_folder: str | None = "./videos",
    name_prefix: str = "",
    record_every: int | None = None,
    # Preprocesado
    grayscale: bool = False,
    screen_size: int = 84,
    stack_frames: int = 4,
    skip_frames: int = 4,
    fire_on_life_lost: bool = True,
) -> gymnasium.Env:
    env = gymnasium.make(env_name, render_mode=render_mode, frameskip=1)

    if video_folder is not None and record_every is not None:
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=name_prefix,
            episode_trigger=lambda ep: ep % record_every == 0,
            fps=env.metadata.get("render_fps", 30) * skip_frames,
        )

    # env = FireOnLifeLostWrapper(env)

    env = AtariPreprocessing(
        env,
        noop_max=10,
        frame_skip=skip_frames,
        screen_size=screen_size,
        grayscale_obs=grayscale,
        grayscale_newaxis=False,
    )

    # stack frames
    env = FrameStackObservation(env, stack_size=stack_frames)

    # clip rewards
    sign_fn = lambda r: 1 if r > 0 else (-1 if r < 0 else 0)
    env = TransformReward(env, sign_fn)

    # fire on life lost
    if fire_on_life_lost:
        env = FireOnLifeLostWrapper(env)

    return env


def plot_rewards_and_max_values(
    rewards, max_values, agent_name, title="Training Progress"
):
    """
    Plot the rewards and maximum Q-values over episodes.
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=16)

    # Grafico de recompensas
    if len(rewards) % 100 != 0:
        print(
            "Advertencia: La longitud de la lista de recompensas no es un múltiplo de 100. Se descartarán los últimos episodios."
        )
        num_episodes = len(rewards) // 100 * 100
        rewards = rewards[:num_episodes]

    avg_rewards = np.mean(np.array(rewards).reshape(-1, 100), axis=1)
    episodes_rewards = np.arange(100, len(rewards) + 1, 100)

    axs[0].plot(episodes_rewards, avg_rewards, color="blue", marker="o", linestyle="-")
    axs[0].set_title("Recompensa Promedio por Episodios")
    axs[0].set_xlabel("Episodios")
    axs[0].set_ylabel("Recompensa Promedio")
    axs[0].grid(True)

    # Grafico de valores Q máximos
    episodes_q = list(max_values.keys())
    q_values = list(max_values.values())

    axs[1].plot(episodes_q, q_values, color="orange", marker="o", linestyle="-")
    axs[1].set_title("Valor Q Máximo Promedio")
    axs[1].set_xlabel("Episodios")
    axs[1].set_ylabel("Valor Q Promedio")
    axs[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(f"training_progress_{agent_name}.png")
    plt.close(fig)
