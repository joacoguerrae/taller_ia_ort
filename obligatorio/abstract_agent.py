import torch
import torch.nn as nn
import numpy as np
from replay_memory import ReplayMemory, Transition
from abc import ABC, abstractmethod
from tqdm import tqdm
import random
from utils import FireOnLifeLostWrapper


class Agent(ABC):
    def __init__(
        self,
        gym_env,
        obs_processing_func,
        memory_buffer_size,
        batch_size,
        learning_rate,
        gamma,
        epsilon_i,
        epsilon_f,
        epsilon_anneal_steps,
        episode_block,
        device,
    ):
        self.device = device

        # Funcion phi para procesar los estados.
        self.state_processing_function = obs_processing_func

        # Asignarle memoria al agente
        self.memory = ReplayMemory(memory_buffer_size)

        self.env = gym_env

        # Hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal_steps = epsilon_anneal_steps

        self.episode_block = episode_block

        self.total_steps = 0

    def train(
        self, number_episodes=50_000, max_steps_episode=10_000, max_steps=1_000_000
    ):
        rewards = []
        total_steps = 0

        metrics = {"reward": 0.0, "epsilon": self.epsilon_i, "steps": 0}

        pbar = tqdm(range(number_episodes), desc="Entrenando", unit="episode")

        for ep in pbar:
            if total_steps > max_steps:
                break

            # Observar estado inicial como indica el algoritmo
            state, _ = self.env.reset()
            state_phi = self.state_processing_function(state)
            current_episode_reward = 0.0
            current_episode_steps = 0
            done = False

            # Bucle principal de pasos dentro de un episodio
            for _ in range(max_steps):
                # TODO: Seleccionar acción epsilon-greedy usando select_action()
                # TODO: Ejecutar action = env.step(action)
                # TODO: Procesar next_state con state_processing_function
                # TODO: Acumular reward y actualizar total_steps, current_episode_steps
                # TODO: Almacenar transición en replay memory
                # TODO: Llamar a update_weights() para entrenar modelo
                # TODO: Actualizar state y state_phi al siguiente estado
                # TODO: Comprobar condición de done o límite de pasos de episodio y break
                action = self.select_action(state_phi, total_steps)

                next_state, reward, terminated, truncated, info = self.env.step(action)

                next_state_phi = self.state_processing_function(next_state)
                current_episode_reward += reward
                total_steps += 1
                current_episode_steps += 1
                done = terminated or truncated
                self.memory.add(state_phi, action, reward, done, next_state_phi)

                self.update_weights()

                state = next_state
                state_phi = next_state_phi
                if done or current_episode_steps >= max_steps_episode:
                    break

            # Registro de métricas y progreso
            rewards.append(current_episode_reward)
            metrics["reward"] = np.mean(rewards[-self.episode_block :])
            metrics["epsilon"] = self.compute_epsilon(total_steps)
            metrics["steps"] = total_steps
            pbar.set_postfix(metrics)

        # Guardar el modelo entrenado
        torch.save(self.policy_net.state_dict(), "GenericDQNAgent.dat")

        return rewards

    def compute_epsilon(self, steps_so_far):
        """
        Compute el valor de epsilon a partir del número de pasos dados hasta ahora.
        """
        if steps_so_far < self.epsilon_anneal_steps:
            epsilon = self.epsilon_i - (self.epsilon_i - self.epsilon_f) * (
                steps_so_far / self.epsilon_anneal_steps
            )
        else:
            epsilon = self.epsilon_f
        return epsilon

    def play(self, env, episodes=1):
        """
        Modo evaluación: ejecutar episodios sin actualizar la red.
        """
        self.policy_net.eval()
        for ep in range(episodes):
            print(f"Episode {ep + 1}/{episodes}")
            state, _ = env.reset()
            with torch.no_grad():
                # Procesar el estado inicial
                state_tensor = self.state_processing_function(state).unsqueeze(0)
            done = False
            step = 1
            while not done:
                # TODO: seleccionar acción sin exploración
                # TODO: ejecutar acción y actualizar estado

                print(f"Step: {step} ")
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].item()

                next_state, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action} ")
                next_state = self.state_processing_function(next_state).unsqueeze(0)
                state = next_state
                done = terminated or truncated
                if step > 300:
                    print(done)
                step += 1
        self.policy_net.train()

    @abstractmethod
    def select_action(self, state, current_steps, train=True):
        """
        Selecciona una acción a partir del estado actual. Si train=False, se selecciona la acción greedy.
        Si train=True, se selecciona la acción epsilon-greedy.

        Args:
            state: El estado actual del entorno.
            current_steps: El número de pasos actuales. Determina el valor de epsilon.
            train: Si True, se selecciona la acción epsilon-greedy. Si False, se selecciona la acción greedy.
        """
        pass

    @abstractmethod
    def update_weights(self):
        pass
