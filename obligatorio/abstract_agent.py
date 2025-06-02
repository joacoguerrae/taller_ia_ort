import torch
import torch.nn as nn
import numpy as np
from replay_memory import ReplayMemory, Transition
from abc import ABC, abstractmethod
from tqdm import tqdm
import random
import os
from utils import FireOnLifeLostWrapper
import matplotlib


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
        checkpoint_interval=1000,  # Add checkpoint interval
    ):
        self.device = device

        # Funcion phi para procesar los estados.
        self.state_processing_function = obs_processing_func

        # Asignarle memoria al agente
        self.memory = ReplayMemory(memory_buffer_size)

        self.env = gym_env

        # Handle both single and vectorized environments
        if hasattr(gym_env, 'num_envs'):
            # This is a vectorized environment
            self.action_space = gym_env.single_action_space
            self.is_vectorized = True
        else:
            # This is a single environment
            self.action_space = gym_env.action_space
            self.is_vectorized = False

        # Hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal_steps = epsilon_anneal_steps

        self.episode_block = episode_block

        self.total_steps = 0
        self.checkpoint_interval = checkpoint_interval  # Store checkpoint interval

    def train(
        self,
        number_episodes=10_000,
        max_steps_episode=10_000,
        max_steps=1_000_000,
    ):
        rewards = []
        total_steps = 0

        pbar = tqdm(range(number_episodes), desc="Training", unit="episode")

        for ep in pbar:
            if total_steps > max_steps:
                break

            state, _ = self.env.reset()
            state_phi = self.state_processing_function(state)
            current_episode_reward = 0.0
            current_episode_steps = 0
            episode_losses = []
            done = False

            while not done and current_episode_steps < max_steps_episode:
                action = self.select_action(state_phi, total_steps)

                next_state, reward, terminated, truncated, info = self.env.step(action)

                next_state_phi = self.state_processing_function(next_state)
                current_episode_reward += reward
                total_steps += 1
                current_episode_steps += 1
                done = terminated or truncated
                self.memory.add(state_phi, action, reward, done, next_state_phi)

                loss = self.update_weights()
                if loss is not None:
                    episode_losses.append(loss)

                state = next_state
                state_phi = next_state_phi
                if done or current_episode_steps >= max_steps_episode:
                    break

            rewards.append(current_episode_reward)

            metrics = {
                "reward": np.mean(rewards[-self.episode_block :]),
                "epsilon": self.compute_epsilon(total_steps),
                "steps": total_steps,
            }

            pbar.set_postfix(metrics)

            if (ep + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_{ep + 1}")

        return rewards

    # In obligatorio/abstract_agent.py, replace the existing train_vec() method with this:

    def train_vec(self, max_steps=1_000_000):
        """
        Trains the agent using a vectorized environment with a clean progress bar
        displaying average loss and epsilon.
        """
        if not self.is_vectorized:
            print("Error: train_vec() can only be called on a vectorized environment.")
            return

        # We only need to track the history of losses now
        losses_history = []
        latest_avg_loss = 0.0
        
        self.total_steps = 0

        states, _ = self.env.reset()
        states_phi = self.state_processing_function(states)

        pbar = tqdm(total=max_steps, desc="Vectorized Training", unit="step")

        while self.total_steps < max_steps:
            actions = self.select_action_vec(states_phi, self.total_steps, train=True)
            
            # We no longer need the 'infos' dictionary from the step
            next_states, rewards, terminateds, truncateds, _ = self.env.step(actions)
            dones = np.logical_or(terminateds, truncateds)
            
            next_states_phi = self.state_processing_function(next_states)

            for i in range(self.env.num_envs):
                self.memory.add(states_phi[i], actions[i], rewards[i], dones[i], next_states_phi[i])

            states_phi = next_states_phi
            self.total_steps += self.env.num_envs
            pbar.update(self.env.num_envs)

            # Update weights and store the loss
            loss = self.update_weights()
            if loss is not None:
                losses_history.append(loss)
                latest_avg_loss = np.mean(losses_history[-100:])

            # The updated, cleaner metrics dictionary
            metrics = {
                "steps": f"{self.total_steps}/{max_steps}",
                "avg_loss": round(latest_avg_loss, 5),
                "epsilon": round(self.compute_epsilon(self.total_steps), 3)
            }
            pbar.set_postfix(metrics)

        pbar.close()
        self.env.close()



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

        for ep in range(episodes):
            state, _ = env.reset()
            state_tensor = self.state_processing_function(state)
            done = False
            while not done:
                # TODO: seleccionar acción sin exploración
                # TODO: ejecutar acción y actualizar estado

                action = self.select_action(
                    state_tensor, current_steps=self.total_steps, train=False
                )

                next_state, reward, terminated, truncated, info = env.step(action)

                next_state = self.state_processing_function(next_state)
                state_tensor = next_state
                done = terminated or truncated

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
    def select_action_vec(self, state, current_steps, train=True):
        """
        Selects a batch of actions for a batch of states from a vectorized environment.
        """
        pass

    @abstractmethod
    def update_weights(self):
        pass

    @abstractmethod
    def save_checkpoint(self, path):
        """
        Saves a checkpoint of the agent's weights.
        """
        pass

    @abstractmethod
    def load_checkpoint(self, path):
        """
        Loads a checkpoint and restores the agent's state.

        Args:
            path (str): Path to the checkpoint file

        Returns:
            bool: True if checkpoint was loaded successfully, False otherwise
        """
        pass