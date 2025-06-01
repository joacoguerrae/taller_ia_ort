import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_memory import ReplayMemory, Transition
import numpy as np
from abstract_agent import Agent
import random


class DoubleDQNAgent(Agent):
    def __init__(
        self,
        gym_env,
        model_a,
        model_b,
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
        sync_target=1000,
    ):
        super().__init__(
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
        )
        # Guardar entorno y función de preprocesamiento
        self.env = gym_env
        self.state_processing_function = obs_processing_func
        # Inicializar online_net (model_a) y target_net (model_b) en device
        self.policy_net = model_a.to(device)
        self.target_net = model_b.to(device)
        # Configurar función de pérdida MSE y optimizador Adam para online_net
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )
        # Crear replay memory de tamaño buffer_size
        self.memory = ReplayMemory(memory_buffer_size)
        # Almacenar batch_size, gamma, parámetros de epsilon y sync_target
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal_steps = epsilon_anneal_steps
        # Inicializar contador de pasos para sincronizar target
        # Initialize target network with same weights as policy network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target net to eval mode

        # Use a smaller sync_target value at start
        self.sync_target = sync_target
        self.steps_done = 0
        # Inicializar epsilon

    def select_action(self, state, current_steps, train=True):
        """Select action using epsilon-greedy policy for single or vectorized environments"""
        is_batched = len(state.shape) == 4  # Check if state is batched (B,C,H,W)
        batch_size = state.shape[0] if is_batched else 1

        if train and random.random() < self.compute_epsilon(current_steps):
            # Handle random actions
            if hasattr(self.env, "single_action_space"):
                # Vectorized environment
                return np.array(
                    [self.env.single_action_space.sample() for _ in range(batch_size)]
                )
            else:
                # Single environment
                return self.env.action_space.sample()
        else:
            # Handle greedy actions
            with torch.no_grad():
                if not is_batched:
                    state = state.unsqueeze(0)
                q_values = self.policy_net(state)
                actions = q_values.max(1)[1].cpu().numpy()
                return actions if is_batched else actions[0]

    def update_weights(self):
        if self.memory.__len__() < self.batch_size:
            return None  # Explicitly return None for loss tracking

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Properly handle dimensions and device
        states = torch.stack([s.clone().detach() for s in batch.state]).to(self.device)
        actions = torch.tensor(
            batch.action, device=self.device, dtype=torch.long
        ).unsqueeze(1)  # Changed
        rewards = torch.tensor(
            batch.reward, device=self.device, dtype=torch.float32
        ).unsqueeze(1)  # Changed
        next_states = torch.stack([s.clone().detach() for s in batch.next_state]).to(
            self.device
        )
        dones = torch.tensor(
            batch.done, device=self.device, dtype=torch.float32
        ).unsqueeze(1)  # Changed

        # Calculate current Q values
        q_current = self.policy_net(states).gather(1, actions)

        # Calculate target Q values using Double DQN
        with torch.no_grad():
            # Get actions from policy net
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            # Get Q-values from target net for those actions
            q_next = self.target_net(next_states).gather(1, next_actions)
            # Calculate target values
            target_q = rewards + (1 - dones) * self.gamma * q_next

        # Compute loss and update
        loss = self.criterion(q_current, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Sync target network
        self.steps_done += 1
        if self.steps_done % self.sync_target == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save_checkpoint(self, path):
        """
        Save both policy and target networks to files in the Double DQN weights directory.
        """
        base_path = f"weights/ddqn/{path}"
        # Save policy network
        policy_path = f"{base_path}_policy_net.pth"
        torch.save(self.policy_net.state_dict(), policy_path)
        # Save target network
        target_path = f"{base_path}_target_net.pth"
        torch.save(self.target_net.state_dict(), target_path)
        print(f"Checkpoints saved to {policy_path} and {target_path}")

    def load_checkpoint(self, path):
        """
        Load both policy and target networks from files in the Double DQN weights directory.
        """
        base_path = f"weights/ddqn/{path}"
        # Load policy network
        policy_path = f"{base_path}_policy_net.pth"
        self.policy_net.load_state_dict(torch.load(policy_path, map_location=self.device))
        # Load target network
        target_path = f"{base_path}_target_net.pth"
        self.target_net.load_state_dict(torch.load(target_path, map_location=self.device))
        print(f"Checkpoints loaded from {policy_path} and {target_path}")
