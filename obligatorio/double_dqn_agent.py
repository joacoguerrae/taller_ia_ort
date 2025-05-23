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
        self.sync_target = sync_target
        self.steps_done = 0
        # Inicializar epsilon

    def select_action(self, state, current_steps, train=True):
        # Calcular epsilon decay según step (entre eps_start y eps_end en eps_steps)
        self.epsilon = self.compute_epsilon(current_steps)
        # Si train y con probabilidad epsilon: acción aleatoria
        if train and random.random() < self.epsilon:
            # Escoger acción aleatoria
            action = self.env.action_space.sample()
            return action
        # En otro caso: usar greedy_action
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, device=self.device).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax().item()
            return action

    def update_weights(self):
        if self.memory.__len__() < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Move tensors to CPU before converting to numpy
        states = torch.stack([s.clone().detach() for s in batch.state]).to(self.device)
        actions = torch.tensor(batch.action, device=self.device)
        rewards = torch.tensor(batch.reward, device=self.device)
        next_states = torch.stack([s.clone().detach() for s in batch.next_state]).to(
            self.device
        )
        dones = torch.tensor(batch.done, device=self.device, dtype=torch.float32)

        # 3) Calcular q_current: online_net(states).gather(…)
        q_current = (self.policy_net(states).gather(1, actions.unsqueeze(1))).squeeze(1)
        # 4) Calcular target Double DQN:
        #    a) best_actions = online_net(next_states).argmax(…)
        #    b) q_next = target_net(next_states).gather(… best_actions)
        #    c) target_q = rewards + gamma * q_next * (1 - dones)
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            q_next = self.target_net(next_states).gather(1, best_actions)
            target_q = rewards + self.gamma * q_next * (1 - dones)
        # 5) Computar loss MSE entre q_current y target_q, backprop y optimizer.step()
        loss = self.criterion(q_current, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # 6) Decrementar contador y si llega a 0 copiar online_net → target_net
        self.steps_done += 1
        if self.steps_done % self.sync_target == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()
