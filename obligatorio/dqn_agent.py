import torch
import torch.nn as nn
import numpy as np
from abstract_agent import Agent
from replay_memory import ReplayMemory, Transition


class DQNAgent(Agent):
    def __init__(
        self,
        env,
        model,
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
        super().__init__(
            env,
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
            checkpoint_interval,  # Pass checkpoint interval to superclass
        )
        # Guardar entorno y función de preprocesamiento
        # Inicializar policy_net en device
        # Configurar función de pérdida MSE y optimizador Adam
        # Crear replay memory de tamaño buffer_size
        # Almacenar batch_size, gamma y parámetros de epsilon-greedy
        self.env = env
        self.state_processing_function = obs_processing_func

        self.policy_net = model.to(device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )

        self.memory = ReplayMemory(memory_buffer_size)

        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal_steps = epsilon_anneal_steps
        self.episode_block = episode_block
        self.checkpoint_interval = checkpoint_interval

    def select_action(self, state, current_steps, train=True):
        # Calcular epsilon según step
        # Durante entrenamiento: con probabilidad epsilon acción aleatoria
        #                   sino greedy_action
        # Durante evaluación: usar greedy_action (o pequeña epsilon fija)
        self.epsilon = self.compute_epsilon(current_steps)
        if train and np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = state.clone().detach().unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax().item()
        return action

    def update_weights(self):
        # 1) Comprobar que hay al menos batch_size muestras en memoria
        # 2) Muestrear minibatch y convertir a tensores (states, actions, rewards, dones, next_states)
        # 3) Calcular q_current con policy_net(states).gather(...)
        # 4) Con torch.no_grad(): calcular max_q_next_state = policy_net(next_states).max(dim=1)[0] * (1 - dones)
        # 5) Calcular target = rewards + gamma * max_q_next_state
        # 6) Computar loss MSE entre q_current y target, backprop y optimizer.step()
        if self.memory.__len__() < self.batch_size:
            return  # No hay suficientes muestras en memoria
        else:
            tranitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*tranitions))
            state_batch = torch.stack([s.clone().detach() for s in batch.state]).to(
                self.device
            )
            action_batch = torch.tensor(batch.action, device=self.device)
            reward_batch = torch.tensor(batch.reward, device=self.device)
            done_batch = torch.tensor(
                batch.done, dtype=torch.float32, device=self.device
            )
            next_state_batch = torch.stack(
                [s.clone().detach() for s in batch.next_state]
            ).to(self.device)

            q_current = (
                self.policy_net(state_batch)
                .gather(1, action_batch.unsqueeze(1))
                .squeeze(1)
            )

            with torch.no_grad():
                max_q_next_state = self.policy_net(next_state_batch).max(dim=1)[0] * (
                    1 - done_batch
                )
                target = reward_batch + self.gamma * max_q_next_state

            loss = self.criterion(q_current, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()
