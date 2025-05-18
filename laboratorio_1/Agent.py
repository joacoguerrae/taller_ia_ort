from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):
    """
    Clase base abstracta para un agente en un entorno Gymnasium.
    Define la estructura básica de cualquier agente, incluyendo la lógica de
    selección de acciones, actualización de estado interno y registro de datos.
    """

    @abstractmethod
    def __init__(self, name):
        """
        Constructor del agente.
        Debe ser implementado por cualquier subclase.

        Args:
            name (str): Nombre del agente.
        """
        self.name = name  # Nombre del agente

    def play(self, n_steps, environment):
        """
        Ejecuta una sesión en el entorno por un número determinado de pasos.

        Args:
            n_steps (int): Número máximo de pasos a ejecutar.
            environment: Entorno en el que el agente interactúa.

        Returns:
            log (dict): Registro de las acciones y recompensas obtenidas.
            info (dict): Información extra obtenida al final del episodio.
        """
        self.environment = environment  # Guardamos referencia al entorno
        self.clear_log()  # Limpiar registros previos
        self.reset_internal_state()  # Resetear estado interno del agente

        # Reiniciar el entorno y obtener la primera observación
        obs, info = self.environment.reset()

        for _ in range(n_steps):
            environment.render()  # Renderiza el entorno (opcional)

            # Selecciona una acción basado en la observación actual
            action = self.select_action(obs, info)

            # Ejecutar la acción en el entorno y obtener la nueva observación
            observation, reward, terminated, truncated, info = environment.step(action)

            # Registrar la acción y la recompensa obtenida
            self.log(action, reward)

            # Actualizar el estado interno del agente
            self.update_internal_state(observation, action, reward, info)

            # Terminar si el episodio ha concluido
            if terminated or truncated:
                break

        # Obtener información adicional del agente y retornar logs
        info = self.get_extra_info()
        log = self.get_log()
        return log, info

    def clear_log(self):
        """
        Inicializa o limpia los registros de acciones y recompensas.
        """
        self.rewards = []  # Lista de recompensas obtenidas
        self.selected_actions = []  # Lista de acciones seleccionadas
        self.actions_rewards = {  # Diccionario con recompensas por acción
            action: [] for action in range(self.environment.action_space.n)
        }

    def log(self, action, reward):
        """
        Registra una acción tomada y su recompensa correspondiente.

        Args:
            action (int): Acción ejecutada.
            reward (float): Recompensa obtenida.
        """
        self.rewards.append(reward)
        self.selected_actions.append(action)
        self.actions_rewards[action].append(reward)

    def get_log(self):
        """
        Retorna el registro de acciones y recompensas obtenidas durante la ejecución.

        Returns:
            dict: Contiene listas de recompensas, acciones seleccionadas y recompensas por acción.
        """
        return {
            "rewards": np.array(self.rewards),
            "selected_actions": np.array(self.selected_actions),
            "actions_rewards": self.actions_rewards,
        }

    @abstractmethod
    def reset_internal_state(self):
        """
        Método abstracto para reiniciar el estado interno del agente.
        Debe ser implementado en cada subclase.
        """
        pass

    @abstractmethod
    def select_action(self, obs, info):
        """
        Método abstracto para seleccionar una acción basada en la observación actual y estado interno.
        Debe ser implementado en cada subclase.

        Args:
            obs: Observación actual del entorno.
            info: Información adicional proporcionada por el entorno.

        Returns:
            int: Acción seleccionada.
        """
        pass

    @abstractmethod
    def update_internal_state(self, observation, action, reward, info):
        """
        Método abstracto para actualizar el estado interno del agente tras cada paso.

        Args:
            observation: Nueva observación obtenida tras ejecutar la acción.
            action (int): Acción tomada por el agente.
            reward (float): Recompensa obtenida.
            info: Información adicional del entorno.
        """
        pass

    @abstractmethod
    def get_extra_info(self):
        """
        Método abstracto para obtener información extra al finalizar un episodio.
        Debe ser implementado en cada subclase.

        Returns:
            dict: Información adicional sobre el comportamiento del agente.
        """
        pass
