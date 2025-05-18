import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import tqdm

import KBanditsEnv
from Agent import Agent

def get_env(print_true_values=False):
    return gym.make("k_bandits_env/KBanditsGaussian-v0", print_true_values=print_true_values, render_mode=None)    

def evaluate_agents(agents, num_steps, num_episodes):
    """
    Evalúa una lista de agentes en un entorno específico durante un número determinado de episodios.

    Args:
        agents (list): Lista de instancias de agentes a evaluar.
        num_steps (int): Número de pasos por episodio.
        num_episodes (int): Número de episodios para cada agente.

    Returns:
        dict: Diccionario con los nombres de los agentes como claves y sus registros de desempeño como valores.
    """
    performance_logs = {}

    for agent in agents:
        all_rewards = []
        optimal_action_counts = np.zeros(num_steps)

        for _ in tqdm.tqdm(range(num_episodes)):
            env = get_env()
            optimal_action = env.unwrapped.true_means.argmax()
            logs, _ = agent.play(n_steps=num_steps, environment=env)
            all_rewards.append(logs['rewards'])

            actions = logs['selected_actions']
            for step, action in enumerate(actions):
                if action == optimal_action:
                    optimal_action_counts[step] += 1

        rewards_array = np.array(all_rewards)
        average_rewards = np.mean(rewards_array, axis=0)
        optimal_action_percentage = (optimal_action_counts / num_episodes) * 100

        performance_logs[agent.name] = {
            'average_rewards': average_rewards,
            'optimal_action_percentage': optimal_action_percentage
        }

    return performance_logs

def plot_average_rewards(performance_logs):
    """
    Grafica el promedio de recompensa por paso para cada agente.

    Args:
        performance_logs (dict): Diccionario con los registros de desempeño de los agentes.
    """
    plt.figure(figsize=(10, 5))

    for agent_name, logs in performance_logs.items():
        plt.plot(logs['average_rewards'], label=f'{agent_name}')

    plt.xlabel('Paso')
    plt.ylabel('Recompensa Promedio')
    plt.title('Promedio de Recompensa por Paso para Cada Agente')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_optimal_action_percentage(performance_logs):
    """
    Grafica el porcentaje de selección de la acción óptima por paso para cada agente.

    Args:
        performance_logs (dict): Diccionario con los registros de desempeño de los agentes.
    """
    plt.figure(figsize=(10, 5))

    for agent_name, logs in performance_logs.items():
        plt.plot(logs['optimal_action_percentage'], label=f'{agent_name}')

    plt.xlabel('Paso')
    plt.ylabel('Porcentaje de Selección de Acción Óptima (%)')
    plt.ylim(0, 100)
    plt.title('Porcentaje de Selección de la Acción Óptima por Paso para Cada Agente')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_agents_performance(agents, num_steps, num_episodes):
    """
    Evalúa y grafica el desempeño de una lista de agentes en un entorno específico.
    """
    performance_logs = evaluate_agents(agents, num_steps, num_episodes)
    plot_average_rewards(performance_logs)
    plot_optimal_action_percentage(performance_logs)
