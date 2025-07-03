import random
from collections import namedtuple

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "done", "next_state")
)

# Ejemplo uso
# nueva_tupla = Transition(state, action, reward, done, next_state)


class ReplayMemory:
    def __init__(self, capacity):
        """
        Inicializa la memoria de repetición con capacidad fija.
        Params:
         - capacity (int): número máximo de transiciones a almacenar.
        """
        # TODO: almacenar capacity, inicializar lista de memoria y puntero de posición
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def add(self, state, action, reward, done, next_state):
        """
        Agrega una transición a la memoria.
        Si la memoria está llena, sobreescribe la transición más antigua.
        """
        # TODO: crear Transition y agregar o reemplazar en la lista según capacity
        # TODO: actualizar puntero de posición circular
        transition = Transition(state, action, reward, done, next_state)
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Devuelve un batch aleatorio de transiciones.
        Params:
         - batch_size (int): número de transiciones a muestrear.
        Returns:
         - lista de Transition de longitud batch_size.
        """
        # TODO: verificar que batch_size <= len(self)
        if batch_size > len(self):
            raise ValueError("Batch size exceeds the number of transitions in memory.")
        # TODO: retornar una muestra aleatoria de self.memory
        return random.sample(self.memory, batch_size)
    
    def add_with_priority(self, state, action, reward, done, next_state, priority):
        """
        Agrega una transición a la memoria con prioridad asociada.
        Si la memoria está llena, sobreescribe la transición más antigua.
        """
        transition = Transition(state, action, reward, done, next_state)
        item = (transition, priority)
        if not hasattr(self, "priority_memory"):
            self.priority_memory = []
            self.priority_position = 0
            self.max_priority = priority
        self.max_priority = max(self.max_priority, priority)
        if len(self.priority_memory) < self.capacity:
            self.priority_memory.append(item)
        else:
            self.priority_memory[self.priority_position] = item
        self.priority_position = (self.priority_position + 1) % self.capacity

    def sample_with_priority(self, batch_size, alpha=0.6, beta=0.4):
        """
        Devuelve un batch aleatorio de transiciones usando muestreo por prioridad.
        Params:
         - batch_size (int): número de transiciones a muestrear.
         - alpha (float): exponente para priorización.
         - beta (float): exponente para pesos de importancia.
        Returns:
         - batch: lista de Transition
         - indices: índices de las transiciones seleccionadas
         - weights: pesos de importancia para cada transición
        """
        if not hasattr(self, "priority_memory") or len(self.priority_memory) == 0:
            raise ValueError("No hay transiciones con prioridad en la memoria.")
        priorities = [p for (_, p) in self.priority_memory]
        scaled_priorities = [p ** alpha for p in priorities]
        total = sum(scaled_priorities)
        probs = [sp / total for sp in scaled_priorities]
        indices = random.choices(range(len(self.priority_memory)), probs, k=batch_size)
        batch = [self.priority_memory[i][0] for i in indices]
        N = len(self.priority_memory)
        weights = [(N * probs[i]) ** (-beta) for i in indices]
        max_weight = max(weights)
        weights = [w / max_weight for w in weights]
        return batch, indices, weights
    
    

    def update_priorities(self, indices, priorities):
        """
        Actualiza las prioridades de las transiciones dadas por sus índices.
        """
        if not hasattr(self, "priority_memory"):
            raise ValueError("No hay memoria de prioridad inicializada.")
        for idx, priority in zip(indices, priorities):
            transition, _ = self.priority_memory[idx]
            self.priority_memory[idx] = (transition, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        """
        Devuelve el número actual de transiciones en memoria.
        """
        # TODO: retornar tamaño de la lista de memoria
        return len(self.memory)

    def clear(self):
        """
        Elimina todas las transiciones de la memoria.
        """
        # TODO: resetear lista de memoria y puntero de posición
        self.memory.clear()
        self.position = 0
