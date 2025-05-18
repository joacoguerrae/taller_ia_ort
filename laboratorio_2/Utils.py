def print_state_values(V, size):
    """
    Imprime el valor de cada estado en una grilla.

    Parámetros:
        V: diccionario que mapea cada estado (tupla (x, y)) a su valor.
        size: tamaño de la grilla (número de filas y columnas).
    """
    for y in range(size):
        row_values = []
        for x in range(size):
            value = V.get((x, y), 0)
            row_values.append(f"{value:8.2f}")  # Se formatea el número con 2 decimales
        print(" ".join(row_values))
        
def print_state_action_values(Q, size):
    """
    Imprime el valor de cada estado y acción en una grilla.

    Parámetros:
        Q: diccionario que mapea cada estado (tupla (x, y)) y acción a su valor.
        size: tamaño de la grilla (número de filas y columnas).
    """
    action_names = {0: "Derecha", 1: "Arriba", 2: "Izquierda", 3: "Abajo"}
    for a in range(4):
        print(f"Valores para la acción {action_names[a]} (acción {a}):")
        for y in range(size):
            row_values = []
            for x in range(size):
                value = Q.get(((x, y), a), 0)
                row_values.append(f"{value:8.2f}")
            print(" ".join(row_values))
        print("\n")        
        
def print_policy_grids(pi, size):
    """
    Imprime 4 grillas, una para cada acción, mostrando la probabilidad
    de tomar esa acción en cada estado.

    Parámetros:
        pi: diccionario que mapea (estado, acción) a la probabilidad de tomarla.
        size: tamaño de la grilla (número de filas y columnas).
    """
    action_names = {0: "Derecha", 1: "Arriba", 2: "Izquierda", 3: "Abajo"}
    for a in range(4):
        print(f"Política para la acción {action_names[a]} (acción {a}):")
        for y in range(size):
            row_values = []
            for x in range(size):
                prob = pi.get(((x, y), a), 0)
                row_values.append(f"{prob:6.2f}")
            print(" ".join(row_values))
        print("\n")
