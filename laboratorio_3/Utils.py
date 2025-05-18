from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_value_distribution(value):
    # min_x = min(k[0] for k in value.keys())
    min_x = 12 # we are not interested in the first 11 values (we alway going to hit)
    max_x = max(k[0] for k in value.keys())
    min_y = min(k[1] for k in value.keys())
    max_y = max(k[1] for k in value.keys())
    
    player_sum = np.arange(min_x, max_x + 1)
    dealer_show = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(player_sum, dealer_show)

    usable_ace = np.array([0, 1])
    state_values = np.zeros((len(player_sum), len(dealer_show), len(usable_ace)))
    for i, player in enumerate(player_sum):
        for j, dealer in enumerate(dealer_show):
            for k, ace in enumerate(usable_ace):
                state_values[i, j, k] = value[player, dealer, ace]
    
    
    # -*-*-*-*-*-*-*-*-* Sin As usable -*-*-*-*-*-*-*-*-*
    
    # Gráfico 3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.title("Distribucion de valores / sin As usable")
    plt.xlabel("Suma jugador")
    plt.ylabel("Carta dealer")
    ax.view_init(ax.elev, -120)
    surf = ax.plot_surface(X, Y, state_values[:, :, 0].T, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    cbar = plt.colorbar(surf, ticks=np.arange(-1, 1.1, 0.1))
    cbar.set_label("Valor")
    plt.show()

    #Gráfico de contorno
    cp = plt.contourf(X, Y, state_values[:, :, 0].T, cmap=cm.coolwarm)
    plt.title("Contorno distribucion de valores / sin As usable")
    plt.xlabel("Suma jugador")
    plt.ylabel("Carta dealer")
    cbar = plt.colorbar(cp, ticks=np.arange(-1, 1.1, 0.1))
    cbar.set_label("Valor")
    plt.show()
    
    # -*-*-*-*-*-*-*-*-* Con As usable -*-*-*-*-*-*-*-*-* 
    
    # Gráfico 3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.title("Distribucion de valores / con As usable")
    plt.xlabel("Suma jugador")
    plt.ylabel("Carta dealer")
    ax.view_init(ax.elev, -120)
    surf = ax.plot_surface(X, Y, state_values[:, :, 1].T, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    cbar = plt.colorbar(surf, ticks=np.arange(-1, 1.1, 0.1))
    cbar.set_label("Valor")
    plt.show()
    
    #Gráfico de contorno
    cp = plt.contourf(X, Y, state_values[:, :, 1].T, cmap=cm.coolwarm)
    plt.title("Contorno distribucion de valores / con As usable")
    plt.xlabel("Suma jugador")
    plt.ylabel("Carta dealer")
    cbar = plt.colorbar(cp, ticks=np.arange(-1, 1.1, 0.1))
    cbar.set_label("Valor")
    plt.show()
    
    
def plot_Q_distribution(Q):
    # Extraemos todos los estados y acciones presentes en Q
    # Se asume que las claves tienen la forma ((player, dealer, ace), action)
    states = [state for (state, action) in Q.keys()]
    actions = sorted(list(set(action for (state, action) in Q.keys())))
    
    # Definimos el rango para "suma jugador" y "carta dealer"
    player_sum_vals = [s[0] for s in states]
    dealer_vals = [s[1] for s in states]
    min_player = 12  # ignoramos sumas menores a 12
    max_player = max(player_sum_vals)
    min_dealer = min(dealer_vals)
    max_dealer = max(dealer_vals)
    
    player_range = np.arange(min_player, max_player + 1)
    dealer_range = np.arange(min_dealer, max_dealer + 1)
    X, Y = np.meshgrid(player_range, dealer_range)
    
    # Consideramos ambos valores para as usable: 0 y 1.
    usable_ace_vals = [0, 1]
    
    # Para cada acción, generamos los gráficos
    for action in actions:
        # Creamos un arreglo para almacenar Q para cada (player, dealer, ace)
        # La forma del arreglo será: (# player sums, # dealer cards, 2) 
        state_values = np.zeros((len(player_range), len(dealer_range), len(usable_ace_vals)))
        
        action_str = "(0) STICK" if action == 0 else "(1) HIT"
        
        for i, player in enumerate(player_range):
            for j, dealer in enumerate(dealer_range):
                for k, ace in enumerate(usable_ace_vals):
                    key = ((player, dealer, ace), action)
                    # Si el par (state, action) no existe en Q, asignamos NaN
                    if key in Q:
                        state_values[i, j, k] = Q[key]
                    else:
                        state_values[i, j, k] = np.nan
        
        # --- Gráficos para estados SIN as usable (ace = 0) ---
        # Gráfico 3D de superficie
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        plt.title(f"Distribución de Q (sin As usable) / acción {action_str}")
        plt.xlabel("Suma jugador")
        plt.ylabel("Carta dealer")
        ax.view_init(elev=ax.elev, azim=-120)
        surf = ax.plot_surface(X, Y, state_values[:, :, 0].T, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        cbar = plt.colorbar(surf)
        cbar.set_label("Valor Q")
        plt.show()
        
        # Gráfico de contorno
        cp = plt.contourf(X, Y, state_values[:, :, 0].T, cmap=cm.coolwarm)
        plt.title(f"Contorno Q (sin As usable) / acción {action_str}")
        plt.xlabel("Suma jugador")
        plt.ylabel("Carta dealer")
        cbar = plt.colorbar(cp)
        cbar.set_label("Valor Q")
        plt.show()
        
        # --- Gráficos para estados CON as usable (ace = 1) ---
        # Gráfico 3D de superficie
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        plt.title(f"Distribución de Q (con As usable) / acción {action_str}")
        plt.xlabel("Suma jugador")
        plt.ylabel("Carta dealer")
        ax.view_init(elev=ax.elev, azim=-120)
        surf = ax.plot_surface(X, Y, state_values[:, :, 1].T, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        cbar = plt.colorbar(surf)
        cbar.set_label("Valor Q")
        plt.show()
        
        # Gráfico de contorno
        cp = plt.contourf(X, Y, state_values[:, :, 1].T, cmap=cm.coolwarm)
        plt.title(f"Contorno Q (con As usable) / acción {action_str}")
        plt.xlabel("Suma jugador")
        plt.ylabel("Carta dealer")
        cbar = plt.colorbar(cp)
        cbar.set_label("Valor Q")
        plt.show()