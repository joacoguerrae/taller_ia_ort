import torch.nn as nn
import torch.nn.functional as F
import torch


def conv2d_output_shape(
    input_size: tuple[int, int],
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> tuple[int, int]:
    """
    Calcula (H_out, W_out) para una capa Conv2d con:
      - input_size: (H_in, W_in)
      - kernel_size, stride, padding, dilation: int o tupla (altura, ancho)
    Basado en:
      H_out = floor((H_in + 2*pad_h - dil_h*(ker_h−1) - 1) / str_h + 1)
      W_out = floor((W_in + 2*pad_w - dil_w*(ker_w−1) - 1) / str_w + 1)
    Fuente: Shape section en torch.nn.Conv2d :contentReference[oaicite:0]{index=0}
    """

    # Unifica todos los parámetros a tuplas (h, w)
    def to_tuple(x):
        return (x, x) if isinstance(x, int) else x

    H_in, W_in = input_size
    ker_h, ker_w = to_tuple(kernel_size)
    str_h, str_w = to_tuple(stride)
    pad_h, pad_w = to_tuple(padding)
    dil_h, dil_w = to_tuple(dilation)

    H_out = (H_in + 2 * pad_h - dil_h * (ker_h - 1) - 1) // str_h + 1
    W_out = (W_in + 2 * pad_w - dil_w * (ker_w - 1) - 1) // str_w + 1

    return H_out, W_out


class DQN_CNN_Model(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super(DQN_CNN_Model, self).__init__()
        # TODO: definir capas convolucionales basadas en obs_shape
        # TODO: definir capas lineales basadas en n_actions
        self.conv1 = nn.Conv2d(obs_shape[0], 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, obs):
        # TODO: 1) aplicar convoluciones y activaciones
        #       2) aplanar la salida
        #       3) aplicar capas lineales
        #       4) devolver tensor de Q-values de tamaño (batch, n_actions)
        self.conv1_out = F.relu(self.conv1(obs))
        self.conv2_out = F.relu(self.conv2(self.conv1_out))
        self.flatten = self.conv2_out.view(self.conv2_out.size(0), -1)
        self.fc1_out = F.relu(self.fc1(self.flatten))
        self.fc2_out = self.fc2(self.fc1_out)
        return self.fc2_out

    def load_weights(self, path, device):
        """
        Carga los pesos del modelo desde un archivo.
        """
        self.load_state_dict(torch.load(path, map_location=device))
