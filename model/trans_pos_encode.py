import os
from pathlib import Path
import random
import math
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from NAPE import Position_encode
# from training_utils_py2 import toNumpy


class TT_Pos_Encode(nn.Module):
    """
        Uses the learned position encoding as the ordering for this
        "transformerlike" positional encoding.
    """

    def __init__(self, hidden_size, N, d, PE_name=''):
        super(TT_Pos_Encode, self).__init__()

        self.PE_name = PE_name
        self.N = N
        self.d = d
        self.hidden_size = hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def get_ordering(self, N, d):
        """This would return the saved position encoding"""
        PE = Position_encode(N=N, d=d)
        if self.N == 25: #NTU RGB+D Dataset
            last_str = self.PE_name
        else: #Kinetics 400 Dataset
            last_str = 'checkpoint_pos_encode_kinetics.pt'
        
        PE.load_state_dict(torch.load(f'{Path.home()}/codes/NAPE-wt-node2vec/Saved_models/'+f'd={d}_{last_str}'))
        PE.eval()
        Z,_,_,_ = PE(test=True)
        err = 0.0001
        Z = Z + err
        Z = torch.round(Z)

        # We convert binary now to decimal
        return self.convert_bin_to_dec(Z)


    def convert_bin_to_dec(self, Z):
        """Function returns the decimal equivalent of the binary input"""
        d = self.d
        two_vec = torch.zeros(d)
        for i in range(d):
            two_vec[i] = pow(2,d-1-i)
        numbers = (Z * two_vec).sum(dim=1)

        return numbers


    def inv_timescales_fn(self, hidden_size):
        """Time scale for positional encoding"""

        num_timescales = hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0

        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            max(num_timescales - 1, 1))

        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)

        return inv_timescales


    def get_position_encoding(self, need_plot=False):
        """The position encoding is computed and returned"""

        # ordering = torch.arange(self.N) #Having 1-25 as ordering
        # ordering = torch.randperm(self.N)
        ordering = self.get_ordering(self.N, self.d)
        # maxi = torch.max(ordering).item()
        # ordering = torch.arange(int(maxi) - self.N, int(maxi))
        # ordering = torch.tensor(random.sample(range(int(maxi)),k=self.N))
        inv_timescales = self.inv_timescales_fn(self.hidden_size)

        max_length = self.N #number of nodes
        position = ordering
        if self.device is not None:
            position.to(self.device)

        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)
        signal = F.pad(signal, (0, 0, 0, self.hidden_size % 2))
        signal = signal.view(1, max_length, self.hidden_size)
        signal = signal.squeeze(0) #shape:([N, hidden_size])
        if need_plot:
            self.get_plot(signal, self.hidden_size)

        return signal #shape:([N, hidden_size])


    def get_plot(self, Z, d):
        """Returns a coloured-scale of the position encoding"""
        plt.figure(figsize=(40,24))
        plt.imshow( toNumpy(Z.t(),self.device) ) #, cmap=cm.Greys_r)
        plt.xticks(range(Z.shape[0]), list(range(1,Z.shape[0]+1)), fontsize=54)
        plt.tick_params(left = False , labelleft = False)
        plt.title(f'Sinusoidal Position Encoding, d={d}', fontdict={'fontsize': 80})
        # plt.title(f'Transformer Type Position Encoding, d={d}')
        plt.savefig(os.getcwd()+f'/Pos_Enc_image/TT_pos_encod_dim={d}.png',
            bbox_inches='tight')
        print('plot saved...')
