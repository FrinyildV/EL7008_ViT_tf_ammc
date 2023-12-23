# -*- coding: utf-8 -*-
'''
Funciones para red ViT:
'''


#%%%%%%%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
import cv2

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

np.random.seed(0)
torch.manual_seed(0)



from torch import Tensor
from einops import rearrange , reduce, repeat
from einops.layers.torch import Rearrange, Reduce



#%%%%%%%%%%%%%%%

#%%%%%%% Bloques de la red:

def patchify(images, n_patches):
    '''
    Clase que hace patches las imagenes de entrada,
    '''
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches # retorna [batch_size , n_patches x n_patches , c x patch_size x patch_size]
class MultiHeadAttention(nn.Module):
    '''
    Clase que realiza la proyección linear multihead de los patches con las
    3 interpretaciones, (v) Value, (k) Key y (q) Query
    Separa en multiples cabezas (multihead) la data entrante (el patch) para cada
    una de estas proyecciones

    '''
    def __init__(self, hidden_d: int = 768, n_heads: int = 8, dropout: float = 0):
        super().__init__()

        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.keys = nn.Linear(hidden_d, hidden_d)
        self.queries = nn.Linear(hidden_d, hidden_d)
        self.values = nn.Linear(hidden_d, hidden_d)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_d, hidden_d)
        self.scaling = (self.hidden_d // n_heads) ** -0.5

    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in n_heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.n_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.n_heads)
        values  = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.n_heads)
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, n_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        att = torch.nn.functional.softmax(energy, dim=-1) * self.scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    '''
    Bloque que realiza la suma residual.
    '''
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class MLP_Block(nn.Sequential):
    '''
    The MLP contains two layers with a GELU non-linearity
    '''
    def __init__(self, hidden_d: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(hidden_d, expansion * hidden_d),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * hidden_d, hidden_d),
        )
class TransformerEncoderBlock(nn.Sequential):
    '''
    Bloque que realiza las operaciones del transformer encoder.
    Recibe los patches de tamaño [batch_size , n_patches x n_patches , c x patch_size x patch_size]
    y retorna [batch_size,n_patch+1,hidden_d]
    1ero: normaliza
    2do: alplica MultiHeadAttention
    3ro Realiza la suma residual
    4to: normaliza
    5to: Aplica capa MLP
    '''
    def __init__(self,
                 hidden_d: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(hidden_d),
                MultiHeadAttention(hidden_d, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(hidden_d),
                MLP_Block(
                    hidden_d, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
        

#%%%%%%%%%%%%%%%%

#%%%%%%%%%% RED:
class ViT(nn.Module):
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        # Super constructor
        super(ViT, self).__init__()

        # Attributes
        self.chw = chw # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Input and patches sizes
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        #alternativa: conv en 2d, pide mas v-ram
        # self.linear_mapper = nn.Conv2d(chw[0] , self.hidden_d , kernel_size = self.input_d ,stride = self.input_d )

        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional embedding
        self.positions = nn.Parameter(torch.randn(n_patches ** 2 + 1,self.hidden_d))


        self.blocks = nn.ModuleList([TransformerEncoderBlock(hidden_d=self.hidden_d,
                                              drop_p=0.5,
                                              forward_drop_p=0.5,
                                              n_heads=self.n_heads) \
                                            for _ in range(n_blocks)])
        # 5) Classification MLP
        self.mlp = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(self.hidden_d),
            nn.Linear(self.hidden_d, out_d)
            )

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)
        patches = patchify(images, self.n_patches).to(device) #.to(self.positional_embeddings.device)

        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Adding positional embedding
        # out = tokens + self.positional_embeddings.repeat(n, 1, 1)
        out = tokens + self.positions

        # Transformer Blocks
        for block in self.blocks:
          out = block(out)

        return self.mlp(out) # Map to output dimension, output category distribution