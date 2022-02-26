import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
from einops import rearrange

class PolylineComponent(nn.Module):
    """
    Get the Encoder output of the MP transformer, and convert
    it to a prediction of n points, representing a polygonal line
    """
    def __init__(self, n_points):
        pass
