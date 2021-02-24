import numpy as np
import megengine.module as M
# from megengine.core import Buffer
from megengine.tensor import Tensor
from megengine import functional as F

class FrozenBatchNorm2d(M.Module):
    """
    BatchNorm2d, which the weight, bias, running_mean, running_var
    are immutable.
    """
    def __init__(self, num_features, eps=1e-5):

        super().__init__()
        self.eps = eps
        self.weight = Tensor(np.ones(num_features, dtype=np.float32))
        self.bias = Tensor(np.zeros(num_features, dtype=np.float32))
        self.running_mean = Tensor(np.zeros((1, num_features, 1, 1), dtype=np.float32))
        self.running_var = Tensor(np.ones((1, num_features, 1, 1), dtype=np.float32))

    def forward(self, x):
        
        # scale = self.weight.reshape(1, -1, 1, 1) * (1.0 / (self.running_var + self.eps).sqrt())
        mask = self.running_var >= 0
        scale = self.weight.reshape(1, -1, 1, 1) * (1.0 / F.sqrt(self.running_var * mask + self.eps))
        bias = self.bias.reshape(1, -1, 1, 1) - self.running_mean * scale
        return x * scale + bias

