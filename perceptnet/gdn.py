from mindspore import nn, ops
from mindspore.common.initializer import Normal,Constant
import torch.nn.functional as F
import mindspore
import mindspore.ops as ops
class GDN(nn.Cell):
    def __init__(self,
                 n_channels,
                #  gamma_init=.1,
                 gamma_init=.1,
                #  reparam_offset=2**-18,
                 reparam_offset=2**-18,
                 beta_min=1e-6,
                 apply_independently=False):
        super(GDN, self).__init__()
        self.n_channels = n_channels
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset
        self.beta_min = beta_min
        self.beta_reparam = (self.beta_min + self.reparam_offset**2)**0.5
        self.apply_independently = apply_independently
        if apply_independently:
            self.groups = n_channels
        else:
            self.groups = 1
        self.sqr=ops.Sqrt()
        self.mul=ops.Mul()
        self.ExpandDims=ops.ExpandDims()
        # self.ClipByValue1=ops.ClipByValue( min_val=self.reparam_offset)
        # self.ClipByValue2=ops.ClipByValue( min_val=self.beta_reparam)
        self.ones = ops.Ones()
        self.initialise_params()
        

    def initialise_params(self):
        gamma_bound = self.reparam_offset
        gamma = ops.eye(self.n_channels, self.n_channels)
        gamma = gamma.view(self.n_channels, self.n_channels, 1, 1)
        gamma = self.sqr(self.gamma_init*gamma + self.reparam_offset**2)
        gamma = self.mul(gamma, gamma)
        if self.apply_independently:
            gamma = self.ExpandDims(gamma[:, 0, :, :],1)

        self.gamma = mindspore.Parameter(gamma)
        beta = self.ones((self.n_channels,))
        beta = self.sqr(beta + self.reparam_offset**2)
        self.beta = mindspore.Parameter(beta)

    def forward(self, x):
        """Forward pass of the layer
        Input must be shape: [batch_size, channels, height, width]
        """
        self.inputs = x
        self.gamma.data = ops.clip_by_value(self.gamma.data,min_val=self.reparam_offset)
        self.beta.data =ops.clip_by_value(self.beta.data,min_val=self.beta_reparam)
        norm_pool = F.conv2d(self.mul(x, x), self.gamma, bias=self.beta,
                             groups=self.groups)
        norm_pool = self.sqr(norm_pool)
        output = x / norm_pool
        return output

