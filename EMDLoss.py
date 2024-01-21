# import torch
# import torch.nn as nn
# from torch.autograd import Variable
import numpy
import mindspore
from mindspore import nn, ops

class EMDLoss(nn.Cell):
    def __init__(self):
        super(EMDLoss, self).__init__()
        self.cum=mindspore.ops.CumSum()
        self.sqr=ops.Sqrt()
        self.mean=mindspore.ops.ReduceMean()
        self.pow=mindspore.ops.Pow()
        self.abs=mindspore.ops.Abs()
    def construct(self, p_estimate, p_target):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]
        cdf_target = self.cum(p_target, 1)
        # cdf for values [1, 2, ..., 10]
        cdf_estimate = self.cum(p_estimate, 1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = self.sqr(self.mean(self.pow(self.abs(cdf_diff), 2)))


        return samplewise_emd.mean()
    
    

