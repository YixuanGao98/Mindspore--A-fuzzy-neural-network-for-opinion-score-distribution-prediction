import mindspore
from mindspore import nn, ops

class MQuantileLoss(nn.Cell):
    def __init__(self):
        super(MQuantileLoss, self).__init__()
        self.cum=mindspore.ops.CumSum()
        self.abs=mindspore.ops.Abs()
        self.zore=mindspore.ops.Zeros()
    def construct(self, p_estimate, p_target):
        #Quantile 25%，50%,75%三个四分位数
        cdf_target =self.cum(p_target, 1)
        # cdf for values [1, 2, ..., 10]
        cdf_estimate =self.cum(p_estimate, 1)
        percentiles =[0.25, 0.5,0.75]
        # x= torch.arange(5, 105, 10)
        x= mindspore.numpy.arange(1, len(p_target[0])+1, 1)
        quan1=self.zore([len(p_target),len(percentiles)])
        quan2=self.zore([len(p_estimate),len(percentiles)])
        # for index,target in enumerate(cdf_target):#计算每个直方图的分位数
        for index in range(0, len(cdf_target)):
            target=cdf_target[index]
            print(cdf_target)
            print(cdf_target[index])
            for k in range(0, len(percentiles)):
                score1=0
                for i in range(0, len(target)):
                    if percentiles[k] <= target[0]:
                        Xa = 0
                        Xb = x[i]
                        Ya = 0
                        Yb = target[i]

                        A = (Yb-Ya)/(Xb-Xa)
                        B = Yb-A*Xb

                        score1 = (percentiles[k]-B)/A
                        break
                    elif percentiles[k] <= target[i] and percentiles[k] > target[i-1]:
                        Xa = x[i-1]
                        Xb = x[i]
                        Ya = target[i-1]
                        Yb = target[i]

                        A = (Yb-Ya)/(Xb-Xa)
                        B = Yb-A*Xb

                        score1 = (percentiles[k]-B)/A
                        break
                quan1[index][k]=score1

        for index,pre0 in enumerate(cdf_estimate):
            for k in range(0, len(percentiles)):
                score3=0
                for i in range(0, len(pre0)):
                    if i == 0 and percentiles[k] <= pre0[i]:
                        Xa = 0
                        Xb = x[i]
                        Ya = 0
                        Yb = pre0[i]

                        A = (Yb-Ya)/(Xb-Xa)
                        B = Yb-A*Xb 

                        score3 = (percentiles[k]-B)/A
                        break
                    elif percentiles[k]<= pre0[i] and percentiles[k] > pre0[i-1]:
                        Xa = x[i-1]
                        Xb = x[i]
                        Ya = pre0[i-1]
                        Yb = pre0[i]

                        A = (Yb-Ya)/(Xb-Xa)
                        B = Yb-A*Xb

                        score3 = (percentiles[k]-B)/A
                        break
                quan2[index][k]=score3
        # quan1=torch.tensor(quan1)
        # quan2=torch.tensor(quan2)
        loss=self.abs(quan1-quan2)
        # print(torch.mean(torch.mean(loss,1),0))#
        # print(loss.mean())#默认不设置dim的时候，返回的是所有元素的平均值。
        



        

        return loss.mean()