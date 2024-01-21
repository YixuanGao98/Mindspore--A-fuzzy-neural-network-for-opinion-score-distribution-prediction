
from download import download

import mindspore as ms
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import nn, ops
import scipy.io
from PIL import Image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
data_dir = "/home/gyx/DATA/imagehist/LIVE" # 数据集根目录
batch_size = 8 # 批量大小
image_size = 448# 训练图像空间大小
workers = 4 # 并行线程个数
num_classes = 10# 分类数量
ms.set_context(device_target="GPU", variable_memory_max_size='18GB')
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    # print f_list
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename
class MyAccessible:
    def __init__(self):
        self.refpath = os.path.join(data_dir, 'refimgs')
        self.refname = getFileName( self.refpath,'.bmp')

        # self.refnames_all = []
        self.imgname=[]
        self.labels = []
        self.csv_file = os.path.join(data_dir, 'LIVEhist_new.txt')
        with open(self.csv_file) as f:
            reader = f.readlines()
            for i, line in enumerate(reader):
                token = line.split("\t")
                token[0]=eval(token[0]) 
                self.imgname.append(token[0])
                values = np.array(token[1:11], dtype='float32')
                values /= values.sum()
                self.labels.append(values)

        refnames_all = scipy.io.loadmat(os.path.join(data_dir, 'refnames_all.mat'))
        self.refnames_all = refnames_all['refnames_all']

        self.dmos = scipy.io.loadmat(os.path.join(data_dir, 'dmos_realigned.mat'))
        self.orgs = self.dmos['orgs']

        sample = []
        image=[]
        for i in range(0, len(self.imgname)):

            sample.append(pil_loader(os.path.join(data_dir,'allimg', self.imgname[i])))
        
        self._data = sample
        self._label = self.labels

    def __getitem__(self, index):
        return self._data[index], self._label[index]

    def __len__(self):
        return len(self._data)
    
    
    
def create_dataset_cifar10(dataset_dir, usage, resize, batch_size, workers):

    data_set = ds.GeneratorDataset(source=MyAccessible(), column_names=["image", "label"])

    trans = []#需要做的变化的集合

    if usage == "train":
        trans += [
            vision.Resize((448,448)),
            # vision.RandomCrop((448,448)),
            vision.RandomHorizontalFlip()
        ]

    """
    再对数据集进行一些操作
    """
    trans += [
        vision.Resize((448,448)),
        # vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225)),
        vision.HWC2CHW()
    ]

    #对于label进行的操作
    target_trans = [(lambda x: np.array([x]).astype(np.int32)[0])]

    # 数据映射操作
    data_set = data_set.map(
        operations=trans,
        input_columns='image',
        num_parallel_workers=workers)

    data_set = data_set.map(
        operations=target_trans,
        input_columns='label',
        num_parallel_workers=workers)

    # 批量操作
    data_set = data_set.batch(batch_size)


    return data_set


# 利用上面写好的那个函数，获取处理后的训练与测试数据集
dataset_train = create_dataset_cifar10(dataset_dir=data_dir,
                                       usage="train",
                                       resize=image_size,
                                       batch_size=batch_size,
                                       workers=workers)
step_size_train = dataset_train.get_dataset_size()
index_label_dict = dataset_train.get_class_indexing()

dataset_val = create_dataset_cifar10(dataset_dir=data_dir,
                                     usage="test",
                                     resize=image_size,
                                     batch_size=1,
                                     workers=workers)
step_size_val = dataset_val.get_dataset_size()



"""
构建VGG16网络
"""
from EMDLoss import EMDLoss
from SoftHistogram import SoftHistogram
from MQuantileLoss_fixedbins import MQuantileLoss
from utils import score_utils
class VGG16(nn.Cell):
    def __init__(self):
        super().__init__()
        numClasses = 10
        self.all_sequential = nn.SequentialCell(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # # 原始模型vgg16输入image大小是224*224，这里使用的数据集输入大小为32*32，缩小7倍
            # # 可以根据需要的大小来调整，比如如果输入的image大小是224*224，那么由于224/32=7，因此就把第一个nn.Dense的参数改成512*7*7，其他不变
            # nn.Flatten(),
            # nn.Dense(512*1*1, 256),
            # nn.ReLU(),
            # nn.Dropout(),
            # nn.Dense(256, 256),
            # nn.ReLU(),
            # nn.Dropout(),
            # nn.Dense(256, numClasses),
        )

    def construct(self, x):
        x = self.all_sequential(x)
        return x


from mindspore import load_checkpoint, load_param_into_net
from perceptnet.gdn import GDN
def _vgg16(pretrained: bool = True):
    model = VGG16()
    "VGG16模型"
    #预训练模型的下载网址
    model_url = "https://download.mindspore.cn/model_zoo/official/cv/vgg/vgg16_ascend_0.5.0_cifar10_official_classification_20200715/vgg16.ckpt"
    #存储路径
    model_ckpt = "./LoadPretrainedModel/vgg16_0715.ckpt"

    if pretrained:
        download(url=model_url, path=model_ckpt)
        param_dict = load_checkpoint(model_ckpt)
        load_param_into_net(model, param_dict)

    return model


class SpatialAttention(nn.Cell):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size,pad_mode='pad', padding=padding,has_bias=False)
        # self.gdn1 = GDN(1, apply_independently=True)
        self.sigmoid = nn.Sigmoid()
        
        self.reduce_mean_op = ops.ReduceMean(keep_dims=True)
        self.reduce_max_op = ops.ReduceMax(keep_dims=True)
        self.concat_op = ops.Concat(axis=1)  
    def construct(self, x):
        avg_out = self.reduce_mean_op(x, axis=1)
        max_out= self.reduce_max_op(x,axis=1)
        x = self.concat_op([avg_out, max_out])
        x = (self.conv1(x))
        return self.sigmoid(x)

from SCNN import SCNN

from SoftHistogram import SoftHistogram
import mindspore
from mindspore.train.serialization import load_checkpoint, load_param_into_net

# context.set_context(device_target="GPU")
class DBCNN(nn.Cell):

    def __init__(self):
        """Declare all needed layers."""
        nn.Cell.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.base = _vgg16(pretrained=True)
        for param in self.base.get_parameters():
            param.requires_grad = True

        self.max=nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # self.features1_3=nn.CellList(*list(self.base.children())[0:31])
        self.features1_3=self.base
        self.up=nn.Upsample(scale_factor=4,mode='nearest')
        
        scnn = SCNN()
        self.features2 = scnn
        self.projection = nn.SequentialCell(nn.Conv2d(128,512,kernel_size=1,stride=1,padding=0), nn.BatchNorm2d(512), nn.ReLU(),
                                        )
        
        self.in_channels = 512
        self.num_bins =5
        
        self.SAfeatures=SpatialAttention()


        
        self.histogram=SoftHistogram(n_features=self.in_channels*14*14,n_examples=1,num_bins=self.num_bins,quantiles=False)
        self.fc = nn.Dense((self.in_channels*self.num_bins)*14*14,10)
        self.softmax_op = ops.Softmax(axis=1)
        self.elementwise_mul_op = ops.Mul()
        self.reshape_op = ops.Reshape()
        self.l2_normalize = ops.L2Normalize()

    def construct(self, X):
        """Forward pass of the network.
        """
        N = X.shape[0]
        
        X2=self.features2(X)
        X2=self.projection(X2)#torch.Size([8, 512, 28, 28])
        X2=self.max(X2)#torch.Size
        X2_w=self.SAfeatures(X2)

        
        
        X1=self.features1_3(X)#torch.Size([8, 512, 14, 14])
        X1=X1*X2_w

        X =self.l2_normalize(self.reshape_op(X1, (N, -1)))
        
        hist = mindspore.Tensor(np.zeros((N,X.shape[1]*self.num_bins)),ms.float32)
        for i,x in enumerate(X):
            hist[i]=self.histogram(x)

        # print(X.size())
        X = self.fc(hist)
        
        # assert X.size() == (N, numbin)
        X=self.softmax_op(X)
        return X
    
"""
训练
"""
import mindspore as ms
import stat
# 定义VGG16网络，此处不采用预训练，即将pretrained设置为False
vgg16 = DBCNN()
for param in vgg16.get_parameters():
    param.requires_grad = True
#param.requires_grad = True表示所有参数都需要求梯度进行更新。


# 设置训练的轮数和学习率，这里训练的轮数设置为10
num_epochs = 500
#基于余弦衰减函数计算学习率。学习率最小值为0.0001，最大值为0.0005，具体API见文档https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.cosine_decay_lr.html?highlight=cosine_decay_lr
lr = 1e-3
# 定义优化器和损失函数
#Adam优化器，具体可参考论文https://arxiv.org/abs/1412.6980
opt = nn.Adam(params=vgg16.trainable_params(), learning_rate=lr)
# 交叉熵损失
loss_fn1 = EMDLoss()
# loss_fn2 = MQuantileLoss()
#前向传播，计算loss
def forward_fn(inputs, targets):
    logits = vgg16(inputs)
    loss = loss_fn1(logits, targets)
    return loss,logits

#计算梯度和loss
grad_fn = ops.value_and_grad(forward_fn, None, opt.parameters,has_aux=True)

def train_step(inputs, targets):
    (loss,_), grads = grad_fn(inputs, targets)
    opt(grads)
    return loss

# 实例化模型
model = ms.Model(vgg16, loss_fn1, opt, metrics={'loss': nn.Loss()})


# 创建迭代器
data_loader_train = dataset_train.create_tuple_iterator(num_epochs=num_epochs)
data_loader_val = dataset_val.create_tuple_iterator(num_epochs=num_epochs)

# 最佳模型存储路径
best_loss = 10
best_ckpt_dir = "./BestCheckpoint"
best_ckpt_path = "./BestCheckpoint/vgg16-best.ckpt"

import os

from mindspore.ops import operations as ops
def EMD(y_true, y_pred):
    cdf_ytrue = np.cumsum(y_true, axis=-1)
    cdf_ypred = np.cumsum(y_pred, axis=-1)
    samplewise_emd = np.sqrt(np.mean(np.square(np.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return np.mean(samplewise_emd)
def JSD(y_true, y_pred):
    M=(y_true+y_pred)/2
    entropy_op = ops.Entropy()
    js=0.5*entropy_op(y_true, M)+0.5*entropy_op(y_pred, M)
    return js
def histogram_intersection(h1, h2):
    intersection = 0
    for i in range(len(h1)):
        intersection += min(h1[i], h2[i])
    return intersection
# 开始循环训练
print("Start Training Loop ...")

for epoch in range(num_epochs):
    losses = []
    vgg16.set_train()

    # 为每轮训练读入数据

    for i, (images, labels) in enumerate(data_loader_train):
        loss = train_step(images, labels)
        if i%100 == 0 or i == step_size_train -1:
            print('Epoch: [%3d/%3d], Steps: [%3d/%3d], Train Loss: [%5.3f]'%(
                epoch+1, num_epochs, i+1, step_size_train, loss))
        losses.append(loss)
    # 每个epoch结束后，验证准确率

    loss = model.eval(dataset_val)['loss']

    print("-" * 50)
    print("Epoch: [%3d/%3d], Average Train Loss: [%5.3f], loss: [%5.3f]" % (
        epoch+1, num_epochs, sum(losses)/len(losses), loss
    ))
    print("-" * 50)

    if loss< best_loss:
        best_loss = loss
        if not os.path.exists(best_ckpt_dir):
            os.mkdir(best_ckpt_dir)
        if os.path.exists(best_ckpt_path):
            os.chmod(best_ckpt_path, stat.S_IWRITE)#取消文件的只读属性，不然删不了
            os.remove(best_ckpt_path)
        ms.save_checkpoint(vgg16, best_ckpt_path)

print("=" * 80)
print(f"End of validation the best loss is: {best_loss: 5.3f}, "
      f"save the best ckpt file in {best_ckpt_path}", flush=True)
