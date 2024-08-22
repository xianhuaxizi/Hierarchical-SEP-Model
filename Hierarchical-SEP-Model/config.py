# coding:utf8
import warnings
import torch as t

class DefaultConfig(object):
    def __init__(self):
        # Model structure
        self.model_arch = 'ScriptNet'  # 固定不变
        self.model_name = 'STPredictor'

        ## GCN
        # 这里针对事件层的输出如何将多个序列输出转换为单个序列, seq_pool, max, mean
        self.sequence_map_strategy = 'seq_pool'
        self.positional_embedding = ['learnable', 'learnable']           # 这里是针对参数层的建模来说, 'periodic', 'learnable', 'none'
        self.embedding_size = 128                       # size of argument
        self.hidden_size = self.embedding_size*4        # size of event, 注意对于STPredictor_2的单独v的嵌入，则只能不包含4
        self.dim_feedforward = [1024, 1024]              # the dimension of the feedforward network model
        self.n_layers = [1, 2, 2]                          # the number of sub-encoder-layers in the event encoder, gcn, or chain encoder
        self.n_heads = [4, 16]                          # 分别对应着不同层面（argument, event）的self-attention or transformer,
        self.seq_len = [36, 13]                         # 参数层的序列个数，事件层的序列个数

        # Training strategy
        self.num_workers = 4      # 不使用多线程
        self.max_epochs = 50      # number of total epochs to run
        self.dropout = float(0.1)
        self.margin = float(0.05)
        self.lr = 2.0e-4      # initial learning rate
        self.weight_decay = 5e-4  # 损失函数
        self.momentum = 0.9
        self.train_batch = 2000  # train batchsize
        self.patients = int(5)  # Number of epochs with no improvement after which learning rate will be reduced
        self.schedule_lr = 10  #每10次下降一次学习率
        self.lr_decay = 0.4    # when val_loss increase, lr = lr*lr_decay  0.1
        self.warm_up_epochs = 5  # the epoches of warmup
        self.lr_policy = 'plateau'   # constant  plateau warmup_constant

        self.log_interval = 5  # print log info and save history model every N epoch, if None,  log_interval = int(np.ceil(max_epochs * 0.02))
        self.betas = (0.9, 0.99)
        self.device = "cuda:0"

        # Data processing
        self.dataset = "NYT"
        self.root_path = "../data/metadata/" # 训练集存放路径
        self.seed = 17  #随机数种子

        # Miscs
        self.checkpoint = "./checkpoints/"  # path to save checkpoint
        self.visible = True   # 是否显示中间结果 tensorboard

    def parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

    def __str__(self):
        config = ""
        for name, value in vars(self).items():
            config += ('%s=%s\t\n' % (name, value))
        return config


if __name__ == '__main__':
    opt = DefaultConfig()
    new_config = {'lr': 0.1, 'device': "CPU"}
    opt.parse(new_config)
    print(opt)
