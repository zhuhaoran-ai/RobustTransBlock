import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
def FFT_for_Period(x, k=2):
    # [B, T, C]  batch  时间框   channel
    xf = torch.fft.rfft(x, dim=1)
    #将一段时域变为许多个正余弦曲线，取他们频率的最小公倍数 即可得到周期
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)    #在batch上平均，在feature维度上平均  得到频域序列
    
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)   #在list中取最大的k个频率对应的下标，构成新的list
    top_list = top_list.detach().cpu().numpy()
    
    period = x.shape[1] // top_list# 将x的第一个维度（时间）大小整除频率的下标，分为不同的周期片段  注意是shape  48//5 计算分为多少个周期
    
    
    return period, abs(xf).mean(-1)[:, top_list] #返回周期片段长度的列表，xf在特征方向的平均，并且，输出 对应的周期对应的频率列表
#xf 在feature 方向平均，前两个维度为，batchsize，频率     选取对应的列， top_list 为最大的k个频率对应的下标    数目为batchsize个的频率
                                                                     #最后是个shape为（batchsize，5）的矩阵
class TimesBlock(nn.Module):
    def __init__(self,configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = 5#5 help='for TimesBlock'
        self.d_model=configs.d_model
        self.d_ff=configs.d_ff
        self.num_kernels=configs.num_kernels
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(16, 32,#512的channel  d_ff为2048
                              self.num_kernels),    #此为已经建好的网络
            nn.GELU(),
            Inception_Block_V1(32, 16,   #从2048到512 channel
                               self.num_kernels)
        )
        
    def forward(self, x):
        B, T, N = x.size()
        
        period_list, period_weight = FFT_for_Period(x, self.k)#k为5   周期片段对应的时间片段的个数,以及period为时间的权重
        out_list=[]
        res = []   #建立一个空列表
        for i in range(self.k):
            period = period_list[i]  #选取周期片段的长度
            # padding
            if (self.seq_len + self.pred_len) % period != 0:  #如果有余数的话
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period #多加个周期  总长度片段
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)#建立多加的片段
                out = torch.cat([x, padding], dim=1)#将其与序列维度拼接
            else:
                length = (self.seq_len + self.pred_len) #否则，就为原长度
                out = x
            # reshape
            out = out.reshape(B, length // period, period,#转换维度  #转换为四维，即时序由一维变为的二维，长度为周期个数，周期长度
                              N).permute(0, 3, 1, 2).contiguous()#并交换维度  特征值为1 时序信息位于2 3   周期的个数维度，周期的具体值维度
            
            # 2D conv: from 1d Variation to 2d Variation  从一维变量到了二维变量
            out = self.conv(out)  #开始进行前向传播
            #for i in range(period):
             #   out1=out[:,:,:,i]
              #  out1=out1.permute(0, 2,1)
               # out2,attn=self.encoder(out1, attn_mask=None)
                #out_list.append(out2)

            #out=torch.stack(out_list)
            #out_list=[]
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)  #时间由二维变为一维
            res.append(out[:, :(self.seq_len + self.pred_len), :]) #生成序列之前   为什么去掉了补加的周期片段，因为需要保证两个维度相同才能进行堆叠

        res = torch.stack(res, dim=-1)  #把不同周期长度生成的结果矩阵 计入列表中 并堆叠起来 由三维变成了四维 有个维度为堆叠的厚度 即为k的值
        
        # adaptive aggregation              res的维度为（B，T，N，K）
        period_weight = F.softmax(period_weight, dim=1)  #对batch个频率大小 进行  （batch_size，k）的矩阵  将频率方向的进行概率化 概率的权重
        period_weight = period_weight.unsqueeze(      #扩展第一维   batch_size,k,1,1
            1).unsqueeze(1).repeat(1, T, N, 1)   #T为 时间序列长度，N为feature值  四维（batch_size,T,N,k）  建立自适应模型
        res = torch.sum(res * period_weight, -1)   #让其 概率值乘结果  对不同的序列进行加权处理，自适应模型，并在k方向上进行求和
        # residual connection                           #维度最终变为（B，T，N）
        res = res + x  #resnet链接，跨层链接(输入的值加上输出的值)，解决梯度消失的问题
        return res