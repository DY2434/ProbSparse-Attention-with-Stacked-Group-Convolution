import torch
import torch.nn as nn
import copy
from masking import TriangularCausalMask, ProbMask
from probsparse_attention import ProbAttention, AttentionLayer
from stacked_group_convolution import ConvolutionBlock1, ConvolutionBlock2

class PositionalEembedding(nn.Module):
    def __init__(self, d_model, max_len=1500):
        super(PositionalEembedding, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        seq_len, d_model = x.size(1), x.size(2)
        pos_embed = self.pos_embed[:, :seq_len, :d_model]
        x = x + pos_embed
        return self.dropout(x)


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(ResidualAdd, self).__init__()
        self.fn = fn  # fn是一个序列，包含了层正规化、多头注意力等

    def forward(self, x, **kwargs):  # 接受额外的关键字参数
        res = x
        # 确保在调用self.fn时传递所有必需的参数
        x = self.fn(x, **kwargs)  # 这里传递额外的参数
        #print(type(x))  # 添加打印语句来检查x的类型
        if isinstance(x, tuple):
            #print("Warning: x is a tuple.")  # 如果x是tuple，打印警告
            x = x[0]  # 只需要tuple中的第一个元素

        x += res
        return x


#提供一种灵活的方式来顺序执行一系列神经网络模块，并在需要的时候传递额外的参数
class CustomSequential(nn.Module):
    def __init__(self, *args):
        super(CustomSequential, self).__init__()
        self.modules_list = nn.ModuleList(args)

    def forward(self, x, attn_mask=None):
        for module in self.modules_list:
            if 'attn_mask' in module.forward.__code__.co_varnames:
                x = module(x, attn_mask=attn_mask)
            else:
                x = module(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, x, attn_mask=None):
        output = x
        for layer in self.layers:
            output = layer(output, attn_mask=attn_mask)
        return output


class TransformerModel(nn.Module):
    def __init__(self, num_classes, d_model, nhead, num_layers, drop_p=0.3, convolution_hidden_dim=342):
        super().__init__()
        self.adaptive_pool1 = nn.AdaptiveMaxPool1d(250) #Avg Max
        self.LN1 = nn.LayerNorm(d_model)  # 加快模型收敛
        self.pos_encoder = PositionalEembedding(d_model)
        self.encoder_layer = CustomSequential(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(d_model),
                AttentionLayer(
                    ProbAttention(mask_flag=False, factor=5, scale=None, 
                                  attention_dropout=0.1, output_attention=False),
                    d_model, nhead, d_model // nhead, d_model // nhead)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(d_model),
                ConvolutionBlock1(d_model, convolution_hidden_dim, 
                                  kernel_size=3, dropout_rate=0.5, groups=3)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(d_model),
                ConvolutionBlock2(250, 250, 
                                  kernel_size=3, dropout_rate=0.5, groups=2)
            ))
        )
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers)

        self.LN2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(drop_p)  # 添加Dropout层

        self.adaptive_pool2 = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(d_model, num_classes)

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.transpose(1, 2)
        x = self.adaptive_pool1(x)
        x = x.transpose(1, 2)
        x = self.LN1(x)
        x = self.pos_encoder(x)

        # 获取批次大小B和序列长度L
        B, L = x.size(0), x.size(1)

        # 创建注意力掩码
        attn_mask = TriangularCausalMask(B, L, device=x.device)

        x = self.transformer_encoder(x, attn_mask=attn_mask)
        x = self.LN2(x)
        x = self.dropout(x)  # 应用Dropout
        x = self.adaptive_pool2(x.transpose(1, 2)).squeeze(2)  # 使用自适应池化

        x = self.dropout(self.LN2(x))
        x = self.fc1(x)
        # 使用nn.CrossEntropyLoss损失函数，已经内置了softmax，故训练时不需要额外添加softmax。
        # 最后需要的是预测的类别，验证测试阶段预测时都使用了torch.argmax()函数获取预测结果的类别索引，故不需要额外添加softmax。
        return x