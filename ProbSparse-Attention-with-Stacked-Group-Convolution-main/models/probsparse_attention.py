import numpy as np
import torch
import torch.nn as nn
from math import sqrt
from masking import TriangularCausalMask, ProbMask

class ProbAttention(nn.Module):
    def __init__(self, mask_flag, factor, scale, attention_dropout, output_attention):
        super(ProbAttention, self).__init__()
        self.factor = factor  # 用于控制采样数量的因子
        self.scale = scale    # 缩放因子
        self.mask_flag = mask_flag   # 是否使用掩码
        self.output_attention = output_attention  # 是否输出注意力权重
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # 对K进行扩展和采样，calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)

        # 随机生成采样索引，这里index_sample是一个二维数组，形状为(L_Q, sample_k)
        # 它从[0, L_K)范围内随机选取sample_k个整数，总共选取L_Q次
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q

        # 根据采样索引，从K_expand中选取对应的样本，得到K_sample
        # K_sample的形状是(B, H, L_Q, sample_k, E)，它包含了随机选取的K的子集
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]

        # 计算Q和K_sample的点积，这里用matmul来进行矩阵乘法
        # Q.unsqueeze(-2)将Q的形状从(B, H, L_Q, E)变为(B, H, L_Q, 1, E)
        # K_sample.transpose(-2, -1)将K_sample的形状变为(B, H, L_Q, E, sample_k)
        # 然后两者进行矩阵乘法，得到的点积形状为(B, H, L_Q, 1, sample_k)
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        # 使用Q_K_sample计算每个query的最大得分和平均得分的差值M
        # Q_K_sample.max(-1)[0]获取每个query的最大得分
        # torch.div(Q_K_sample.sum(-1), L_K)计算得分的均值
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)

        # 选择M中的前n_top个最大值，即选择最重要的得分
        # M.topk(n_top, sorted=False)[1]返回这些最大值对应的索引
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        # 使用M_top索引从原始的Q中选择对应的query
        # 这里Q_reduce的形状是(B, H, n_top, E)，包含了最重要的query
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # 选择最重要的query

        # 使用Q_reduce和K的转置计算完整的注意力得分Q_K
        # 这次计算不再是对采样后的K_sample，而是对整个K进行
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # 如果没有设置掩码标记，那么直接计算V的平均值作为初始上下文表示。
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # 如果设置了掩码标记，那么要求queries的长度必须等于values的长度（只适用于自注意力机制），
            # 然后计算V的累积和作为初始上下文表示。
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * int(np.ceil(np.log(L_K)))  # c*ln(L_k)
        u = self.factor * int(np.ceil(np.log(L_Q)))  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        # 调用_prob_QK得到简化的注意力得分和索引
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # 添加缩放因子
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        # 获取初始上下文
        context = self._get_initial_context(values, L_Q)

        # 使用选出的顶部查询更新上下文
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        #self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        #self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        #self.value_projection = nn.Linear(d_model, d_values * n_heads)

        # 替换为1x1卷积层
        self.query_projection = nn.Conv1d(in_channels=d_model, out_channels=d_keys * n_heads, kernel_size=1)
        self.key_projection = nn.Conv1d(in_channels=d_model, out_channels=d_keys * n_heads, kernel_size=1)
        self.value_projection = nn.Conv1d(in_channels=d_model, out_channels=d_values * n_heads, kernel_size=1)

        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, x, attn_mask=None):
        B, L, _ = x.shape
        H = self.n_heads

        # 初始queries, keys, values 都设置为 x
        #queries = keys = values = x

        # 然后根据模型的需要对 queries, keys, values 进行线性变换
        #queries = self.query_projection(queries).view(B, L, H, -1)
        #keys = self.key_projection(keys).view(B, L, H, -1)
        #values = self.value_projection(values).view(B, L, H, -1)

        # 转置以匹配1x1卷积的输入期望形状 (batch_size, channels, length)
        x = x.transpose(1, 2)
        # 应用1x1卷积并保持维度
        queries = self.query_projection(x).view(B, H, -1, L).transpose(1, 2).transpose(2, 3)
        keys = self.key_projection(x).view(B, H, -1, L).transpose(1, 2).transpose(2, 3)
        values = self.value_projection(x).view(B, H, -1, L).transpose(1, 2).transpose(2, 3)

        # 调用内部注意力机制并传入 attn_mask 计算注意力并生成输出
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        # 将注意力输出的形状从 (batch_size, heads, d_values, length) 转换为 (batch_size, length, heads * d_values)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn