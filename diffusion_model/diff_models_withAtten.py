import torch
from inspect import isfunction
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

def get_torch_trans(heads=8, layers=1, channels=64):
    '''
    函数的作用是构建一个 Transformer 编码器，用于处理序列数据。
    参数包括头数 (heads)、层数 (layers) 和通道数 (channels)。
    '''
    #首先创建一个 Transformer 编码器层 (nn.TransformerEncoderLayer)
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    
    #然后创建一个 Transformer 编码器 (nn.TransformerEncoder)，并返回该编码器的实例
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    '''
    定义并Kaiming初始化一个1D卷积层
    '''
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        '''
        num_steps:扩散步数, projection_dim:投影维度
        '''
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )#注册为模块的缓冲区（buffer）
        
        # 创建两个线性投影层，用于将嵌入映射到指定的维度
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        '''
        创建嵌入表
        '''
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        # unsqueeze(1) 是 PyTorch 的张量操作，它在指定的维度上插入一个新的维度。
        # 在这里，(1) 表示在第一个维度（即维度索引为 1 的位置）插入一个新的维度。
        # 通过这个操作，一维张量就变成了二维张量，其中原本的整数序列被当作列向量。
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        # unsqueeze(0):在第0维度插入一个新的维度，将原本的一维张量变成了一个行向量。
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_open(nn.Module):
    def __init__(self, config, inputdim=1):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)



        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        B, L = x.shape

        x = x.unsqueeze(1)#B,1, L


        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, L)
        x = self.output_projection1(x)  # (B,channel,L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,L)
        x = x.reshape(B, L)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__() #调用父类的构造方法
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim // 2, 2 * channels, 1)
        self.cond_projection2 = Conv1d_with_init(side_dim // 2, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=2*channels)


        self.cross = CrossAttention(2*channels, 2*channels)
        self.norm1 = nn.LayerNorm(2*channels)
        self.ff = FeedForward(2*channels, dropout=0)
        self.norm2 = nn.LayerNorm(2*channels)
    def forward_time(self, y, base_shape):
        B, channel, L = base_shape
        if L == 1:
            return y
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel,  L)
        # 对扩散嵌入进行线性投影，并加上输入数据
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb #

        # condition先拆分：
        cond1, cond2 = torch.chunk(cond_info, 2, dim=2)


        # # 重新设计算法：
        _, _, cond_dim = cond1.shape
        cond1 = cond1.permute(0,2,1)
        cond1 = self.cond_projection(cond1)  # (B,2*channel,L)
        y = self.forward_time(y, base_shape) # (B,channel,L)
        y = self.mid_projection(y)  # (B,2*channel,L),1D


        cond2 = self.feature_layer(cond2).permute(0,2,1)  # (B,2*channel,L)
        y = self.cross(self.norm1(y.permute(0,2,1)).permute(0,2,1), cond2) + y
        y = self.ff(self.norm2(y.permute(0,2,1))).permute(0,2,1) + y


        y = y + cond1

        # 执行门控操作，使用 sigmoid 函数作为门控，tanh 函数作为过滤器
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,L)
        y = self.output_projection(y)

        # 分离残差项和用于跳跃连接的项
        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        
        # 返回残差块的输出，包括残差项和用于跳跃连接的项
        return (x + residual) / math.sqrt(2.0), skip

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, condition, mask=None):
        h = self.heads
        x =x.permute(0,2,1)
        condition = condition.permute(0,2,1)
        q = self.to_q(x)
        k = self.to_k(condition)
        v = self.to_v(condition)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out).permute(0,2,1)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        )

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)



    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x