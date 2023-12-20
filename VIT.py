import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 正则化
        self.fn = fn  # 具体的操作
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        # 前向传播
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    # attention
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads  # 计算最终进行全连接操作时输入神经元的个数
        project_out = not (heads == 1 and dim_head == dim)  # 多头注意力并且输入和输出维度相同时为True

        self.heads = heads  # 多头注意力中“头”的个数
        self.scale = dim_head ** -0.5  # 缩放操作，论文 Attention is all you need 中有介绍

        self.attend = nn.Softmax(dim = -1)  # 初始化一个Softmax操作
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  # 对Q、K、V三组向量先进性线性操作

        # 线性全连接，如果不是多头或者输入输出维度不相等，进行空操作
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads  # 获得输入x的维度和多头注意力的“头”数
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # 先对Q、K、V进行线性操作，然后chunk乘三三份
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)  # 整理维度，获得Q、K、V

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # Q, K 向量先做点乘，来计算相关性，然后除以缩放因子

        attn = self.attend(dots)  # 做Softmax运算

        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # Softmax运算结果与Value向量相乘，得到最终结果
        out = rearrange(out, 'b h n d -> b n (h d)')  # 重新整理维度
        return self.to_out(out)  # 做线性的全连接操作或者空操作（空操作直接输出out）

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])  # Transformer包含多个编码器的叠加
        for _ in range(depth):
            # 编码器包含两大块：自注意力模块和前向传播模块
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),  # 多头自注意力模块
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))  # 前向传播模块
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            # 自注意力模块和前向传播模块都使用了残差的模式
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size[0] % patch_size[0] == 0, 'Image dimensions must be divisible by the patch size.'  # 保证一定能够完整切块
        assert image_size[1] % patch_size[1] == 0, 'Image dimensions must be divisible by the patch size.'  # 保证一定能够完整切块
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])  # 获取图像切块的个数
        patch_dim = channels * patch_size[0] * patch_size[1]  # 线性变换时的输入大小，即每一个图像宽、高、通道的乘积
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'  # 池化方法必须为cls或者mean

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size[0], p2 = patch_size[1]),  # 将批量为b通道为c高为h*p1宽为w*p2的图像转化为批量为b个数为h*w维度为p1*p2*c的图像块
                                                                                                    # 即，把b张c通道的图像分割成b*（h*w）张大小为P1*p2*c的图像块
                                                                                                    # 例如：patch_size为16  (8, 3, 48, 48)->(8, 9, 768)
            nn.Linear(patch_dim, dim),  # 对分割好的图像块进行线性处理（全连接），输入维度为每一个小块的所有像素个数，输出为dim（函数传入的参数）
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 位置编码，获取一组正态分布的数据用于训练
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 分类令牌，可训练
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)  # Transformer模块

        self.pool = pool
        self.to_latent = nn.Identity()  # 占位操作

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),  # 正则化
            nn.Linear(dim, num_classes)  # 线性输出
        )

    def forward(self, img):
        print("原图像的大小：", img.shape)
        x = self.to_patch_embedding(img)  # 切块操作，shape (b, n, dim)，b为批量，n为切块数目，dim为最终线性操作时输入的神经元个数
        print("shape (b, n, dim), b为批量，n为切块数目，dim为最终线性操作时输入的神经元个数")
        print("b批量:", x.shape[0])
        print("n切块数目:", x.shape[1])
        print("dim为最终线性操作时输入的神经元个数:", x.shape[2])
        b, n, _ = x.shape  # shape (b, n, 1024)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)  # 分类令牌，将self.cls_token（形状为1, 1, dim）赋值为shape (b, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # 将分类令牌拼接到输入中，x的shape (b, n+1, 1024)
        x += self.pos_embedding[:, :(n + 1)]  # 进行位置编码，shape (b, n+1, 1024)
        x = self.dropout(x)

        x = self.transformer(x)  # transformer操作

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)  # 线性输出

if __name__=='__main__':
    # model = torchvision.models.resnet50()

    # 模型测试
    v = ViT(
        image_size = [800, 400],   #int 类型参数，图片大小。 如果您有矩形图像，请确保图像尺寸为宽度和高度的最大值
        patch_size = [40, 20],    #int 类型参数，patches数目。image_size 必须能够被 patch_size整除。
        num_classes = 256, #int 类型参数，分类数目。
        dim = 1024,         #int 类型参数，线性变换nn.Linear(..., dim)后输出张量的尺寸 。
        depth = 6,          #int 类型参数，Transformer模块的个数。
        heads = 16,         #int 类型参数，多头注意力中“头”的个数。
        mlp_dim = 2048,     #int 类型参数，多层感知机中隐藏层的神经元个数。
        channels= 1,        #int 类型参数，输入图像的通道数，默认为3。
        dropout = 0.1,      #float类型参数，Dropout几率，取值范围为[0, 1]，默认为 0.。
        emb_dropout = 0.1,  #float类型参数，进行Embedding操作时Dropout几率，取值范围为[0, 1]，默认为0。
        pool="cls"          #string类型参数，取值为 cls或者 mean 。
    )

    # img = torch.randn(1, 3, 256, 256)
    img = torch.randn(2, 1, 800, 400)

    preds = v(img)
    print("网络的最终输出：", preds.shape)
