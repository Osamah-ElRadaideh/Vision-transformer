import torch.nn as nn
import numpy as np
import torch

class Patcher(nn.Module):
    #divide a square images into non overlapping patches
    def __init__(self, num_channels,out_channels, patch_size):
        super().__init__()
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.conv = nn.Conv2d(num_channels, out_channels,kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
    def forward(self, x):
        assert x.shape[-1] % self.patch_size == 0, 'image shape must be divisble by the patch size to create equal patches'
        x = self.conv(x)
        x = self.flatten(x)
        return x.permute(0, 2 ,1)
    

class Attentionblock(nn.Module): 
    def __init__(self, patch_dim, num_heads, dropout=0.1):
        super().__init__()
        self.layernorm = nn.LayerNorm(patch_dim)
        self.attention = nn.MultiheadAttention(patch_dim,num_heads=num_heads,dropout=dropout,batch_first=True)
    def forward(self, x):
        x = self.layernorm(x)
        x, _ = self.attention(key=x, query=x, value=x, need_weights=False )
        return x
    


class FF(nn.Module):
    #the feedfoward block in the paper
    def __init__(self, patch_dim, FF_dim, dropout=0.1):
        super().__init__()
        self.layernorm= nn.LayerNorm(patch_dim)
        self.fc = nn.Linear(patch_dim, FF_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(FF_dim,patch_dim)
    def forward(self, x):
        x = self.layernorm(x)
        x = self.fc(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)
    

class Encoder(nn.Module):
    #manual implementation of the transformer encoder layer, 
    def __init__(self,patch_dim, num_heads,FF_dim,FF_dropout,attention_dropout):
        super().__init__()
        self.attention = Attentionblock(patch_dim, num_heads, attention_dropout)
        self.FF = FF(patch_dim,FF_dim, FF_dropout)
    def forward(self, x):
        x = self.attention(x) + x
        x = self.FF(x) + x 
        return x 
    

class VIT(nn.Module):
    #the complete vision transformer model
    def __init__(self, input_size=32, in_channels=3, patch_size=8, num_layers=12, patch_dim=192, FF_dim=3072, num_heads=12, FF_dropout=0.1, att_dropout=0, n_classes=100):
        super().__init__()
        self.num_patches = input_size ** 2 // patch_size ** 2
        #the class embeddings learned by the transformer
        self.class_emb = nn.Parameter(data=torch.randn(1, 1, patch_dim),
                                            requires_grad=True)
        
        #positional embedding, equivalent to the positional encoding of text
        self.position_emb = nn.Parameter(data=torch.randn(1, self.num_patches+1, patch_dim),
                                               requires_grad=True)
        self.emb_dropout = nn.Dropout(0.1)
        self.patcher = Patcher(in_channels, patch_dim, patch_size)
        self.encoder = nn.Sequential(*[Encoder(patch_dim, num_heads, FF_dim, FF_dropout,att_dropout) for i in range(num_layers)])
        self.layernorm = nn.LayerNorm(patch_dim)
        self.fc = nn.Linear(patch_dim, n_classes)
    def forward(self, x):
        bs = x.shape[0]
        x = x.permute(0,-1,1,2)
        x = self.patcher(x)
        #tokenize the classes 
        class_token = self.class_emb.expand(bs, -1, -1)
        x = torch.cat([class_token, x], dim=1)
        x - self.position_emb + x
        x = self.emb_dropout(x)
        x = self.encoder(x)
        x = self.layernorm(x)
        x = self.fc(x[:, 0])
        return x 