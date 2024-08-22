import math

import torch
from torch import nn
from torch.nn import Parameter, init

##### Transformer-Encoding #########
class PositionEncoder(nn.Module):
    """Position encoder used by transformer."""

    def __init__(self, d_model, seq_len, dropout=0.1):
        super(PositionEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0., seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)  # 缓冲区变量，参数不更新

    def forward(self, x):
        """Add position embedding to input tensor.

        :param x: size(batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class LearnablePositionEncoder(nn.Module):
    """Learnable Position encoder used by transformer."""

    def __init__(self, d_model, seq_len, dropout=0.1):
        super(LearnablePositionEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = Parameter(torch.zeros(1, seq_len, d_model),
                            requires_grad=True)
        init.trunc_normal_(self.pe, std=0.2)

    def forward(self, x):
        """Add position embedding to input tensor.

        :param x: size(batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class SelfAttentionModel(nn.Module):
    """multi-head Self-Attention Model."""

    def __init__(self, d_model, nhead=16, dropout=0.1, batch_first=False, norm_first=False):
        super(SelfAttentionModel, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x, mask=None, src_key_padding_mask=None):
        """Forward transformer layer.

        :param x: size(batch_size, seq_len, d_model)
        :param mask: size(seq_len, seq_len)  ;the float/binary/byte mask for the src sequence
        :param src_key_padding_mask: size(batch_size, seq_len)  ;the binary mask for the src keys per batch

        :output: [batch_size, seq_len, d_model]
        """
        if self.norm_first:
            x = x + self._sa_block(self.norm(x), mask, src_key_padding_mask)
        else:
            x = self.norm(x + self._sa_block(x, mask, src_key_padding_mask))

        return x

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout(x)

class TransformerEModel(nn.Module):
    """Transformer Encoder layer with position encoding."""

    def __init__(self, d_model, seq_len=8, nhead=16,
                 dim_feedforward=256, num_layers=1, dropout=0.1, position_emb='periodic'):
        super(TransformerEModel, self).__init__()
        self.position_emb = position_emb
        if self.position_emb == 'periodic':
            self.position_encoder = PositionEncoder(d_model, seq_len, dropout)
        elif self.position_emb == 'learnable':
            self.position_encoder = LearnablePositionEncoder(d_model, seq_len, dropout)
        else:
            self.position_encoder = nn.Dropout(dropout) if dropout > 0 else nn.Identity()  # 恒等映射(nn.Identity())，但是需要dropout

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True)
        self.transformerE = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers)

    def forward(self, x, mask=None, src_key_padding_mask=None):
        """Forward transformer layer.

        :param x: size(batch_size, seq_len, d_model)
        :param mask: size(seq_len, seq_len)  ;the float/binary/byte mask for the src sequence
        :param src_key_padding_mask: size(batch_size, seq_len)  ;the binary mask for the src keys per batch

        :output: [batch_size, seq_len, d_model]
        """
        __input = self.position_encoder(x)
        __output = self.transformerE(__input, mask=mask, src_key_padding_mask=src_key_padding_mask)
        return __output

