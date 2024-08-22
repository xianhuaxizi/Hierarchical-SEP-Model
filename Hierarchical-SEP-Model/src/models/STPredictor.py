'''
(c) Pengpeng Zhou
这一个版本，参数部分每四个参数独立建模，使用self-attention机制组织为一个事件表示
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.util.tools import *
from .my_modules import *
from .transformer import *
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class STPredictor(Module):
    def __init__(self, vocab_size, embedding_size, word_embedding, hidden_size, dropout,
                 seq_len, num_layers, n_heads, dim_feedforward,
                 positional_embedding, sequence_map_strategy):
        super(STPredictor, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.positional_embedding = [single_pos if single_pos in ['periodic', 'learnable', 'none'] else 'learnable'
                                     for single_pos in positional_embedding]
        self.sequence_map_strategy = sequence_map_strategy if \
            sequence_map_strategy in ['seq_pool', 'max', 'mean'] else 'seq_pool'

        if self.sequence_map_strategy == 'seq_pool':
            self.attention_pool = nn.Linear(self.hidden_size*2, 1)

        # self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight.data = torch.from_numpy(word_embedding)

        #  event-level context representation
        self.event_level_block = nn.ModuleList()
        self.event_level_block.add_module('arg_transformer_model', TransformerEModel(d_model=self.embedding_size,
                                                                                          seq_len=self.seq_len[0],
                                                                                          nhead=self.n_heads[0],
                                                                                          num_layers=self.num_layers[0],
                                                                                          dim_feedforward=self.dim_feedforward[0],
                                                                                          dropout=self.dropout,
                                                                                          position_emb=self.positional_embedding[0]))

        self.event_level_block.add_module('event_composition', EventComposition(self.embedding_size, self.hidden_size, self.dropout))

        #  chain-level context representation
        self.chain_level_block = nn.ModuleList()
        # self.chain_level_block.add_module('gcn', GCNModel(self.hidden_size, self.num_layers[1], self.dropout))
        self.chain_level_block.add_module('gate_gnn', GateGNN(self.hidden_size, self.dropout))

        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size*2),
            nn.ReLU(),
            nn.Linear(self.hidden_size*2, 1))
        # self.regressor = nn.Linear(self.hidden_size*2, 1)
        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)


    def adjust_event_chain_embedding_type1(self, embedding):
        """
        shape: (batch_size, 52, embedding_size) -> (batch_size * 5, 9, hidden_size)
            52: (8 context_event + 5 candidate event) * 4 arguments
            9: (8 context_event + 1 candidate event)
        """
        embedding = torch.cat(tuple([embedding[:, i:i+13, :] for i in range(0, 52, 13)]), 2)
        context_embedding = embedding[:, 0:8, :].repeat(1, 5, 1).view(-1, 8, self.hidden_size)
        candidate_embedding = embedding[:, 8:13, :].contiguous().view(-1, 1, self.hidden_size)
        event_chain_embedding = torch.cat((context_embedding, candidate_embedding), 1)
        return event_chain_embedding

    def calculate_all_event_mask(self, inputs):
        """
        judge wether the input event is valid.
        shape: (batch_size, 52) -> (batch_size*5, 9)
            52: (8 context_event + 5 candidate event) * 4 arguments
             9: (8 context_event + 1 candidate event)
            True value indicates that the corresponding key value will be ignored for the purpose of attention.
        """
        inputs = torch.cat(tuple([inputs[:, i:i + 13].unsqueeze(0) for i in range(0, 52, 13)]), 0)
        event_mask = inputs.sum(0).to(torch.bool) == 0
        context_event_mask = event_mask[:, 0:8].repeat(1, 5).view(-1, 8)
        candidate_event_mask = event_mask[:, 8:13].contiguous().view(-1, 1)
        event_chain_mask = torch.cat((context_event_mask, candidate_event_mask), 1)
        return context_event_mask, event_chain_mask

    def calculate_all_arg_mask(self, inputs):
        """
        judge wether the input event is valid.
        shape: (batch_size, 52) -> (batch_size*5, 36)
            52: (8 context_event + 5 candidate event) * 4 arguments
            36: (8 context_event + 1 candidate event) * 4 arguments
            True value indicates that the corresponding key value will be ignored for the purpose of attention.
        """
        inputs = torch.cat(tuple([inputs[:, i::13] for i in range(13)]), 1)
        arg_mask = inputs.to(torch.bool) == 0
        context_arg_mask = arg_mask[:, 0:32].repeat(1, 5).view(-1, 32)
        candidate_arg_mask = arg_mask[:, 32:52].contiguous().view(-1, 4)
        arg_chain_mask = torch.cat((context_arg_mask, candidate_arg_mask), 1)
        return arg_chain_mask

    def adjust_event_chain_embedding_type2(self, embedding):
        """
        shape: (batch_size, 52, embedding_size) -> (batch_size * 5, 36, embedding_size)
            52: (8 context_event + 5 candidate event) * 4 arguments
            36: (8 context_event + 1 candidate event) * 4 arguments
        """
        embedding = torch.cat(tuple([embedding[:, i::13, :] for i in range(13)]), 1)
        context_embedding = embedding[:, 0:32, :].repeat(1, 5, 1).view(-1, 32, self.embedding_size)
        candidate_embedding = embedding[:, 32:52, :].contiguous().view(-1, 4, self.embedding_size)
        event_chain_embedding = torch.cat((context_embedding, candidate_embedding), 1)
        return event_chain_embedding

    def adjust_event_chain_embedding_type3(self, embedding):
        """
        shape: (batch_size, 52, embedding_size) ->(batch_size, 52, embedding_size)
        """
        embedding = torch.cat(tuple([embedding[:, i::13, :] for i in range(13)]), 1)
        return embedding

    def adjust_event_embedding(self, embedding):
        """
        shape: (batch_size * 5, 9, hidden_size) -> (batch_size, 13, hidden_size)
        """
        embedding = embedding.view(embedding.size(0) // 5, -1, self.hidden_size)
        context_embedding = torch.zeros(embedding.size(0), 8, self.hidden_size).to(embedding.device)
        for i in range(0, 45, 9):
            context_embedding += embedding[:, i:i+8, :]
        context_embedding /= 8.0  # ???为什么出以8而不是5
        candidate_embedding = embedding[:, 8::9, :]
        event_embedding = torch.cat((context_embedding, candidate_embedding), 1)
        return event_embedding

    def adjust_event_embedding_reverse(self, embedding):
        """
        shape: (batch_size, 13, hidden_size)-> (batch_size * 5, 9, hidden_size)
        """
        context_embedding = embedding[:, 0:8, :].repeat(1, 5, 1).view(-1, 8, self.hidden_size)
        candidate_embedding = embedding[:, 8:13, :].contiguous().view(-1, 1, self.hidden_size)
        event_chain_embedding = torch.cat((context_embedding, candidate_embedding), 1)
        return event_chain_embedding

    def get_params(self):
        model_grad_params = filter(lambda p: p.requires_grad, self.parameters())
        train_params = list(map(id, self.embedding.parameters()))
        tune_params = filter(lambda p: id(p) not in train_params, model_grad_params)
        return tune_params

    def forward(self, inputs, matrix):
        # embedding layer
        inputs_embed = self.embedding(inputs)

        # 1. 事件的四个参数经过非线性组合,组合后的维度取决于hidden_size
        # inputs_embed = self.adjust_event_chain_embedding_type3(inputs_embed)
        # event_embed_level1 = self.event_level_block.event_composition(inputs_embed)

        # 2. 使用arguments self-attention组合事件参数
        inputs_embed = self.adjust_event_chain_embedding_type2(inputs_embed)
        # 上三角
        # arg_mask = compute_mask(self.seq_len[0]).to(inputs_embed.device)
        # 块上三角
        arg_mask = compute_arg_mask(self.seq_len[0]).to(inputs_embed.device)
        arg_embed = self.event_level_block.arg_transformer_model(inputs_embed, mask=arg_mask, src_key_padding_mask=None)
        event_embed_level1 = self.event_level_block.event_composition(arg_embed)
        event_embed_level1 = self.adjust_event_embedding(event_embed_level1)

        #  event-level context representation by gnn
        # event_embed_level2_all = self.chain_level_block.gcn(event_embed_level1, matrix)
        event_embed_level2_all = self.chain_level_block.gate_gnn(event_embed_level1, matrix)
        event_embed_level2_gnn = self.adjust_event_embedding_reverse(event_embed_level2_all)

        #  local matching
        event_l = event_embed_level2_gnn[:, 0:8, :]
        event_r = event_embed_level2_gnn[:, 8:, :]
        event_mul = torch.mul(event_l, event_r)
        event_dif = torch.abs(torch.add(event_l, -event_r))
        event_embed_level3 = torch.cat([event_mul, event_dif], -1)  # vertex match vectors.

        #  global matching
        if self.sequence_map_strategy == 'seq_pool':
            event_embed_level4 = torch.matmul(F.softmax(self.attention_pool(event_embed_level3), dim=1).transpose(-1, -2), event_embed_level3).squeeze(-2)
        elif self.sequence_map_strategy == 'max':
            event_embed_level4 = event_embed_level3.max(dim=1)[0]
        elif self.sequence_map_strategy == 'mean':
            event_embed_level4 = event_embed_level3.mean(dim=1)

        event_score = self.regressor(event_embed_level4).view(-1, 5)

        return event_score

