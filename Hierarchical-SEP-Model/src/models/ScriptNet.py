from .STPredictor import STPredictor
from .base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.util.tools import *
from config import DefaultConfig
from sklearn import preprocessing
import pickle as pkl
import numpy as np

class ScriptNet(BaseModel):
    def __init__(self, cfg=DefaultConfig()):
        super(ScriptNet, self).__init__(cfg)
    #    BaseModel.__init__(self, cfg)

        self.word_embedding = get_word_embedding(cfg.root_path)

        if cfg.model_name == "STPredictor":
            self.model_H = STPredictor(vocab_size= len(self.word_embedding),
                           embedding_size=cfg.embedding_size,
                           word_embedding=self.word_embedding,
                           hidden_size= cfg.hidden_size,
                           dropout=cfg.dropout,
                           seq_len=cfg.seq_len,
                           num_layers= cfg.n_layers,
                           n_heads= cfg.n_heads,
                           dim_feedforward = cfg.dim_feedforward,
                           positional_embedding=cfg.positional_embedding,
                           sequence_map_strategy = cfg.sequence_map_strategy
                           )
        else:
            return

        self.model_names = ['model_H']
        self.acc_names = ['H']

    def set_input(self, input, device):
        self.event_chain = input['event_chain'].to(device)
        self.adj_matrix = input['adj_matrix'].to(device)
        self.label = input['label'].to(device)

    def forward(self):
        self.predict = self.model_H(self.event_chain, self.adj_matrix)

    # 输入是torch
    def predict_func(self, predict, label):
        _, predict = torch.sort(predict, descending=True)
        n_correct = torch.sum((predict[:, 0] == label)).item()
        n_label = label.size(0)
        acc = n_correct / n_label * 100.0
        return acc

    def process_test(self, outlier_index):
        self.forward()
        for index in outlier_index:
            self.predict[index] = -1e9
        acc = self.predict_func(self.predict, self.label)
        return acc

if __name__ == "__main__":
    cfg = DefaultConfig()
    net = ScriptNet(cfg)
    net.print_networks(True)

