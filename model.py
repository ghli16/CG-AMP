import torch
from torch_geometric.nn import global_max_pool
import torch.nn as nn
import torch.nn.functional as F

from module import TransformerLayer, GatedCon, SelfAttention, Attention




class newModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.aa = nn.Linear(1280, 512)
        self.LN = nn.LayerNorm(512)
        # self.pool = global_max_pool
        self.pool = nn.MaxPool1d(kernel_size=183)
        self.self_att = Attention(512)   # 45
        self.cnn = GatedCon(45, 256, 3, 3, 0.1, 'cuda')
        # self.po = PositionwiseFeedforward(512)
        self.atte_fusion = nn.MultiheadAttention(64, 4)
        self.line_fusion = nn.Linear(128, 64)
        self.filter_sizes = [1, 2, 3, 4, 6, 8, 16, 32]
        filter_num = 64

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (fsz, 45)) for fsz in self.filter_sizes])
        # [nn.Conv2d(1, filter_num, (fsz, 45)) for fsz in self.filter_sizes])
        self.dropout = nn.Dropout(0.5)

        # Bi-directional LSTM layer
        self.lstm = nn.LSTM(input_size=len(self.filter_sizes) * filter_num, # 45
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)

        # Adjusting the linear layer's input size to account for BiLSTM
        self.block1 = nn.Sequential(nn.Linear(256, 256),  # 2 * hidden_size for BiLSTM
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 64),
                                    )

        self.block2 = nn.Sequential(nn.Linear(256*3, 256),  # 2 * hidden_size for BiLSTM
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 64),

                                    )

        # transformer
        # Encoder layers
        # Embedding layers
        self.input_block = nn.Sequential(
            nn.LayerNorm(1280, eps=1e-6),
            nn.Linear(1280, 512),
            nn.LeakyReLU(),
        )

        self.hidden_block = nn.Sequential(
            nn.LayerNorm(128, eps=1e-6)
            , nn.Dropout(0.2)
            , nn.Linear(128, 128)
            , nn.LeakyReLU()
            , nn.LayerNorm(128, eps=1e-6)
        )

        self.encoder_layers = nn.ModuleList([
            TransformerLayer(512, 4, 0.3)  # 45,1,0.3
            for _ in range(2)
        ])

        self.readout = nn.Sequential(
            nn.LayerNorm(512, eps=1e-6),
            nn.Linear(512, 256),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
        )

        self.classification = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, features1, features2):
        features1 = features1.squeeze(1)
        esm = self.input_block(features1)

        # for t in self.encoder_layers:
        #     esm = t(esm)
        esm = esm.permute(0, 2, 1)  # (batch, len, dim  ->  batch, dim, len)
        esm = self.pool(esm)
        esm = esm.permute(0, 2, 1)  # (batch, dim, 1  ->  batch, 1, dim)
        esm = esm.view(esm.size(0), -1)
        esm = self.readout(esm)
        x = features2
        x = self.cnn(x)
        # x = self.selfa_att(x)
        # for t in self.encoder_layers:
        #     x = t(x)
        # x = x.unsqueeze(1)
        # x = [F.relu(conv(x)) for conv in self.convs]
        # x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]
        # x = [x_item.view(x_item.size(0), -1) for x_item in x]
        # x = [x_item.view(x_item.size(0), -1) for x_item in x]
        # x = torch.cat(x, 1)


        # for t in self.encoder_layers:
        #     x = t(x)

        x = x.permute(0, 2, 1)   # (batch, len, dim  ->  batch, dim, len)
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # (batch, dim, 1  ->  batch, 1, dim)
        x = x.view(x.size(0), -1)

        # # Pass through LSTM
        # x = x.view(x.size(0), 1, -1)  # (batch_size, seq_len, features)
        # lstm_out, _ = self.lstm(x)
        # x = lstm_out[:, -1, :]
        # x = x.view(x.size(0), -1)

        x = self.block2(x)

        a = torch.cat((x, esm), dim=1)
        # a, _ = self.atte_fusion(esm, esm, x, need_weights=False)
        a = self.line_fusion(a)

        return a, esm, x

    def trainmodel(self, features1, features2):
        output, _, _ = self.forward(features1, features2)
        return self.classification(output)


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

class PolyLoss(nn.Module):
    def __init__(self, epsilon=1.0):
        super(PolyLoss, self).__init__()
        self.BCELoss = nn.BCELoss(reduction='none')
        self.epsilon = epsilon

    def forward(self, predicted, labels):
        # 计算 BCEWithLogitsLoss
        bce = self.BCELoss(predicted, labels)

        # 将预测值通过 sigmoid 函数转换为概率
        pt = predicted

        # 计算多项式项
        poly_term = self.epsilon * (1 - pt)

        # 返回综合损失
        return torch.mean(bce + poly_term)
