import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

class CustomModel(nn.Module):
    def __init__(self, bertmodel, labeltwo, labelthree):
        super().__init__()

        self.bert = bertmodel
        self.fc_dropout = nn.Dropout(0.35)
        self.labeltwo_out = nn.Sequential(nn.Linear(768, 256),nn.ReLU(),nn.Dropout(0.35))
        self.labelthree_out = nn.Sequential(nn.Linear(768, 256),nn.ReLU(),nn.Dropout(0.35))

        self.fc_two = nn.Sequential(nn.Linear(256, labeltwo),nn.Softmax(dim=-1))
        self.fc_three = nn.Sequential(nn.Linear(256, labelthree),nn.Softmax(dim=-1))

    def feature(self, inputs):
        outputs = self.bert(**inputs)
        last_hidden_states = outputs[1]
        return last_hidden_states

    def forward(self, inputs):
        feature = self.feature(inputs)
        feature= self.fc_dropout(feature)

        TwoOut = self.fc_two(self.labeltwo_out(feature))
        ThreeOut = self.fc_three(self.labelthree_out(feature))

        return TwoOut, ThreeOut


# class CustomModel(nn.Module):
#     def __init__(self, bertmodel, labeltwo, labelthree):
#         super().__init__()
#
#         self.bert = bertmodel
#         self.fc_dropout = nn.Dropout(0.35)
#         self.labeltwo_out = nn.Sequential(nn.Linear(768, 256),nn.ReLU(),nn.Dropout(0.35))
#         # self.labelthree_out = nn.Sequential(nn.Linear(768, 256),nn.ReLU(),nn.Dropout(0.35))
#
#         self.fc_two = nn.Sequential(nn.Linear(256, labeltwo),nn.Softmax(dim=-1))
#
#         self.labelthree_out = nn.Sequential(nn.Linear(768+256, labelthree),nn.Softmax(dim=-1))
#         # self.fc_three = nn.Sequential(nn.Linear(256, labelthree),nn.Softmax(dim=-1))
#
#     def feature(self, inputs):
#         outputs = self.bert(**inputs)
#         last_hidden_states = outputs[1]
#         return last_hidden_states
#
#     def forward(self, inputs):
#         feature = self.feature(inputs)
#         feature= self.fc_dropout(feature)
#
#         label_two = self.labeltwo_out(feature)
#         TwoOut = self.fc_two(label_two)
#         ThreeOut = self.labelthree_out(torch.cat([feature, label_two], dim=-1))
#
#         return TwoOut, ThreeOut

# import torch.nn.functional as F
# class CustomModel(nn.Module):
#     def __init__(self, bertmodel, labeltwo, labelthree):
#         super().__init__()
#
#         self.bert = bertmodel
#         self.fc_dropout = nn.Dropout(0.1)
#         self.convs = nn.ModuleList(
#             [nn.Conv2d(1, 64, (k, 768)) for k in [3,4]])
#
#         self.oneout = nn.Sequential(nn.Linear(64, labeltwo),nn.Softmax(-1))
#         self.twoout = nn.Sequential(nn.Linear(64*2, labelthree), nn.Softmax(-1))
#
#
#     def conv_and_pool(self, x, conv):
#         x = x.unsqueeze(1)
#         x = F.relu(conv(x)).squeeze(3)
#         x = F.max_pool1d(x, x.size(2)).squeeze(2)
#         return x
#
#
#     def feature(self, inputs):
#         outputs = self.bert(**inputs)
#         last_hidden_states = outputs[0]
#         return last_hidden_states
#
#     def forward(self, inputs):
#         feature = self.feature(inputs)
#         feature= self.fc_dropout(feature)
#
#         allout = [self.conv_and_pool(feature, conv) for conv in self.convs]
#
#         TwoOut = self.oneout(allout[0])
#         ThreeOut = self.twoout(torch.cat(allout, dim=-1))
#
#         return TwoOut, ThreeOut