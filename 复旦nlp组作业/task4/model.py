import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_

'''
    Ref: https://arxiv.org/pdf/1603.01354.pdf  CNN + LSTM + CRF
         https://arxiv.org/abs/1603.01360  LSTM + CRF
         arXiv:1511.08308 CharCNN
         https://zhuanlan.zhihu.com/p/97829287 
'''

class CharCNN(nn.Module):
    def __init__(self, char_size, dim=30, max_length=10, WindowSize=3, FilterNums=30, dropout=0.5):
        super(CharCNN, self).__init__()

        padding = WindowSize - 1
        feature_map_x = ( max_length + 2 * padding - WindowSize ) + 1   # (n + 2p - f) / stride + 1
        feature_map_y = 1 * FilterNums

        self.CharEmbed = nn.Embedding(char_size, dim, padding_idx=0)

        # CNN-input: batch_size * channels * row * col, out-put: batch_size * channels * feature_map_row * feature_map_col
        # Kernel_size: (row, col)
        # padding: (row_padding, col_padding) row_padding is the padding numbers, both up and down direction.
        self.cnn = nn.Conv2d(in_channels=1, out_channels=FilterNums, kernel_size=(WindowSize, dim),
                             stride=(1, 1), padding=(padding,0))

        self.maxpool = nn.MaxPool2d(kernel_size=(feature_map_x, 1),stride=1) # extract the max_feature within each Feature
        self.dropout = nn.Dropout(dropout)


    def forward(self, char_x):
        bacth_size, sentence_lens, char_nums = char_x.shape
        char_x = char_x.view(bacth_size * sentence_lens, char_nums)
        Xs_embed = self.CharEmbed(char_x).unsqueeze(1) # sentences_lens * 1 * char_nums * dim
        Xs_embed = self.dropout(Xs_embed)
        feature_map = torch.transpose(self.cnn(Xs_embed).squeeze(-1), 2, 1) # sentences_lens * feature_map_x * feature_map_y
        word_embed = self.maxpool(feature_map).squeeze()
        return word_embed.view(bacth_size,sentence_lens,-1)  # batch_size * sentence_lens * feature_map_y(FilterNums)



class BiLSTM(nn.Module):
    def __init__(self, word_size, tag_nums, word_dim=100, input_dim=130, hidden_state=100, num_layers=1,dropout=0.5):
        super(BiLSTM, self).__init__()

        self.WordEmbed = nn.Embedding(word_size, word_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=input_dim , hidden_size=hidden_state,
                            batch_first=True, num_layers=num_layers, dropout=0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Linear(hidden_state*2,tag_nums)

    def forward(self, word_x, char_embedding):
        word_x = self.WordEmbed(word_x)
        concate_x = torch.cat([word_x, char_embedding], dim=-1)
        concate_x = self.dropout(concate_x)
        output, (h, c) = self.lstm(concate_x)
        y = self.project(self.dropout(output))
        return  y # return the emission matrix. Note: need logits score rather than possibility.



class CRF(nn.Module):
    def __init__(self, tag_nums, start_idx, end_idx):
        super(CRF, self).__init__()

        self.tag_nums = tag_nums
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.TransitionMatrix = nn.Parameter(torch.rand(tag_nums, tag_nums)) # T_i,j: possibility of tags i -> j
        self.reset_parm()


    def reset_parm(self):
        kaiming_normal_(self.TransitionMatrix)
        self.TransitionMatrix.detach()[:, self.start_idx] = torch.tensor([-10000]*self.tag_nums)
        self.TransitionMatrix.detach()[self.end_idx, : ] = torch.tensor([-10000]*self.tag_nums)


    def compute_y_score(self, emission, y, mask):
        '''
        :param emission: batch_size * seq_len * tag_nums
        :param y:  batch_size * seq_len
        :param mask: batch_size * seq_len
        :return: y_score: batch_size
        '''
        batch_size, seq_len, tag_nums = emission.shape
        score = torch.zeros(size=(batch_size,))
        y = torch.cat([torch.full(size=(batch_size,1),fill_value=self.start_idx), y], dim=-1)  # add the <start> label
        for i in range(seq_len):
            mask_i = mask[:, i]
            pre_label = y[:, i]
            label = y[:, i+1]
            emi = emission[:,i,:]
            p_ij = torch.tensor([ emi[idx,la] for idx, la in enumerate(label)]) * mask_i
            t_ij = torch.tensor([ self.TransitionMatrix[pre_la, la] for pre_la, la in zip(pre_label, label)]) * mask_i
            score += (p_ij + t_ij)

        # compute the score of path from the last_tags to <END> in sequence
        score += torch.tensor([ self.TransitionMatrix[pre_la, self.end_idx] for idx, pre_la in enumerate(label)])
        return score


    def compute_all_score(self, emission, mask):
        '''
        :param emission: bacth_size * seq_len * tag_nums
        :param mask:  batch_size * seq_len
        :return:
        '''
        batch_size, seq_len, tag_nums = emission.shape
        log_alpha = torch.full(size=(batch_size, self.tag_nums, 1),fill_value=-10000)  # batch_size * tag_nums * 1
        log_alpha[:,self.start_idx,:] = 0
        for i in range(seq_len):
            mask_i = mask[:,i].view(batch_size, 1, 1) # batch_size * 1
            emi = emission[:,i,:].unsqueeze(1) # batch_size * 1 * tag_nums

            trans = self.TransitionMatrix.unsqueeze(0)  # 1 * tag_nums * tag_nums
            log_alpha_tmp = log_alpha + trans + emi  # batch_size * tag_nums * tag_nums
            log_alpha = torch.logsumexp(log_alpha_tmp,dim=1).unsqueeze(-1) * mask_i + log_alpha * (1-mask_i)

        # the path of last_tags -> END
        end_score_path = self.TransitionMatrix[:,self.end_idx].view(1, tag_nums, 1)  # 1 * tag_nums * 1
        log_alpha += end_score_path # batch_size * tag_nums * 1
        total_score = torch.logsumexp(log_alpha, dim=1)

        return total_score


    def neg_log_likelihood(self, emission, y, mask):
        score_y = self.compute_y_score(emission, y, mask)
        log_all_y = self.compute_all_score(emission, mask)
        loss = (score_y - log_all_y).mean()
        return - loss


    def predict(self, emission, mask):
        '''
        :param emission: batch_size * seq_len * tag_nums
        :param mask: batch_size * seq_len
        :return:
        '''
        batch_size, seq_len, tag_nums = emission.shape
        score = torch.full(size=(batch_size,tag_nums,1), fill_value=-10000)  # batch_size * tag_nums * 1
        score[:,self.start_idx,:] = 0  # alpha in START_TAG is 1. log(1) = 0
        pointers = []
        for i in range(seq_len):
            mask_i = mask[:,i].view(batch_size,1,1)  # batch_size
            emi = emission[:,i,:].unsqueeze(1) # batch_size * 1 * tag_nums
            trans = self.TransitionMatrix.unsqueeze(0) # 1 * tag_nums * tag_nums
            log_alpha_tmp = score + trans + emi  # batch_size * tag_nums * tag_nums

            # max_index: batch_size * tag_nums | max_values: batch_size * tag_nums
            max_values, max_index = torch.max(log_alpha_tmp, dim=1)
            pointers.append(max_index)

            max_values = max_values.unsqueeze(-1)
            score = max_values * mask_i + score * (1-mask_i)

        pointers = torch.stack(pointers, 1) # # (batch_size, seq_len, tag_nums)
        # last_tags -> END tags
        end_path_score = self.TransitionMatrix[:,self.end_idx].view(1, tag_nums, 1)
        score += end_path_score
        max_values, max_index = torch.max(score, dim=1) # 最后max_index是路径到达 END Tags的前一个时刻最大tags index

        best_path=[]
        for i in range(batch_size):
            len_seq_i = mask[i,:].sum()
            pointer_i = reversed(pointers[i,:len_seq_i,:]) # 从后往前进行回溯寻最优的路径

            best_idx = max_index[i].item()
            seq_i_best_path = [ best_idx ]
            for j in range(len_seq_i):
                best_idx = pointer_i[j,best_idx].item()
                seq_i_best_path = [best_idx] + seq_i_best_path

            seq_i_best_path = seq_i_best_path[1:] # remove the <START> label
            best_path.append(seq_i_best_path)

        return best_path




class CnnLstmCrf(nn.Module):
    def __init__(self, char_size, word_size, tag_nums, start_idx, end_idx, char_dim=30, char_max_length=10, cnn_WindowSize=3, cnn_FilterNums=30,
                    word_dim=100, hidden_state=100, num_layers=1,
                    dropout=0.5):

        super(CnnLstmCrf, self).__init__()
        lstm_input_dim = word_dim + cnn_FilterNums
        self.CharCnn = CharCNN(char_size=char_size,dim=char_dim,max_length=char_max_length,
                               WindowSize=cnn_WindowSize,FilterNums=cnn_FilterNums,dropout=dropout)
        self.bilstm = BiLSTM(word_size=word_size, tag_nums=tag_nums, word_dim=word_dim,
                             input_dim=lstm_input_dim, hidden_state=hidden_state, num_layers=num_layers,dropout=dropout)
        self.crf = CRF(tag_nums, start_idx, end_idx)

    def compute_loss(self, word_x, char_x, y, mask):
        char_embed = self.CharCnn(char_x)
        emission = self.bilstm(word_x=word_x, char_embedding=char_embed)
        loss = self.crf.neg_log_likelihood(emission, y, mask)
        return loss


    def predict(self, word_x, char_x, mask):
        char_embed = self.CharCnn(char_x)
        emission = self.bilstm(word_x=word_x, char_embedding=char_embed)
        best_path = self.crf.predict(emission, mask)

        return best_path


if __name__ == '__main__':
    # charCNN = CharCNN(5, dim=10, max_length=7, WindowSize=3, FilterNums=4)
    # xs = torch.randint(low=0,high=5,size=(10,7)) # sentences_lens * char_nums
    # print(xs.shape)
    # print(charCNN(xs).shape)

    cfr = CRF(tag_nums=5, start_idx=0, end_idx=4)
    emission = torch.rand(size=(2,3,5))
    mask = torch.tensor([[1,1,0],[1,0,0]])
    y = torch.randint(low=1,high=4,size=(2,3))
    print(y)
    print(mask)
    cfr.compute_all_score(emission, mask)
    cfr.compute_y_score(emission, y, mask)
    cfr.neg_log_likelihood(emission, y, mask)
    path = cfr.predict(emission,mask)
    print(path)


    model = CnnLstmCrf(char_size=5, word_size=5, tag_nums=5, start_idx=0, end_idx=4, char_max_length=5,cnn_FilterNums=10)
    char_x = torch.randint(low=0,high=5,size=(2,5,5))
    word_x = torch.randint(low=0,high=5,size=(2,5))
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    y = torch.randint(low=1, high=4, size=(2, 5))

    print(model.compute_loss(char_x, word_x, y, mask))





