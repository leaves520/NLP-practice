# Ref: https://zhuanlan.zhihu.com/p/97858739
# Ref: pad_packed and pack_padded https://zhuanlan.zhihu.com/p/374876781

from data_set import label_dict, id2label
import torch
from torch import nn
from torch.nn import init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


# pack_padded...压缩, pad_packed...解压还原


def get_mask(length):
    # 求出一个batch中最大的输入长度
    max_len = int(max(length))
    mask = torch.Tensor()
    # length = length.numpy()
    # 与每个序列等长度的全1向量连接长度为最大序列长度-当前序列长度的全0向量。
    for len_ in length:
        mask = torch.cat((mask, torch.Tensor([[1] * len_ + [0] * (max_len - len_)])), dim=0)
    return mask


class LstmCrf(nn.Module):

    def __init__(self, input_dim, hidden_dim, n_class, vocab_size, gensim_embedding=None):
        super(LstmCrf, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_class = n_class
        self.vocab_size = vocab_size

        # The emission matrix got from the code below
        # padding_idx = 0 : denote the idx=0 -> zeros vector
        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=0)
        if gensim_embedding is not None:  # 使用word2vec(gensim)训练好的权重
            self.embedding.weight.data.copy_(torch.from_numpy(gensim_embedding))

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=2,
                            bidirectional=True, batch_first=True)
        self.linear = nn.Linear(in_features=2 * hidden_dim, out_features=n_class)

        # The transition matrix is defined below
        self.transition_martix = nn.Parameter(torch.rand((n_class, n_class)))  # matrix[i][j]: From i to j (state)
        self.reset_parameters()
        self.softmax = nn.Softmax(dim=-1)

    def reset_parameters(self):
        init.normal_(self.transition_martix)
        # initialize start_tag, end_tag probability
        # 这样设置是为了保证其他标签不会转向起始标签，同样结束标签也不会转向其他标签
        # matrix[i][j]: From i to j (state)
        self.transition_martix.detach()[:, label_dict["<START>"]] = - 10000
        self.transition_martix.detach()[label_dict["<END>"], :] = - 10000

    def forward(self, input_data, input_len):
        '''
        :param input_data: [batch_size, len]
        :param input_len: list of seq_len
        :return: emission matrix. shape:[batch, len, embed_dim]
        '''
        embed_x = self.embedding(input_data)
        packed_x = pack_padded_sequence(input=embed_x, lengths=input_len, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.lstm(packed_x)
        output, _ = pad_packed_sequence(packed_output)
        output = self.linear(output)

        output = output.transpose(0, 1)
        return output

    def forward_alpha(self, emission, mask):
        '''
          alpha_score(t) = alpha_score(t-1) + transition_score + emit_score
        :param emission: [batch, len, embed_dim]
        :param mask:
        :return:
        '''

        batch_size, seq_len = mask.size()
        # log a_0
        log_alpha_init = torch.full((batch_size, self.n_class), fill_value=-10000)
        # i=0时，label为start的概率为1，其余为0，取log后的结果
        log_alpha_init[:, 0] = 0
        # alpha, [batch_size, n_class]
        log_alpha = log_alpha_init
        for w in range(seq_len):
            # 取出当前时刻的mask,每个batch中的元素，对应一个n_class*n_class的矩阵，矩阵所有元素相同
            mask_t = mask[:, w].unsqueeze(-1)  # batch_size, 1
            currnet = emission[:, w, :]  # batch_size, n_class
            log_alpha_matrix = log_alpha.unsqueeze(2).expand(-1, -1, self.n_class)  # 增列, batch_size, n_class, n_class
            currnet_matrix = currnet.unsqueeze(1).expand(-1, self.n_class, -1)  # 增行 batch_size, n_class, n_class
            # self.transition_martix[i][j] 表示从i到j状态的转移概率
            log_M_matrix = currnet_matrix + self.transition_martix  # 节点分数(e)+边分数(t), 当前时刻的M_i(x) = exp(Transition+Emission)
            add_matrix = log_alpha_matrix + log_M_matrix  # wise add (batch, n_class, n_class)
            # mask_t的作用，如果该位置后都为padding,位置后的值都一样(使用上一状态的alpha值)
            log_alpha = torch.logsumexp(add_matrix, dim=1) * mask_t + log_alpha * (1 - mask_t)

        # n -> END
        alpha = log_alpha + self.transition_martix[:, label_dict["<END>"]].unsqueeze(0)
        # alpha: [batch_size,n_class], total_score: [batch_size,1]
        total_score = torch.logsumexp(alpha, dim=1)  #
        return total_score

    def get_sentence_score(self, emission, labels, mask):
        batch_size, seq_len, n_class = emission.size()
        # 增加<start>label 这一列
        labels = torch.cat([labels.new_full((batch_size, 1), fill_value=label_dict["<START>"]), labels], 1)
        scores = emission.new_zeros(batch_size)

        # score(y) = sum_(i=1->n)(e_i) + sum_(i=2->n)(Transition[i-1,i])  # y的得分为: 路径的节点的得分 + 所有边的得分
        # from M_1 to M_n
        for i in range(seq_len):
            mask_i = mask[:, i]
            current = emission[:, i, :]
            # 取出每个词对应的score，i+1表明跳过start
            emit_score = torch.cat(
                [each_score[next_label].unsqueeze(-1) for each_score, next_label in zip(current, labels[:, i + 1])],
                dim=0)
            transition_score = torch.stack(
                [self.transition_martix[labels[b, i], labels[b, i + 1]] for b in range(batch_size)])
            scores += (emit_score + transition_score) * mask_i

        transition_to_end = torch.stack(
            [self.transition_martix[label[mask[b, :].sum().long()], label_dict["<END>"]] for b, label in
             enumerate(labels)])
        scores += transition_to_end
        return scores

    def get_loss(self, emission, labels, mask):
        # loss = -log_p(y|x)
        # log_p(y|x) = score(y) - log(sum_y[exp(score(y))]) : max == min -1*()
        log_z = self.forward_alpha(emission, mask)
        log_alpha_n = self.get_sentence_score(emission, labels, mask)
        loss = (-1 * (log_alpha_n - log_z)).sum()
        return loss

    def best_path(self, emission, mask):
        '''
        viterbi_decode, dp算法
           Predict by trained model
           alpha_score(t) = alpha_score(t-1) + transition_score + emit_score
        '''
        batch_size, seq_len = mask.size()

        log_alpha_init = torch.full((batch_size, self.n_class), fill_value=-10000)
        log_alpha_init[:, 0] = 0
        log_alpha = log_alpha_init

        pointers = []
        for w in range(seq_len):
            mask_t = mask[:, w].unsqueeze(-1)
            currnet = emission[:, w, :]
            log_alpha_matrix = log_alpha.unsqueeze(2).expand(-1, -1, self.n_class)
            trans = log_alpha_matrix + self.transition_martix
            max_trans, pointer = torch.max(trans, dim=1)  # 选取上一时刻的y_i-1使得到当前时刻的某个y_i的路径分数最大
            pointers.append(pointer)  # 添加路径，此时pointer为上一时刻的label(state)
            cur_log_alpha = currnet + max_trans
            log_alpha = log_alpha * (1 - mask_t) + cur_log_alpha * mask_t  # 根据mask判断是否更新(PAD位置处理)，得到当前时刻的log_alpha

        pointers = torch.stack(pointers, dim=1)  # 时间t维度去stack
        log_alpha = log_alpha + self.transition_martix[:, label_dict['<END>']]  # n -> END
        best_log_alpha, best_label = torch.max(log_alpha, dim=1)  # 找到n->END的最优路径, [batch_size, 1] # 找到n时刻最好的状态得分

        best_path = []
        # 从后往前,回溯找dp记录的最优路径
        for i in range(batch_size):  # 依次处理batch内每一个数据
            seq_len_i = int(mask[i].sum())  # 当前数据的路径长度
            pointers_i = pointers[i, :seq_len_i]  # [seq_len_i, n_class]

            # 从后往前回溯
            best_label_i = best_label[i]  # 当前数据best label
            best_path_i = [best_label_i]
            for j in range(seq_len_i):
                index = seq_len_i - j - 1
                best_label_i = pointers_i[index][best_label_i]  # pointers_i[t]表示t-1时刻到t时刻的最优状态(即箭头的起点)
                best_path_i = [best_label_i] + best_path_i

            # 除去时刻1之前的路径(即去掉<start>标识符)
            best_path_i = best_path_i[1:]
            best_path.append([id2label[path.item()] for path in best_path_i])  # 将预测的label保存并返回

        return best_path


if __name__ == '__main__':
    emission = torch.rand(size=(1, 4, 6))
    mask = get_mask([4])
    t = LstmCrf(input_dim=2, hidden_dim=2, n_class=6, vocab_size=2, embed_dim=2)
    t.best_path(emission, mask)
