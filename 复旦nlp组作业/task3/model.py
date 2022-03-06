import torch.nn as nn
import torch
import torch.nn.functional as F

class CEloss(nn.Module):
    def __init__(self):
        super(CEloss, self).__init__()
        self.compute_loss = nn.CrossEntropyLoss()

    def forward(self, pre, label):
        return self.compute_loss(pre, label)

    @property
    def name(self):
        return 'Cross-entropy'


class PoolingAndPrediction(nn.Module):
    def __init__(self, dim, task_nums, dropout=0.5):
        super(PoolingAndPrediction, self).__init__()

        self.MLP = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Dropout(dropout),
                                 nn.Linear(dim, task_nums), nn.Softmax(dim=-1))

    def forward(self, v_p, v_h):
        concate_v = self.pooling(v_p, v_h)
        pre_proba = self.MLP(concate_v)
        return pre_proba


    def pooling(self, v_p, v_h):
        # 需要更改的地方
        v_p_mean, v_p_max = torch.mean(v_p, dim=1), torch.max(v_p, dim=1)[0]
        v_h_mean, v_h_max = torch.mean(v_h, dim=1), torch.max(v_h, dim=1)[0]

        return torch.cat([v_p_mean, v_p_max, v_h_mean, v_h_max], dim=-1)


class InputEncoding(nn.Module):
    def __init__(self, vocab_size, dim, OOV_size=0, word_embedding=None):
        super(InputEncoding, self).__init__()

        if word_embedding is None:
            self.wordembed = nn.Embedding(vocab_size+OOV_size, dim, padding_idx=0)
        else:
            assert isinstance(word_embedding, nn.Module)
            self.wordembed = word_embedding
            # todo: OOV + Glove vocabulary, please ESIM paper for more detail.(https://arxiv.org/abs/1609.06038)

        self.lstm_p = nn.LSTM(input_size=dim,hidden_size=dim,batch_first=True,bidirectional=True)
        self.lstm_h = nn.LSTM(input_size=dim,hidden_size=dim,batch_first=True,bidirectional=True)

    def forward(self, premise, hypothesis):

        px = self.wordembed(premise) # batch_size * seq_len * dim
        hx = self.wordembed(hypothesis)

        out1, (h1,c1) = self.lstm_p(px)
        out2, (h2,c2) = self.lstm_p(hx)

        # out1, out2: batch_size * seq_len * (dim*2)
        return out1, out2


class LocalInterface(nn.Module):
    def __init__(self, dim):
        super(LocalInterface, self).__init__()
        self.dim = dim

    def forward(self, px, hx, p_mask, h_mask):
        attention = self.locality_of_inference(px, hx)
        px_hat, hx_hat = self.local_inference_collected_over_sequences(px, hx, p_mask, h_mask, attention)
        m_p, m_h = self.enhancement_of_local_inference_information(px, hx, px_hat, hx_hat)

        return m_p, m_h


    def locality_of_inference(self, px, hx):
        '''
        :param px:  batch_size * px_len * dim
        :param hx:  bacth_size * hx_len * dim
        :return:  batch_size * px_len * hx_len
        '''
        hx = torch.transpose(hx, 2, 1)  # batch_size * dim * hx_len
        e_matrix = torch.matmul(px, hx)
        return e_matrix



    def local_inference_collected_over_sequences(self, px, hx, p_mask, h_mask, e_martix):
        # e_martix: batch_size * px_len * hx_len

        p_mask_attention = p_mask.float().masked_fill_(p_mask, float('-inf'))
        h_mask_attention = h_mask.float().masked_fill_(h_mask, float('-inf'))

        weight_b = F.softmax(e_martix + h_mask_attention.unsqueeze(1), dim=2) # numpy Broadcasting
        px_hat = torch.matmul(weight_b, hx)

        weight_a = F.softmax(e_martix + p_mask_attention.unsqueeze(2), dim=1) # numpy Broadcasting
        hx_hat = torch.matmul(weight_a.transpose(2,1), px)

        return px_hat, hx_hat



    def enhancement_of_local_inference_information(self, px, hx, px_hat, hx_hat):
        difference_p = px - px_hat
        difference_h = hx - hx_hat
        element_wise_px = px * px_hat
        element_wise_hx = hx * hx_hat

        return torch.cat([px,px_hat,difference_p,element_wise_px],dim=-1), \
               torch.cat([hx,hx_hat,difference_h,element_wise_hx],dim=-1)


class InferenceComposition(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.5):
        super(InferenceComposition, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.project_px = nn.Linear(input_dim, hidden_dim)
        self.project_hx = nn.Linear(input_dim, hidden_dim)
        self.lstm_px = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True)
        self.lstm_hx = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True)

    def forward(self, m_p, m_h):

        px = self.dropout(self.relu(self.project_px(m_p)))
        hx = self.dropout(self.relu(self.project_hx(m_h)))

        out_p, (hp, cp) = self.lstm_px(px)
        out_h, (hh, ch) = self.lstm_hx(hx)

        return out_p, out_h



class ESIM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, task_nums, dropout=0.5, OOV_size=0, word_embedding=None):
        super(ESIM, self).__init__()
        self.inputencoding = InputEncoding(vocab_size, hidden_dim, OOV_size=OOV_size, word_embedding=word_embedding)
        self.localinterface = LocalInterface(hidden_dim)
        self.composition = InferenceComposition(hidden_dim*2*4, hidden_dim, dropout=dropout)
        self.classify = PoolingAndPrediction(hidden_dim*2*4, task_nums, dropout=dropout)

    def forward(self, *input):
        p, h, p_mask, h_mask = input
        px, hx = self.inputencoding(p, h)
        mp, mh = self.localinterface(px, hx, p_mask, h_mask)
        vp, vh = self.composition(mp, mh)
        y = self.classify(vp, vh)
        return y


if __name__ == '__main__':
    # test code:
    model = ESIM(vocab_size=5, hidden_dim=10, task_nums=2, dropout=0.5, OOV_size=0, word_embedding=None)
    p = torch.randint(low=1, high=5, size=(2,6))
    h = torch.randint(low=1, high=5, size=(2,3))
    mask_p = torch.tensor([[0,0,0,1,1,1],[0,0,0,0,1,1]])
    mask_h = torch.tensor([[0,0,1],[0,1,1]])
    print(p, h)
    y = model(p, h, mask_p, mask_h)
    print(y)








