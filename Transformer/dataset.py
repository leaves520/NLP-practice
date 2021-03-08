import torch
from torch.utils.data import DataLoader,Dataset



class Vocabulary:
    def __init__(self):
        self.freeze = False
        self.word2index = {}
        self.index2word = {}
        self.dict_size = 0

    @property
    def dict_size_(self):
        return self.dict_size

    def index(self, word):
        if not self.freeze:
            if word in self.word2index:
                return self.word2index[word]
            else:
                self.word2index[word] = self.dict_size
                self.index2word[self.dict_size] = word
                self.dict_size += 1
                return self.word2index[word]
        else:
            if word in self.word2index:
                return self.word2index[word]
            else:
                raise ValueError('Word is not in this dictionary ! ')

    def word(self, index):
        assert index < self.dict_size and index >= 0, f"index is out of the dict_size({self.dict_size})"
        return self.index2word[index]


    def freeze_(self):
        self.freeze = True

class MyDataset(Dataset):
    def __init__(self,text,vocab):
        super(MyDataset, self).__init__()
        self.vocab = vocab
        self.text = text
        self.enc_input,self.dec_input,self.dec_output = self.get_index(text)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        return self.enc_input[item],self.dec_input[item],self.dec_output[item]


    def get_index(self,text):
        enc_inputs,dec_inputs,dec_outputs = [],[],[]
        for sentences in text:
            enc_input = [[ self.vocab.index(w) for w in sentences[0].split(' ')]]
            dec_input = [[ self.vocab.index(w) for w in sentences[1].split(' ')]]
            dec_output = [[ self.vocab.index(w) for w in sentences[2].split(' ')]]

            enc_inputs.extend(enc_input)
            dec_inputs.extend(dec_input)
            dec_outputs.extend(dec_output)

        return torch.LongTensor(enc_inputs),torch.LongTensor(dec_inputs),torch.LongTensor(dec_outputs)




# only for test

# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps
# dec_input is the shifted right dec_output
sentences = [
        # enc_input           dec_input         dec_output
        ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
        ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]


# create the dictionary
vocabulary = Vocabulary()
vocabulary.index('P')  #Padding should be zero
vocabulary.index('S')
vocabulary.index('E')

for sen in sentences:
    for s in sen:
        for word in s.split(' '):
            vocabulary.index(word)

vocabulary.freeze_()

dataset = MyDataset(text=sentences,vocab=vocabulary)
dataloader = DataLoader(dataset,batch_size=2,drop_last=False)

