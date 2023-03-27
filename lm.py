import torch
import torch.nn as nn

class LM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_dim, num_layers, 
            dropout_rate):
        super(LM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(input_size = embed_size, 
                           hidden_size = hidden_dim, 
                           num_layers = num_layers, 
                           dropout=dropout_rate)

        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim = 2)



    def forward(self, x, logits=False):
        x = x.long()
        _ = self.embedding(x)
        _ = _.permute(1, 0, 2)
        h_all, _ = self.rnn(_)
        h_all = h_all.permute(1, 0, 2)
        _ = self.output_layer(h_all)
        if logits:
            return _
        else:
            return self.log_softmax(_)


    def sample(self, x):
        log_prob = self.forward(x)
        prob = torch.exp(log_prob)[:, -1, :]
        prob[:,1] = 0
        prob = prob / prob.sum()
        
        return torch.multinomial(prob, 1)
