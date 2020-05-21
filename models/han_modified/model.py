import torch
import torch.nn as nn

from models.han_modified.sent_level_rnn import SentLevelRNN
from models.han_modified.word_level_rnn import WordLevelRNN


class HAN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.word_attention_rnn = WordLevelRNN(config)
        self.sentence_attention_rnn = SentLevelRNN(config)
        self.vae_struct = config.vae_struct

    def forward(self, x,  **kwargs):

        x = x.permute(1, 2, 0) # Expected : # sentences, # words, batch size
        num_sentences = x.size(0)
        word_attentions = []
        word_vecs = []
        for i in range(num_sentences):

            word_attn = self.word_attention_rnn(x[i, :, :])[0] 
            word_attentions.append(word_attn)
            if self.vae_struct:
                word_vecs.append(self.word_attention_rnn(x[i, :, :])[1])
        
        word_attentions = torch.cat(word_attentions, dim=0)

        if self.vae_struct:
            word_vecs = torch.cat(word_vecs, dim=0)
            output = self.sentence_attention_rnn(word_attentions, word_vecs)
        else:
            output = self.sentence_attention_rnn(word_attentions)

        return output

if __name__ == "__main__":
    pass