import torch
import torch.nn as nn

from models.han.sent_level_rnn import SentLevelRNN
from models.han.word_level_rnn import WordLevelRNN


class HAN(nn.Module):

    def __init__(self, config):
        super().__init__()
        # word embedding
        self.mode = config.mode
        self.words_num = config.words_num
        self.words_dim = config.words_dim
        dataset = config.dataset
        if self.mode == 'rand':
            rand_embed_init = torch.Tensor(self.words_num, self.words_dim).uniform(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif self.mode == 'static':
            self.embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
        elif self.mode == 'non-static':
            self.embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
        else:
            print("Unsupported order")
            exit()
        self.word_attention_rnn = WordLevelRNN(config)
        self.sentence_attention_rnn = SentLevelRNN(config)
        self.vae_struct = config.vae_struct

    def forward(self, x,  **kwargs):

        # x Expected : # batch size, sentences, words, 
        num_sentences = x.size(1)
        word_attentions = []
        word_vecs = []
        for i in range(num_sentences):
            input = self.embed(x[:,i,:])
            word_attn = self.word_attention_rnn(input)
            if not self.vae_struct:
                word_attentions.append(word_attn[0])
            else:
                word_attentions.append(word_attn[0])
                word_vecs.append(word_attn[1])
        
        word_attentions = torch.cat(word_attentions, dim=0)

        if self.vae_struct:
            word_vecs = torch.cat(word_vecs, dim=0)
            output = self.sentence_attention_rnn(word_attentions, word_vecs)
        else:
            output = self.sentence_attention_rnn(word_attentions)

        return output

if __name__ == "__main__":
    pass