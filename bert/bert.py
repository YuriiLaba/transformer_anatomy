import torch
from encoder import Encoder
from pre_training_strategies import NextSentencePrediction, MaskedLanguageModel

class BERT(torch.nn.Module):
    def __init__(self, hidden_size=768, vocab_size=30_000):
        super().__init__()
        self.encoder = Encoder()
        self.next_sentence = NextSentencePrediction(hidden_size)
        self.mask_lm = MaskedLanguageModel(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.encoder(x)
        return self.next_sentence(x), self.mask_lm(x)