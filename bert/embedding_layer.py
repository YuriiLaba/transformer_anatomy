import torch
from torch import nn

class Embeddings(torch.nn.Module):
    def __init__(self, vocab_size=30_000, hidden_size=768, max_length=128):

        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_length = max_length  

        self.token_embeddings = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(self.max_length, self.hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(3, self.hidden_size, padding_idx=0)

        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-12) #  normalizes across the features of each individual sample
        self.dropout = nn.Dropout()
    
    def forward(self, input_ids, token_type_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long)
        
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

        