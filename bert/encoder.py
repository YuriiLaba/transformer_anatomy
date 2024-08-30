from torch import nn
from embedding_layer import Embeddings
from encoder_block import EncoderBlock

class Encoder(nn.Module):
    def __init__(self, num_hidden_layers=12):
        super().__init__()
        self.embeddings = Embeddings()
        self.encoder_blocks = nn.ModuleList([EncoderBlock() for _ in range(num_hidden_layers)])
        
    def forward(self, x):
        mask = (x["input_ids"] > 0).unsqueeze(1).repeat(1, x["input_ids"].size(1), 1)

        x = self.embeddings(x["input_ids"], x["token_type_ids"]) 
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask) 
        return x