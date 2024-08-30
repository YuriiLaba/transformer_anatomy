from torch import nn
from multi_head_attention import MultiHeadAttention, FeedForward

class EncoderBlock(nn.Module): 
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(hidden_size, num_heads) 
        self.feed_forward = FeedForward(hidden_size)

    def forward(self, x, mask):
        x = self.multi_head_attention(x, mask)
        x = self.feed_forward(x)
        return x