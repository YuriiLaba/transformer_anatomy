from torch import nn
from multi_head_attention import MultiHeadAttention, FeedForward

class EncoderBlock(nn.Module): 
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(hidden_size) 
        self.layer_norm_2 = nn.LayerNorm(hidden_size)

        self.multi_head_attention = MultiHeadAttention(hidden_size, num_heads).to("cuda") 
        self.feed_forward = FeedForward(hidden_size).to("cuda")

    def forward(self, x, mask):
        
        mask_ = (x.sum(dim=-1) != 0).unsqueeze(-1)

        hidden_state = self.layer_norm_1(x) * mask_

        x = x + self.multi_head_attention(hidden_state, mask)

        x = self.feed_forward(self.layer_norm_2(x * mask_))
        return x