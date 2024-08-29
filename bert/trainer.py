from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from bert_dataset import BERTDataset
from bert_embedding import BERTEmbedding
from encoder import AttentionHead

import torch

vocab_size = 30_000
hidden_size = 768
max_length = 128

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

train_data = BERTDataset(tokenizer=tokenizer, path_to_dataset="datasets/dataset.pkl", max_length=max_length)
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, pin_memory=True)
sample_data = next(iter(train_loader))

mask = (sample_data["input_ids"] > 0).unsqueeze(1).repeat(1, sample_data["input_ids"].size(1), 1)

bert_embedding = BERTEmbedding(vocab_size, hidden_size, max_length)
emb = bert_embedding(sample_data["input_ids"], sample_data["token_type_ids"])

# print(sample_data)
# print(emb)
# print(emb.shape)

att_head = AttentionHead(hidden_size, hidden_size)
print(att_head(emb, mask))