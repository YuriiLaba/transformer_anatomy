from bert_dataset import BERTDataset
from encoder import Encoder

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

vocab_size = 30_000
hidden_size = 768
max_length = 128
num_heads=12

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

train_data = BERTDataset(tokenizer=tokenizer, path_to_dataset="datasets/dataset.pkl", max_length=max_length)
train_loader = DataLoader(train_data, batch_size=2, shuffle=True, pin_memory=True)
sample_data = next(iter(train_loader))
# print(sample_data)
# print(next(iter(train_loader).shape))

encoder_layer = Encoder()
# print(sample_data.keys())

print(encoder_layer(sample_data).shape)