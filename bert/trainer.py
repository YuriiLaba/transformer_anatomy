from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from bert_dataset import BERTDataset

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

train_data = BERTDataset(tokenizer=tokenizer, path_to_dataset="datasets/dataset.pkl")
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, pin_memory=True)
sample_data = next(iter(train_loader))
print(sample_data)