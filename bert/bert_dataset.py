import torch
import random
import pickle
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# 15% of the words with MASK token and predict them
    # 80% of the time, masked tokens are replaced with [MASK].
    # 10% of the time, masked tokens are replaced with random tokens.
    # 10% of the time, masked tokens remain unchanged.

class BERTDataset(Dataset):

    def __init__(self, tokenizer, path_to_dataset, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(path_to_dataset, 'rb') as f:
            self.corpus = pickle.load(f)

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):        
        
        if random.random() > 0.5:
            sentence_a = self.corpus[idx][0]
            sentence_b = self.corpus[idx][1]
            is_next = 0 
        else:
            sentence_a = self.corpus[idx][0]
            sentence_b = self.corpus[random.randint(0, len(self.corpus) - 1)][1]
            is_next = 1
        
        encoding = self._encode(sentence_a, sentence_b)

        return encoding

    def _encode(self, sentence_a, sentence_b):
        tokens_sent_a = self.tokenizer.tokenize(sentence_a)
        tokens_sent_b = self.tokenizer.tokenize(sentence_b)

        available_length = self.max_length - 3  # Space for [CLS], [SEP], and [SEP]

        # truncation
        if len(tokens_sent_a) + len(tokens_sent_b) > available_length:
            tokens_sent_a = tokens_sent_a[:available_length - len(tokens_sent_b)]
            tokens_sent_b = tokens_sent_b[:available_length - len(tokens_sent_a)]

        tokens_all = ['[CLS]'] + tokens_sent_a + ['[SEP]'] + tokens_sent_b + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens_all)
        attention_mask = [1] * len(token_ids)
        token_type_ids = [0] * (len(tokens_sent_a) + 2) + [1] * (len(tokens_sent_b) + 1)
        
        # padding
        padding_length = self.max_length - len(token_ids)
        token_ids += [0] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length

        return {
            "tokens_all": tokens_all, 
            "input_ids": torch.tensor(token_ids, dtype=torch.long), 
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long), 
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            }

        
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

train_data = BERTDataset(tokenizer=tokenizer, path_to_dataset="datasets/dataset.pkl")
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, pin_memory=True)
sample_data = next(iter(train_loader))
print(sample_data)