import torch
import random
import pickle
from torch.utils.data import Dataset


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
            nsp_label = 0 
        else:
            sentence_a = self.corpus[idx][0]
            sentence_b = self.corpus[random.randint(0, len(self.corpus) - 1)][1]
            nsp_label = 1
        
        encoding = self._encode(sentence_a, sentence_b)
        encoding["nsp_label"] = nsp_label

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
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens_all)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * (len(tokens_sent_a) + 2) + [1] * (len(tokens_sent_b) + 1)
        
        # padding
        padding_length = self.max_length - len(input_ids)
        input_ids += [0] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length

        input_ids, mlm_labels = self._add_mask_tokens(input_ids, attention_mask)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long), 
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long), 
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),

            "tokens_all": tokens_all, 
            "mlm_labels": torch.tensor(mlm_labels, dtype=torch.long), 
            }

    def _add_mask_tokens(self, input_ids, attention_mask):
        mlm_labels = input_ids.copy()
        mlm_labels = [-100 if mask == 0 else label for label, mask in zip(mlm_labels, attention_mask)]
        
        for i in range(len(input_ids)):
            # Skip special tokens [CLS], [SEP], and padding tokens (with id 0)
            if input_ids[i] in {self.tokenizer.convert_tokens_to_ids('[SEP]'),  self.tokenizer.convert_tokens_to_ids('[CLS]'),  0}:
                continue

            if random.random() < 0.15:
                prob = random.random()
                if prob < 0.8:
                    input_ids[i] = self.tokenizer.convert_tokens_to_ids('[MASK]')
                elif prob < 0.9:
                    input_ids[i] = random.randint(0, self.tokenizer.vocab_size - 1)
                # 10% chance of leaving the token unchanged (no action needed)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(mlm_labels, dtype=torch.long)