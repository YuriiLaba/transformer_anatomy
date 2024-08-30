import torch
from bert_dataset import BERTDataset
from encoder import Encoder
from bert import BERT

import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self):
        model_ckpt = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.vocab_size = 30_000
        self.hidden_size = 768
        self.max_length = 128
        self.num_heads = 12
        self.device = 'cpu'

        self.model = BERT()
        self.criterion = torch.nn.NLLLoss(ignore_index=-100)

        self.train_data = BERTDataset(tokenizer=self.tokenizer, path_to_dataset="datasets/dataset.pkl", max_length=self.max_length)
        self.train_loader = DataLoader(self.train_data, batch_size=2, shuffle=True, pin_memory=True)
    
    def train(self, epoch):
        data_iter = tqdm.tqdm(enumerate(self.train_loader), desc="EP_%s:%d" % ("train", epoch), total=len(self.train_loader), bar_format="{l_bar}{r_bar}")

        for i, batch in data_iter:

            batch = {key: value.to(self.device) for key, value in batch.items()}

            next_sent_output, mask_lm_output = self.model.forward(batch)

            next_loss = self.criterion(next_sent_output, batch["nsp_label"])
            print(mask_lm_output.transpose(1, 2).shape)
            print(batch["mlm_labels"].shape)

            print(mask_lm_output.shape)
            
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), batch["mlm_labels"]) # why .transpose(1, 2)?

            loss = next_loss + mask_loss
            loss.backward()

            print(loss.item())

    

trainer = Trainer()

for i in range(3):
    trainer.train(i)

 
# train_loader = DataLoader(train_data, batch_size=2, shuffle=True, pin_memory=True)
# sample_data = next(iter(train_loader))
# print(sample_data)
# # print(next(iter(train_loader).shape))

# # encoder_layer = Encoder()
# # print(sample_data.keys())
# # print(encoder_layer(sample_data).shape)

# # 
# # print(model(sample_data)[0].shape)
# # print(model(sample_data)[1])