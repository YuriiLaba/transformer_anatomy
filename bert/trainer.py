import torch
from bert_dataset import BERTDataset
from encoder import Encoder
from bert import BERT
import neptune

import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader



class Trainer:
    def __init__(self):
        model_ckpt = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.vocab_size = 30_522
        self.hidden_size = 768
        self.max_length = 128
        self.num_heads = 12
        self.batch_size = 32
        self.device = 'cpu'
        
        self.run = neptune.init_run(
            project="laba/bert-training",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwM2RmMzQyOS01ZWJkLTRmOGMtYTIxMy1mYmE4NzVjNDJhZWYifQ==",
        ) 

        self.model = BERT()
        self.criterion = torch.nn.NLLLoss(ignore_index=-100)

        self.train_data = BERTDataset(tokenizer=self.tokenizer, path_to_dataset="datasets/train_dataset.pkl", max_length=self.max_length)
        self.eval_data = BERTDataset(tokenizer=self.tokenizer, path_to_dataset="datasets/val_dataset.pkl", max_length=self.max_length)

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        self.eval_loader = DataLoader(self.eval_data, batch_size=self.batch_size, shuffle=False, pin_memory=True)
    
    def stop_logging(self):
        self.run.stop()

    def _train_step(self, batch):
        next_sent_output, mask_lm_output = self.model.forward(batch)

        next_sent_loss = self.criterion(next_sent_output, batch["nsp_label"])
        mask_loss = self.criterion(mask_lm_output.transpose(1, 2), batch["mlm_labels"]) # why .transpose(1, 2)?
        return next_sent_loss, mask_loss
    
    def run_epoch(self, epoch, is_training=True):
        loader = self.train_loader if is_training else self.eval_loader
        mode = "train" if is_training else "validate"
        
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        # total_loss = 0.0
        data_iter = tqdm.tqdm(enumerate(loader), desc=f"EP_{mode}:{epoch}", total=len(loader), bar_format="{l_bar}{r_bar}")

        with torch.set_grad_enabled(is_training):
            for idx, batch in data_iter:
                batch = {key: value.to(self.device) for key, value in batch.items()}

                next_sent_loss, mask_loss = self._train_step(batch)
                loss = next_sent_loss + mask_loss

                self.run[f"{mode}/batch_loss"].log(loss.item())

                if is_training:
                    loss.backward()
    

if __name__ == '__main__':
    num_epoch = 5

    trainer = Trainer()

    for i in range(num_epoch):
        trainer.run_epoch(i)
        trainer.run_epoch(i, is_training=False)
    
    trainer.stop_logging()

 
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