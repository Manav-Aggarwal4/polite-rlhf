import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoConfig,
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom MSE loss for single-label regression.
        """
        labels = inputs.pop("labels")         
        outputs = model(**inputs)
        logits = outputs.logits.view(-1)      

        loss_fn = nn.MSELoss()
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


class RegressionDataset(Dataset):
    def __init__(self, texts, scores, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.scores = scores
        self.max_length = max_length
        self.encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.scores[idx], dtype=torch.float)
        return item


def main():
    df = pd.read_csv("./data/MIT_dataset.csv")
    texts = df["Text"].astype(str).tolist()
    scores = df["Score"].tolist()

    # 2. Prepare BERT for single_label_regression
    from transformers import AutoConfig, BertForSequenceClassification, BertTokenizer

    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 1
    config.problem_type = "single_label_regression"
    model = BertForSequenceClassification.from_pretrained(model_name, config=config)

    # 3. Build Dataset
    dataset = RegressionDataset(texts, scores, tokenizer)

    # Optional train/val split
    val_ratio = 0.1
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 4. TrainingArguments
    training_args = TrainingArguments(
        output_dir="./bert-regression-checkpoints",
        overwrite_output_dir=True,
        eval_strategy="epoch",  
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=100,
        logging_dir="./logs",
    )

    # 5. Instantiate our custom RegressionTrainer
    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    trainer.save_model("./bert-regression-model")
    print("Finished training regression model. Saved at ./bert-regression-model")

if __name__ == "__main__":
    main()
