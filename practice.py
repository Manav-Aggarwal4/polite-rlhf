from convokit import Corpus, download
import pandas as pd


corpus = Corpus(filename=download("wikipedia-politeness-corpus"))


polite_texts = []

for utt in corpus.iter_utterances():
    if 'Binary' in utt.meta:
        # Keep only utterances labeled "polite"
        if utt.meta['Binary'] == 1:
            polite_texts.append(utt.text)


import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments


# Load tokenizer and model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a default pad token

model = GPT2LMHeadModel.from_pretrained(model_name)

class PoliteDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=64):
        self.texts = texts
        self.tokenizer = tokenizer
        self.encodings = self.tokenizer(
            self.texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors="pt"
        )
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        input_ids = self.encodings['input_ids'][idx]
        attention_mask = self.encodings['attention_mask'][idx]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }

# Build the dataset from your polite texts
dataset = PoliteDataset(polite_texts, tokenizer, max_length=64)

# Train-test split (80/20)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

training_args = TrainingArguments(
    output_dir="./polite-gpt2-checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    save_strategy="epoch",  # Changed from "steps" to "epoch"
    eval_strategy="epoch",  # Matches save strategy
    logging_steps=100,
    load_best_model_at_end=True,
)


# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,  # Add test dataset for evaluation
)

# Train and save model
trainer.train()
trainer.save_model("./polite-gpt2-model")


from transformers import pipeline

polite_model = GPT2LMHeadModel.from_pretrained("./polite-gpt2-model")
polite_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
polite_tokenizer.pad_token = polite_tokenizer.eos_token

generator = pipeline("text-generation", model=polite_model, tokenizer=polite_tokenizer)

prompt = "Please let me know how to improve"
outputs = generator(prompt, max_length=50, num_return_sequences=1)
print("Generated response:", outputs[0]["generated_text"])
