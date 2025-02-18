import pandas as pd

# Load the dataset
test_df = pd.read_csv('cnn_dailymail/test.csv')
val_df = pd.read_csv('cnn_dailymail/validation.csv')
train_df = pd.read_csv('cnn_dailymail/train.csv')
# Get the number of samples
num_test_samples = test_df.shape[0]
num_val_samples = val_df.shape[0]
num_train_samples = train_df.shape[0]
print(f"Number of test samples: {num_test_samples}")
print(f"Number of val samples: {num_val_samples}")
print(f"Number of train samples: {num_train_samples}")

train_subset = train_df.sample(n=2100, random_state=42)
val_subset = val_df.sample(n=600, random_state=42)
test_subset = test_df.sample(n=300, random_state=42)

# Check the length to verify
print(f"Train subset: {len(train_subset)} samples")
print(f"Validation subset: {len(val_subset)} samples")
print(f"Test subset: {len(test_subset)} samples")

import os
import numpy as np
import random
from datasets import load_dataset
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, AdamW, get_scheduler
import torch
from tqdm import tqdm  # Import tqdm for progress bars
import evaluate
import time
from torch.utils.data import DataLoader, Subset

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

path = "/teamspace/studios/this_studio/cnn_dailymail"

# Load dataset
data_files = {
    "train": os.path.join(path, "train.csv"),
    "validation": os.path.join(path, "validation.csv"),
    "test": os.path.join(path, "test.csv")
}
dataset = load_dataset("csv", data_files=data_files)

input_column = 'article'      
target_column = 'highlights'  

# Step 2: Sample 10% of the training data
train_size = len(dataset['train'])
subset_size = int(0.1 * train_size)
print(f"\nTotal number of training examples: {train_size}")
print(f"Sampling 10% of the training data: {subset_size} examples")
indices = random.sample(range(train_size), subset_size)
train_subset = Subset(dataset['train'], indices)
print("Train subset created.")

# Step 3: Initialize tokenizer
print("\nInitializing tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer initialized. Pad token set to:", tokenizer.pad_token)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Step 4: Define custom collate function
def custom_collate_fn(batch):
    inputs = [example[input_column] for example in batch]
    targets = [example[target_column] for example in batch]

    # Tokenize inputs and targets separately
    tokenized_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    tokenized_targets = tokenizer(
        targets,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    # Concatenate inputs and targets for conditional generation
    input_ids = tokenized_inputs['input_ids']
    target_ids = tokenized_targets['input_ids']

    # Create labels: mask the input tokens with -100 and set labels for target tokens
    labels = torch.cat([
        torch.full((input_ids.size(0), input_ids.size(1)), -100, dtype=torch.long),
        target_ids
    ], dim=1)

    # Concatenate input_ids and target_ids
    input_ids = torch.cat([input_ids, target_ids], dim=1)

    # Update attention_mask
    attention_mask = torch.cat([
        tokenized_inputs['attention_mask'],
        tokenized_targets['attention_mask']
    ], dim=1)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

# Step 5: Initialize GPT-Neo model
print("\nLoading GPT-Neo 125M model...")
model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M')
model.config.pad_token_id = tokenizer.eos_token_id

# Step 6: Freeze all layers except the last transformer block
print("\nFreezing all layers except the last transformer block...")
for name, param in model.named_parameters():
    if 'transformer.h.' in name:
        layer_num = int(name.split('.')[2])
        if layer_num != (len(model.transformer.h) - 1): 
            param.requires_grad = False
    else:
        param.requires_grad = True  
print("Freezing complete. Only the last transformer block is trainable.")

# Step 7: Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nNumber of trainable parameters: {trainable_params}")
    
model.to(device)

# Step 8: Check GPU availability and print GPU information
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

# Step 9: Set up DataLoader
print("\nSetting up DataLoader...")
train_dataloader = DataLoader(train_subset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
print("DataLoader set up successfully.")

# Step 10: Setup optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_dataloader) 
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

print("\nStarting training...")
start_time = time.time()
num_epochs = 10
max_grad_norm = 1.0

for epoch in range(num_epochs):
    model.train()
    # Use tqdm for progress bar
    with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

    print(f"Epoch {epoch + 1}/{num_epochs} finished.")

end_time = time.time()
training_time = end_time - start_time
print(f"\nTraining complete! Total training time: {training_time / 60:.2f} minutes.")

# Step 12: Save the fine-tuned model
print("\nSaving the fine-tuned model...")
model.save_pretrained("./gpt-neo-finetuned")
tokenizer.save_pretrained("./gpt-neo-finetuned")
print("Model saved successfully.")

# Save the model's weights as a .pt file after training
torch.save(model.state_dict(), "./gpt-neo-finetuned-weights.pt")
print("Model weights saved as gpt-neo-finetuned-weights.pt")

# Step 13: Evaluate the model on the test set (test loss)
print("\nEvaluating the model on the test set for loss...")
model.eval() 
total_loss = 0
num_batches = 0

# Use tqdm for test evaluation progress
with torch.no_grad():
    with tqdm(total=len(dataset['test']), desc="Evaluating Test Set", unit="batch") as pbar:
        for batch in DataLoader(dataset['test'], batch_size=8, shuffle=False, collate_fn=custom_collate_fn):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
            
            pbar.update(1)

avg_test_loss = total_loss / num_batches
print(f"Average Test Loss: {avg_test_loss}")

# Step 14: Generate summaries and compute ROUGE scores
print("\nGenerating summaries and computing ROUGE scores...")
rouge_metric = evaluate.load('rouge')

generated_summaries = []
references = []

with torch.no_grad():
    for batch in tqdm(DataLoader(dataset['test'], batch_size=8, shuffle=False, collate_fn=custom_collate_fn), desc="Generating Summaries", unit="batch"):
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=128,
            num_beams=5,
            early_stopping=True
        )

        # Decode generated summaries
        decoded_summaries = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        generated_summaries.extend(decoded_summaries)

        # Add references (ground truth summaries)
        decoded_references = [
            tokenizer.decode([token_id for token_id in target if token_id != -100], skip_special_tokens=True) 
            for target in batch['labels']
        ]
        references.extend(decoded_references)

# Compute ROUGE scores
rouge_scores = rouge_metric.compute(predictions=generated_summaries, references=references)
print(f"\nROUGE Scores: {rouge_scores}")
