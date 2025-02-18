import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import pandas as pd
from tqdm import tqdm
import os

class SummarizationDataset(Dataset):
    """Custom Dataset for text summarization task"""
    
    def __init__(self, data_path, tokenizer, max_length=1024, data_percentage=1.0):
        """Initialize the dataset"""
        super().__init__()  # Add proper Dataset initialization
        try:
            self.data = pd.read_csv(data_path)
            if 'article' not in self.data.columns or 'highlights' not in self.data.columns:
                raise ValueError("CSV must contain 'article' and 'highlights' columns")
            
            # Take only a percentage of the dataset
            if data_percentage < 1.0:
                self.data = self.data.sample(frac=data_percentage, random_state=42).reset_index(drop=True)
                
        except Exception as e:
            raise RuntimeError(f"Error loading dataset from {data_path}: {str(e)}")
            
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            article = str(self.data.iloc[idx]['article'])
            highlights = str(self.data.iloc[idx]['highlights'])
            
            # Format input with special tokens
            input_text = f"Summarize: {article} Summary: {highlights}"
            
            # Tokenize with proper handling of truncation and padding
            encodings = self.tokenizer(
                input_text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors="pt"
            )
            
            input_ids = encodings['input_ids'].squeeze()
            attention_mask = encodings['attention_mask'].squeeze()
            
            # Create labels (shift input_ids left by 1)
            labels = input_ids.clone()
            labels[:-1] = input_ids[1:].clone()
            labels[-1] = -100  # Don't compute loss for last token
            
            # Mark padding tokens in labels as -100 so they're ignored in loss
            labels[attention_mask == 0] = -100
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            raise

def train_epoch(model, dataloader, optimizer, device, gradient_accumulation_steps=1):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    optimizer.zero_grad()  # Reset gradients at start of epoch
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for idx, batch in enumerate(progress_bar):
        try:
            labels = batch['labels'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            input_ids = batch['input_ids'].to(device)

            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Scale loss by gradient accumulation steps
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            # Only update weights after accumulating enough gradients
            if (idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
            
        except Exception as e:
            print(f"Error in training batch {idx}: {str(e)}")
            continue
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            try:
                labels = batch['labels'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                input_ids = batch['input_ids'].to(device)
                
                
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
            except Exception as e:
                print(f"Error in evaluation batch: {str(e)}")
                continue
    
    return total_loss / len(dataloader)

def main():
    # Hyperparameters
    config = {
        'batch_size': 4,
        'gradient_accumulation_steps': 4,  # Effective batch size will be 16
        'epochs': 10,
        'learning_rate': 2e-5,
        'max_length': 1024,
        'warmup_steps': 100,
        'save_dir': 'model_checkpoints',
        'data_percentage': 0.1
    }
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Set device and enable deterministic behavior
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Using device: {device}")
    
    try:
        # Load tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=64,
            target_modules=["c_attn", "c_proj"]
        )
        
        # Get PEFT model
        model = get_peft_model(model, peft_config)
        model.to(device)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/all_params:.2%} of {all_params:,} total)")
        
        # Load datasets
        data_paths = {
            'train': 'cnn_dailymail/train.csv',
            'val': 'cnn_dailymail/validation.csv',
            'test': 'cnn_dailymail/test.csv'
        }
        
        datasets = {
            split: SummarizationDataset(
                path,
                tokenizer,
                max_length=config['max_length'],
                data_percentage=config['data_percentage']
            ) for split, path in data_paths.items()
        }
        
        dataloaders = {
            split: DataLoader(
                dataset,
                batch_size=config['batch_size'],
                shuffle=(split == 'train')
            ) for split, dataset in datasets.items()
        }
        
        # Initialize optimizer with warmup
        from transformers import get_linear_schedule_with_warmup
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
        num_training_steps = len(dataloaders['train']) * config['epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=num_training_steps
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(config['epochs']):
            print(f"\nEpoch {epoch + 1}/{config['epochs']}")
            
            # Train
            train_loss = train_epoch(
                model,
                dataloaders['train'],
                optimizer,
                device,
                config['gradient_accumulation_steps']
            )
            print(f"Training loss: {train_loss:.4f}")
            
            # Evaluate
            val_loss = evaluate(model, dataloaders['val'], device)
            print(f"Validation loss: {val_loss:.4f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained(os.path.join(config['save_dir'], "best_model"))
                print("Saved best model!")
            
            scheduler.step()
        
        # Final test evaluation
        test_loss = evaluate(model, dataloaders['test'], device)
        print(f"\nFinal test loss: {test_loss:.4f}")
        
    except Exception as e:
        print(f"Error in training process: {str(e)}")
        raise

if __name__ == "__main__":
    main()