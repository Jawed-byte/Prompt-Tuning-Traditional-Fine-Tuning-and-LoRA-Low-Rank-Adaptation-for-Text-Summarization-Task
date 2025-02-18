import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
from rouge_score import rouge_scorer
from tqdm import tqdm
from torch.nn import CrossEntropyLoss

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

train_subset = train_df.sample(n=21000, random_state=42)
val_subset = val_df.sample(n=6000, random_state=42)
test_subset = test_df.sample(n=3000, random_state=42)

# Check the length to verify
print(f"Train subset: {len(train_subset)} samples")
print(f"Validation subset: {len(val_subset)} samples")
print(f"Test subset: {len(test_subset)} samples")

def load_and_preprocess_data_from_df(df, num_prompts):
    tokenized_articles = []
    tokenized_summaries = []

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    for article, summary in zip(df["article"], df["highlights"]):  
        max_length_article = MAX_LEN - num_prompts 
        article_tokens = tokenizer.encode(article, truncation=True, max_length=max_length_article)
        summary_tokens = tokenizer.encode(summary, truncation=True, max_length=300)

        max_length_summary = MAX_LEN
        padded_article = article_tokens + [tokenizer.eos_token_id] * (max_length_article - len(article_tokens))
        padded_summary = summary_tokens + [tokenizer.eos_token_id] * (max_length_summary - len(summary_tokens))

        tokenized_articles.append(padded_article)
        tokenized_summaries.append(padded_summary)

    return tokenized_articles, tokenized_summaries

# Constants
MODEL_NAME = "gpt2"
PROMPT_TOKEN = "[SUMMARIZE]"
MAX_LEN = 1024

# Soft Prompt Vocabulary
soft_prompt_vocab = ["[SUMMARIZE]"]  # Define your custom vocabulary here

# Create a word2idx dictionary for the soft prompt vocabulary
soft_prompt_word2idx = {word: idx for idx, word in enumerate(soft_prompt_vocab)}

num_prompts = len([soft_prompt_word2idx[word] for word in PROMPT_TOKEN.split()])
prompt_id = torch.tensor([soft_prompt_word2idx[word] for word in PROMPT_TOKEN.split()])

# Model Architecture
class GPT2WithSoftPrompt(torch.nn.Module):
    def __init__(self, model_name, num_prompts, embedding_size=768):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        for param in self.gpt2.parameters():
            param.requires_grad = Falsef
        self.soft_prompt = torch.nn.Embedding(num_prompts, embedding_size)

    def forward(self, input_ids, prompt_ids):
        prompt_embeddings = self.soft_prompt(prompt_ids)
        base_embeddings = self.gpt2.transformer.wte(input_ids)
        embeddings = torch.cat([prompt_embeddings, base_embeddings.squeeze(0)], dim=0)
        outputs = self.gpt2(inputs_embeds=embeddings)
        return outputs

# Load and preprocess the data using the correct DataFrames
tokenized_articles_train, tokenized_summaries_train = load_and_preprocess_data_from_df(train_subset, num_prompts)
tokenized_articles_validation, tokenized_summaries_validation = load_and_preprocess_data_from_df(val_subset, num_prompts)
tokenized_articles_test, tokenized_summaries_test = load_and_preprocess_data_from_df(test_subset, num_prompts)

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Model Initialization
model = GPT2WithSoftPrompt(MODEL_NAME, num_prompts).to(device)

# Constants
BATCH_SIZE = 4
EPOCHS = 10

prompt_id = prompt_id.to(device)

def decode_summary(logits, tokenizer):
    """Decode the logits into a summary text."""
    generated_ids = torch.argmax(logits, dim=-1)
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

def calculate_rouge_scores(preds, refs):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    for pred, ref in zip(preds, refs):
        scores = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key] += scores[key].fmeasure
    for key in rouge_scores:
        rouge_scores[key] /= len(preds)
    return rouge_scores

def fine_tune_on_summarization(model, train_articles, train_summaries, val_articles, val_summaries, test_articles, test_summaries):
    optimizer = torch.optim.Adam(model.soft_prompt.parameters())
    best_val_loss = float('inf')
    no_improvement_epochs = 0

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        total_train_loss = 0

        with tqdm(enumerate(zip(train_articles, train_summaries)), total=len(train_articles), desc=f"Epoch {epoch + 1}/{EPOCHS}", unit="batch") as progress:
            for idx, (article, summary) in progress:
                input_ids = torch.tensor(article).to(device)
                labels = torch.tensor(summary).to(device)
                outputs = model(input_ids, prompt_id)

                ignore_index = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -100
                loss = CrossEntropyLoss(ignore_index=ignore_index)(outputs.logits, labels)
                total_train_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_articles)
        print(f"Train Loss: {avg_train_loss}")

        # Validation
        model.eval()
        total_val_loss = 0
        val_preds, val_refs = [], []

        with torch.no_grad():
            for article, summary in tqdm(zip(val_articles, val_summaries), total=len(val_articles), desc="Validation", unit="batch"):
                input_ids = torch.tensor(article).to(device)
                labels = torch.tensor(summary).to(device)
                outputs = model(input_ids, prompt_id)

                val_loss = CrossEntropyLoss(ignore_index=ignore_index)(outputs.logits, labels)
                total_val_loss += val_loss.item()

                generated_summary = decode_summary(outputs.logits, tokenizer)
                val_preds.append(generated_summary)
                val_refs.append(tokenizer.decode(summary, skip_special_tokens=True))

        avg_val_loss = total_val_loss / len(val_articles)
        rouge_scores = calculate_rouge_scores(val_preds, val_refs)
        print(f"Validation Loss: {avg_val_loss}")
        print(f"Validation ROUGE scores: {rouge_scores}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
                break

    # Testing
    model.eval()
    total_test_loss = 0
    test_preds, test_refs = [], []

    with torch.no_grad():
        for article, summary in tqdm(zip(test_articles, test_summaries), total=len(test_articles), desc="Test", unit="batch"):
            input_ids = torch.tensor(article).to(device)
            labels = torch.tensor(summary).to(device)
            outputs = model(input_ids, prompt_id)

            test_loss = CrossEntropyLoss(ignore_index=ignore_index)(outputs.logits, labels)
            total_test_loss += test_loss.item()

            generated_summary = decode_summary(outputs.logits, tokenizer)
            test_preds.append(generated_summary)
            test_refs.append(tokenizer.decode(summary, skip_special_tokens=True))

    avg_test_loss = total_test_loss / len(test_articles)
    test_rouge_scores = calculate_rouge_scores(test_preds, test_refs)
    print(f"Test Loss: {avg_test_loss}")
    print(f"Test ROUGE scores: {test_rouge_scores}")

    return model

# Debugging to confirm tokenized data
# print(f"Train tokenized: {len(tokenized_articles_train)}, Validation tokenized: {len(tokenized_articles_validation)}, Test tokenized: {len(tokenized_articles_test)}")

# Now call fine-tune function
fine_tuned_model = fine_tune_on_summarization(
    model, 
    tokenized_articles_train, 
    tokenized_summaries_train, 
    tokenized_articles_validation, 
    tokenized_summaries_validation, 
    tokenized_articles_test, 
    tokenized_summaries_test
)

