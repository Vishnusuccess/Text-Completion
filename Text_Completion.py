import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split

# Download NLTK tokenizer
nltk.download("punkt")

# Load dataset from CSV
df = pd.read_csv("cleaned_reddit_comments.csv")  # Update with actual file path

# Function to split comments into sentences
def preprocess_comments(comments):
    sentences = []
    for comment in comments:
        for sentence in sent_tokenize(comment):
            sentences.append(f"<|startoftext|> {sentence} <|endoftext|>")  # Add markers
    return sentences


# Preprocess the dataset (sentence-level)
sentences = preprocess_comments(df['comment'].dropna().tolist())
sentences = sentences[:1000]
print(sentences[:5])
# Train-validation split (90% train, 10% validation)
train_texts, val_texts = train_test_split(sentences, test_size=0.1, random_state=42)
print("Train Test Split is Completed")
# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Ensure tokenizer has padding token
tokenizer.pad_token = tokenizer.eos_token

# Create dataset dictionary
dataset = DatasetDict({
    "train": Dataset.from_dict({"text": train_texts}),
    "validation": Dataset.from_dict({"text": val_texts}),
})

# ✅ Fix: Add labels for loss calculation
def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=50)
    tokenized["labels"] = tokenized["input_ids"].copy()  # Add labels
    return tokenized

# Tokenize datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=1000,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",        # Save model at each epoch
    learning_rate=0.001,  # 🔹 Adjusted learning rate for 40k sentences
    lr_scheduler_type="cosine",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Adds validation set
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# Save the model and tokenizer
model.save_pretrained("./gpt2_reddit_model")
tokenizer.save_pretrained("./gpt2_reddit_model")

print("Model training complete and saved!")


# import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
# from datasets import Dataset
# import pandas as pd
# import nltk
# from nltk.tokenize import sent_tokenize

# # Download NLTK sentence tokenizer
# nltk.download("punkt")

# # Load dataset from CSV
# df = pd.read_csv("cleaned_reddit_comments.csv")  # Update with actual file path

# # Function to split comments into sentences
# def preprocess_comments(comments):
#     sentences = []
#     for comment in comments:
#         for sentence in sent_tokenize(comment):
#             sentences.append(sentence)  # Store sentence-level data
#     return sentences

# # Preprocess the dataset (sentence-level)
# sentences = preprocess_comments(df['comment'].dropna().tolist())
# print(sentences[:5])
# # Load tokenizer and model
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")
# # Ensure tokenizer has padding token
# tokenizer.pad_token = tokenizer.eos_token

# # Convert dataset into a Hugging Face `Dataset` object
# dataset = Dataset.from_dict({"text": sentences})

# # Tokenization function
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=50)

# # Tokenize dataset
# tokenized_dataset = dataset.map(tokenize_function, batched=True)

# # Set training arguments
# training_args = TrainingArguments(
#     output_dir="./results",
#     num_train_epochs=1,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=8,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     logging_steps=10,
#     save_strategy="epoch",
#     evaluation_strategy="epoch",
# )

# # Initialize Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
# )

# # Train the model
# trainer.train()

# # Save the model and tokenizer
# model.save_pretrained("./gpt2_reddit_model")
# tokenizer.save_pretrained("./gpt2_reddit_model")

# print("Model training complete and saved!")

