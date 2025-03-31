from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re

# Load the trained model and tokenizer
model_path = "./gpt2_reddit_model" 
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

app = FastAPI()

class InputText(BaseModel):
    prompt: str  

# Function to preprocess input (clean spaces, remove unwanted characters)
def preprocess_text(text: str) -> str:
    text = text.strip() 
    text = re.sub(r"\s+", " ", text)  
    return text

# Function to clean model output (remove special tokens and unwanted spaces)
def clean_output(output_text: str) -> str:
    output_text = output_text.replace("<|startoftext|>", "").replace("<|endoftext|>", "")
    return output_text.strip()

@app.post("/generate/")
def generate_text(input_text: InputText):
    # Preprocess user input
    processed_prompt = preprocess_text(input_text.prompt)

    input_ids = tokenizer.encode(processed_prompt, return_tensors="pt")

    output = model.generate(
        input_ids,
        max_length=10,  
        num_return_sequences=1,
        no_repeat_ngram_size=2,  
        top_k=40,  
        top_p=0.85,  
        temperature=0.6,
        do_sample=True
    )

    # Decode the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    
    # Clean and format the final output
    completion = clean_output(generated_text[len(processed_prompt):])

    return {"completion": completion}

# Run the API using: uvicorn filename:app --reload
