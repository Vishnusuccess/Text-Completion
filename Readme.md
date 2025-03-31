# Text Completion API (FastAPI + GPT-2)

## Overview
This project provides an API for text completion using a fine-tuned GPT-2 model. It takes a prompt from the user and generates a natural language continuation. The goal is to assist customer service agents by suggesting sentence completions, improving response efficiency.

---

## Installation

### Prerequisites
Ensure you have Python 3.8+ installed.

### 1. Clone the Repository
```sh
git clone git@github.com:Vishnusuccess/Text-Completion.git
cd Text-Completion 
```

### 2. Create a Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

---

## Model Training
The GPT-2 model was trained using **1000 rows** from the provided dataset for memory efficiency.

### Training Steps (Jupyter Notebook)
1. **Load the Data**: Preprocessed Reddit comments were used.
2. **Tokenization**: The dataset was tokenized using `GPT2Tokenizer`.
3. **Training**: GPT-2 was fine-tuned using **PyTorch & Transformers**.
4. **Saving the Model**: The trained model is saved in `./gpt2_reddit_model`.

---

### Create the new cleaned dataset -----> include the reddit.db in the directory 
Run the following command:
```sh
python data_creation.py
```
### Train the Model 
Run the following command:
```sh
python Text_Completion.py
```
## API Endpoints

### Start the API Server
Run the following command:
```sh
uvicorn api:app --host 0.0.0.0 --port 8000
```
### Go to the API 
In the brower copy and past this url
```sh
http://127.0.0.1:8000/docs
```
### Endpoint: `/generate/`
#### **Request Format**
```json
{
  "prompt": "Hi, please could you provide me with "
}
```

#### **Response Format**
```json
{
  "completion": "your account number and postcode so I can assist you?"
}
```

---

## Preprocessing & Improvements

### Input Preprocessing
- Strips extra spaces.
- Normalizes multiple spaces.

### Output Cleaning
- Removes special tokens (`<|startoftext|>`, `<|endoftext|>`).
- Ensures output is properly formatted.

---

## Possible Improvements
1. **Model Fine-tuning**: Increase training dataset size for better accuracy.
2. **Batch Processing**: Handle multiple prompts efficiently.
3. **Logging & Monitoring**: Track model performance in real time.
4. **Caching Mechanism**: Reduce response time for repeated queries.

---


