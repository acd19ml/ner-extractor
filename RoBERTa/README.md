# RoBERTa Named Entity Recognition (NER) on CoNLL-2003

This project implements a Named Entity Recognition (NER) system using the RoBERTa-base model fine-tuned on the CoNLL-2003 dataset. The model identifies and classifies named entities in text into four categories: PERSON, ORGANIZATION, LOCATION, and MISC.

Original file is located at
    https://colab.research.google.com/drive/1973zB8OObU-PZ4g2arNNLKIOCuaom5he

## Model Overview

- **Base Model**: `roberta-base`
- **Task**: Token Classification (Named Entity Recognition)
- **Dataset**: CoNLL-2003 English dataset
- **Entity Types**: PER, ORG, LOC, MISC
- **Labels**: 9 classes including BIO tagging scheme

### Label Schema
- `O`: Outside of any entity
- `B-PER`, `I-PER`: Person entities
- `B-ORG`, `I-ORG`: Organization entities  
- `B-LOC`, `I-LOC`: Location entities
- `B-MISC`, `I-MISC`: Miscellaneous entities

## Training Process

### 1. Data Preparation
- Loads CoNLL-2003 dataset in CONLL format
- Extracts tokens and NER tags
- Maps labels to IDs for model training

### 2. Model Configuration
- Uses `roberta-base` with token classification head
- Handles subword tokenization with label alignment
- Implements smart label strategy for word pieces

### 3. Training Parameters
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3
- Weight decay: 0.01
- Max sequence length: 512

### 4. Evaluation Metrics
Uses `seqeval` for proper NER evaluation:
- Precision, Recall, F1-score
- Exact match entity evaluation

## Usage

### Load Trained Model

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

model_path = "/path/to/ner_roberta_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

### Run Prediction

```python
def predict_entities(sentence, model, tokenizer):
    tokens = sentence.split()
    inputs = tokenizer(
        tokens, 
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
    
    # Process predictions and align with original tokens
    # Returns tokens with their corresponding NER labels
```

### Example Output

```
==================================================
Input Sentence: Microsoft was founded by Bill Gates and Paul Allen
Tokenize result: ['Microsoft', 'was', 'founded', 'by', 'Bill', 'Gates', 'and', 'Paul', 'Allen']

NER prediction:
----------------------------------------
Microsoft       -> B-ORG
was             -> O
founded         -> O
by              -> O
Bill            -> B-PER
Gates           -> I-PER
and             -> O
Paul            -> B-PER
Allen           -> I-PER
----------------------------------------
```

## Performance

The model achieves competitive performance on the CoNLL-2003 test set with metrics including:
- F1-score
- Precision  
- Recall
- Accuracy

Exact evaluation results are available in the training outputs.

## Key Features

- **Label Alignment**: Proper handling of RoBERTa's subword tokenization for NER
- **BIO Scheme**: Standard Beginning-Inside-Outside tagging
- **Efficient Training**: Optimized training with Hugging Face Trainer
- **Comprehensive Evaluation**: Full NER evaluation using seqeval

## Applications

This NER model can be used for:
- Information extraction from text
- Document preprocessing and analysis
- Entity-aware text processing pipelines
- Chatbots and virtual assistants
- Search and recommendation systems

## Notes

- The model expects pre-tokenized input or will split on whitespace
- Best performance on formal text similar to CoNLL-2003
- For production use, consider adding post-processing for entity grouping

## License

The model is based on RoBERTa-base and CoNLL-2003 data. Please check the respective licenses for commercial use.