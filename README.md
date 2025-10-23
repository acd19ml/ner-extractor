# Named Entity Recognition (NER) Extractor

> A lightweight and extensible Named Entity Recognition (NER) system built for sequence labeling tasks, evaluated on the CoNLL-2003 dataset.

---

## 📘 Project Overview

**Named Entity Recognition (NER)** aims to extract named entities such as **persons (PER)**, **organizations (ORG)**, **locations (LOC)**, and **miscellaneous entities (MISC)** from text.

For example:
> _"I study at City University of Hong Kong."_  
> → **City University of Hong Kong** → `ORG`

This project implements and evaluates a NER extractor using **sequence labeling** methods (token-level classification), with flexibility to integrate more advanced architectures like **transformers** or **prompt-based models**.

---

## ⚙️ Key Features

- ✅ Sequence labeling model with token-level classification  
- ✅ Compatible with **CoNLL03** and **Hugging Face datasets**  
- ✅ Modular pipeline: data preprocessing → model training → evaluation  
- ✅ Easily extensible to other datasets or architectures  
- ✅ Supports PyTorch and Hugging Face Transformers

---

## 🧩 Methodology

This project treats NER as a **sequence labeling problem**, where each token is assigned a tag (BIO or BIOES scheme):

| Tag   | Meaning                      |
| ----- | ---------------------------- |
| B-ORG | Beginning of an organization |
| I-ORG | Inside an organization       |
| O     | Outside any named entity     |

### 1️⃣ Model Options

You may explore multiple approaches:

| Approach             | Description                                                |
| -------------------- | ---------------------------------------------------------- |
| **BiLSTM + CRF**     | Classical sequence labeling model for NER                  |
| **BERT / RoBERTa**   | Transformer-based contextual embedding model               |
| **Prompt-based NER** | Treats NER as a text generation or span prediction problem |

Example model architecture (sequence labeling):

[Input Text] → [Tokenizer] → [Encoder (BERT/BiLSTM)] → [Linear Layer] → [Softmax/CRF] → [Entity Tags]

### 2️⃣ Loss Function
- Cross-Entropy for token classification  
- Optional: CRF layer log-likelihood if using CRF  

---

## 📊 Dataset

This project uses the **CoNLL-2003 NER dataset**, which contains four entity types:

| Entity | Example        | Description   |
| ------ | -------------- | ------------- |
| PER    | Barack Obama   | Person        |
| ORG    | United Nations | Organization  |
| LOC    | London         | Location      |
| MISC   | FIFA           | Miscellaneous |

You can load the dataset via **Hugging Face Datasets**:

```python
from datasets import load_dataset
dataset = load_dataset("conll2003")
```



------





## **🏗️ Repository Structure**



```
ner-extractor/
 ┣━ data/                 # Dataset and preprocessing scripts
 ┣━ models/               # Model architectures (BiLSTM, BERT, etc.)
 ┣━ utils/                # Tokenization, label mapping, metrics
 ┣━ train.py              # Training loop
 ┣━ evaluate.py           # Evaluation script
 ┣━ requirements.txt      # Dependencies
 ┗━ README.md             # Project documentation
```



------





## **🧪 Training & Evaluation**







### **1️⃣ Setup Environment**



```
conda create -n ner python=3.13
conda activate ner
pip install -r requirements.txt
```



### **2️⃣ Train Model**



```
python train.py \
  --model bert-base-cased \
  --epochs 5 \
  --batch_size 16 \
  --lr 3e-5 \
  --dataset conll2003
```



### **3️⃣ Evaluate Model**



```
python evaluate.py --model_path ./checkpoints/best_model.pt
```



### **4️⃣ Sample Output**



```
Sentence: "Barack Obama was born in Hawaii."
Entities:
  Barack Obama  → PER
  Hawaii        → LOC
```



------





## **📈 Evaluation Metrics**



| **Metric**                  | **Description**                                     |
| --------------------------- | --------------------------------------------------- |
| **Precision**               | Correct predicted entities / All predicted entities |
| **Recall**                  | Correct predicted entities / All true entities      |
| **F1-score**                | Harmonic mean of precision and recall               |
| **Entity-level evaluation** | Using seqeval for CoNLL-style F1 computation        |

Example:

```
from seqeval.metrics import classification_report
print(classification_report(true_labels, pred_labels))
```



------





## **📚 References**





- Tjong Kim Sang, E. F., & De Meulder, F. (2003).

  *Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition.*

  [Paper Link](https://aclanthology.org/W03-0419.pdf)

- Hugging Face Datasets: https://huggingface.co/datasets/conll2003

- Stanford NLP Course Slides: [slp3-17 NER lecture](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/slides/cs224n-2021-lecture17-ner.pdf)





------





## **🧠 Future Work**





- Explore **prompt-based NER** with instruction-tuned LLMs

- Add support for **multilingual datasets (WikiANN)**

- Experiment with **domain adaptation** and **low-resource NER**

  

------





## **🧾 License**





This project is licensed under the [MIT License](LICENSE).