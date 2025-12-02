# Named Entity Recognition (NER) Extractor

> A comprehensive Named Entity Recognition (NER) system comparing BiLSTM-CRF, RoBERTa, and DistilBERT-CRF architectures on the CoNLL-2003 dataset.

---

## 📘 Project Overview

This project implements and evaluates three distinct neural architectures for **Named Entity Recognition (NER)**:
1.  **BiLSTM-CRF**: A classic recurrent baseline with structured decoding.
2.  **RoBERTa**: A high-performance fine-tuned transformer.
3.  **DistilBERT-CRF**: An optimized lightweight transformer with a CRF layer, featuring advanced training strategies like R-Drop, EMA, and Differential Learning Rates.

**Goal**: To extract named entities (PER, ORG, LOC, MISC) from text while analyzing the trade-offs between model capacity, inference efficiency, and structured prediction.

---

## 🎥 Project Presentation

<div align="center">
  <a href="https://drive.google.com/file/d/1ZlAHbuG2L2DojMAHfnrWev_4G_GqadFD/view?usp=drive_link">
    <img src="https://img.youtube.com/vi/placeholder/maxresdefault.jpg" alt="Watch the Project Presentation" width="600"/>
    <br>
    <b>Click here to watch the full project presentation</b>
  </a>
</div>

*(Note: Since GitHub README does not support direct Google Drive video embedding, please click the link above to view the presentation.)*

---

## ⚙️ Key Features

- **Multi-Architecture Support**: Implementations for BiLSTM, BiLSTM-CRF, DistilBERT (Sequence Classification & CRF), and RoBERTa.
- **Advanced Training**:
    - **Differential Learning Rates**: Layer-wise decay for transformer fine-tuning.
    - **R-Drop**: Consistency regularization for robust generalization.
    - **EMA**: Exponential Moving Average of model weights.
    - **Data Augmentation**: Entity-aware synonym replacement.
- **Comprehensive Evaluation**:
    - **5-Fold Cross-Validation** with strict GroupKFold (by document ID).
    - **Strict Entity-level F1** (CoNLL standard).
    - **Detailed Error Analysis**: Confusion matrices and type-confusion breakdowns.
- **Visualization**:
    - **Embeddings**: PCA and UMAP projections with token annotations.
    - **Clustering**: K-Means analysis of representation quality.
    - **Training Dynamics**: Loss curves and F1 progression plots.

---

## 🧩 Methodology & Results

We benchmarked three model families on the CoNLL-2003 dataset.

| Model | Test F1 | Key Characteristics |
| :--- | :--- | :--- |
| **RoBERTa (Base)** | **91.8% - 92.1%** | State-of-the-art accuracy, heavy computational cost. |
| **DistilBERT-CRF** | **89.64%** | **Best Balance**: 60% smaller than RoBERTa, robust structured prediction. |
| **BiLSTM-CRF** | ~83.2% | Strong non-transformer baseline, validates CRF necessity. |

### Visualization Highlights
The project includes scripts to generate dimensionality reduction plots demonstrating the superior separability of Transformer embeddings compared to BiLSTM.

*(See `report/figures/` for generated plots)*

---

## 🏗️ Repository Structure

```
ner-extractor/
 ┣━ BiLSTM-CRF/           # BiLSTM baseline implementation (Notebooks)
 ┣━ RoBERTa/              # RoBERTa fine-tuning experiments
 ┣━ DistilBERT-CRF/       # MAIN: DistilBERT-CRF implementation
 ┃  ┣━ src/               # Modeling, Trainer, DataModule
 ┃  ┣━ scripts/           # Train, Eval, Visualize, K-Fold
 ┃  ┣━ configs/           # YAML configs for ablation studies
 ┃  ┣━ analysis/          # Visualization scripts & outputs
 ┣━ report/               # LaTeX report & figures (Final Deliverable)
 ┣━ doc/                  # Project documentation & schema
 ┣━ requirements.txt      # Python dependencies
 ┗━ README.md             # This file
```

---

## 🧪 Quick Start (DistilBERT-CRF)

### 1️⃣ Setup Environment

```bash
# Create environment
conda create -n ner python=3.10
conda activate ner

# Install dependencies
pip install -r DistilBERT-CRF/requirements.txt
```

### 2️⃣ Data Preparation

We use the CoNLL-2003 dataset. Follow these steps to download and prepare it:

1.  **Download Data**:
    The dataset is loaded automatically via the Hugging Face `datasets` library. No manual download is strictly required for the transformer models as the training script handles it.
    
    *However, if you need the raw files for the BiLSTM baseline or custom processing:*
    *   **Source**: [Hugging Face CoNLL2003 Viewer](https://huggingface.co/datasets/conll2003)
    *   **Alternative**: [Deep Learning Demos (github)](https://github.com/davidsbatista/NER-datasets/tree/master/CONLL2003)

2.  **Preprocessing**:
    To generate the processed format (JSON/CSV) used by our `TokenizedNERDataset`:

    ```bash
    cd DistilBERT-CRF
    python scripts/prepare_conll03.py --output_dir data/processed/conll03
    ```

3.  **Directory Structure**:
    Ensure your data is placed as follows:
    ```
    ner-extractor/
    ┗━ DistilBERT-CRF/
       ┗━ data/
          ┗━ processed/
             ┗━ conll03/
                ┣━ train.txt
                ┣━ validation.txt
                ┗━ test.txt
    ```

### 3️⃣ Train Model

Run the optimized training pipeline (DistilBERT + CRF + Augmentation + EMA):

```bash
cd DistilBERT-CRF
python scripts/train_distilbert_crf.py \
    --config configs/ablation/aug_on.yaml \
    --run-name final_run
```

### 3️⃣ Visualize Embeddings

Generate PCA and UMAP plots for learned representations:

```bash
# Ensure you are in DistilBERT-CRF directory and have trained a model
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src
python scripts/visualize_embeddings.py \
    --model_path models/distilbert_crf/final_run/best \
    --output_dir ../report/figures/DistiBERT-CRF \
    --sample_size 2000
```

---

## 📈 Evaluation Metrics

Performance is measured using **SeqEval** (strict CoNLL-2003 evaluation):
- **Precision**: % of predicted entities that match gold standard exactly.
- **Recall**: % of gold entities correctly recovered.
- **F1-Score**: Harmonic mean of Precision and Recall.

---

## 📚 References

- **CoNLL-2003 Shared Task**: [Tjong Kim Sang & De Meulder (2003)](https://aclanthology.org/W03-0419.pdf)
- **Hugging Face Transformers**: [Wolf et al. (2020)](https://huggingface.co/transformers/)
- **R-Drop**: [Liang et al. (2021)](https://arxiv.org/abs/2106.14448)

---

## 🧾 License

This project is licensed under the [MIT License](LICENSE).
