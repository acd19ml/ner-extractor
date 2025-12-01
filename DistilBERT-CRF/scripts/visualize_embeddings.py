#!/usr/bin/env python
"""
Visualize learned embeddings using PCA and UMAP, and perform KMeans clustering.
Usage: python scripts/visualize_embeddings.py --model_path models/distilbert_crf_best --data_dir data/processed
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Try importing UMAP, fallback to TSNE if not available
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from src.data_module import create_dataloaders
from src.modeling import DistilBertCrfForTokenClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_embeddings(model, dataloader, device, tokenizer) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Extract hidden states for valid entity tokens (ignoring 'O' tag if desired, or keeping all).
    For visualization, we typically focus on named entities to see separation.
    """
    model.eval()
    embeddings = []
    labels = []
    token_texts = []
    
    # Map ID to Label (assuming standard BIOES or IOB scheme)
    # We need to access the label map from the dataset or config
    # For now, we'll assume standard CoNLL-2003 labels if not provided
    id2label = model.config.id2label if hasattr(model.config, "id2label") else {}
    
    logger.info("Extracting embeddings...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # Labels might be -100 for ignored tokens
            batch_labels = batch["labels"].to(device)
            
            # Get hidden states from the encoder (before CRF)
            # DistilBertCrfForTokenClassification has .distilbert attribute
            if hasattr(model, "distilbert"):
                outputs = model.distilbert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            elif hasattr(model, "bert"): # Fallback standard naming
                 outputs = model.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            else:
                 # Try typical HF output if it's just a base model
                 outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            # Last hidden state: [batch_size, seq_len, hidden_dim]
            if hasattr(outputs, "last_hidden_state"):
                last_hidden_state = outputs.last_hidden_state
            else:
                # Some custom outputs might differ, check structure
                last_hidden_state = outputs[0] # Usually logits, but if we asked for hidden states... 
                # Actually for HF, hidden_states is a tuple. 
                if hasattr(outputs, "hidden_states"):
                    last_hidden_state = outputs.hidden_states[-1]

            # Flatten
            bs, seq_len, dim = last_hidden_state.shape
            last_hidden_state = last_hidden_state.view(-1, dim)
            batch_labels = batch_labels.view(-1)
            input_ids_flat = input_ids.view(-1)
            
            # Filter: ignore -100 and optionally 'O' (label id 0 usually)
            # Let's keep entities only for clearer visualization
            mask = batch_labels != -100
            
            valid_hidden = last_hidden_state[mask]
            valid_lbls = batch_labels[mask]
            valid_input_ids = input_ids_flat[mask]
            
            embeddings.append(valid_hidden.cpu().numpy())
            labels.append(valid_lbls.cpu().numpy())
            
            # Decode tokens
            decoded_tokens = tokenizer.convert_ids_to_tokens(valid_input_ids.cpu().numpy())
            token_texts.extend(decoded_tokens)
            
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    token_texts = np.array(token_texts)
    
    # Convert label IDs to strings
    str_labels = np.array([id2label.get(l, str(l)) for l in labels])
    
    return embeddings, labels, str_labels, token_texts

def plot_projection(X_proj, labels, tokens, title, output_path):
    plt.figure(figsize=(12, 10))
    
    # Create DataFrame for Seaborn
    df = pd.DataFrame(X_proj, columns=["Dim 1", "Dim 2"])
    df["Label"] = labels
    
    sns.scatterplot(
        data=df,
        x="Dim 1",
        y="Dim 2",
        hue="Label",
        palette="tab10",
        s=15,
        alpha=0.6
    )
    
    # Annotate representative samples
    # Pick random samples for each class to annotate
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        if lbl == 'O': continue # Skip O annotation to reduce clutter
        
        # Get indices for this label
        indices = np.where(labels == lbl)[0]
        if len(indices) > 0:
            # Pick 2-3 random samples
            sample_indices = np.random.choice(indices, min(3, len(indices)), replace=False)
            for idx in sample_indices:
                x, y = X_proj[idx]
                txt = tokens[idx]
                # Clean up special chars
                txt = txt.replace('Ġ', '') 
                plt.annotate(txt, (x, y), fontsize=8, alpha=0.8, 
                             xytext=(5, 5), textcoords='offset points')
    
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved plot with annotations to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="analysis/figures", help="Dir to save plots")
    parser.add_argument("--sample_size", type=int, default=2000, help="Number of tokens to visualize")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    # Load Model
    logger.info(f"Loading model from {args.model_path}")
    try:
        # Load custom model class if saved with save_pretrained
        model = DistilBertCrfForTokenClassification.from_pretrained(args.model_path)
    except Exception as e:
        logger.warning(f"Could not load as DistilBertCrfForTokenClassification: {e}")
        logger.info("Trying generic AutoModel (might fail if architecture differs)...")
        model = AutoModelForTokenClassification.from_pretrained(args.model_path)
        
    model.to(device)
    # Fallback to base tokenizer if local load fails (avoids protobuf/path issues)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except Exception as e:
        logger.warning(f"Could not load tokenizer from checkpoint: {e}")
        logger.info("Falling back to 'distilbert-base-cased' tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    
    # Load Data
    # Assuming standard processed data path structure relative to project root or provided arg
    # We need a data_dir argument ideally, defaulting to something sensible
    data_dir = Path("DistilBERT-CRF/data/processed/conll03") 
    if not data_dir.exists():
         data_dir = Path("data/processed/conll03") # Fallback
    
    logger.info(f"Loading validation data from {data_dir}")
    dataloaders, label_info = create_dataloaders(
        processed_dir=data_dir,
        tokenizer=tokenizer,
        max_length=128, # Match training config ideally
        batch_size=32,
        eval_batch_size=32,
        shuffle=False
    )
    val_loader = dataloaders["validation"]
    
    # Extract Embeddings
    model.config.id2label = label_info.id_to_label # Ensure mapping exists
    embeddings, numeric_labels, str_labels, token_texts = extract_embeddings(model, val_loader, device, tokenizer)
    
    # Subsample if too large
    if len(embeddings) > args.sample_size:
        indices = np.random.choice(len(embeddings), args.sample_size, replace=False)
        embeddings = embeddings[indices]
        str_labels = str_labels[indices]
        numeric_labels = numeric_labels[indices]
        token_texts = token_texts[indices]

    # 1. PCA
    logger.info("Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(embeddings)
    plot_projection(X_pca, str_labels, token_texts, "PCA of Entity Embeddings", output_dir / "pca_embeddings.png")
    
    # 2. UMAP / t-SNE
    if HAS_UMAP:
        logger.info("Running UMAP...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
        X_umap = reducer.fit_transform(embeddings)
        plot_projection(X_umap, str_labels, token_texts, "UMAP of Entity Embeddings", output_dir / "umap_embeddings.png")
    else:
        logger.info("UMAP not installed, running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
        X_umap = tsne.fit_transform(embeddings)
        plot_projection(X_umap, str_labels, token_texts, "t-SNE of Entity Embeddings", output_dir / "tsne_embeddings.png")
        
    # 3. KMeans Clustering
    k = len(set(str_labels)) - 1 # exclude O ideally, or just use all
    if k < 2: k = 5
    logger.info(f"Running KMeans with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Evaluate Homogeneity (how well clusters match ground truth)
    score = homogeneity_score(numeric_labels, cluster_labels)
    logger.info(f"KMeans Homogeneity Score: {score:.4f}")
    
    with open(output_dir / "clustering_metrics.txt", "w") as f:
        f.write(f"Homogeneity Score: {score:.4f}\n")

if __name__ == "__main__":
    main()

