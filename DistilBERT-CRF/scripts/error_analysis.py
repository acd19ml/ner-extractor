#!/usr/bin/env python
"""
Generate Confusion Matrix and perform Error Analysis.
Usage: python scripts/error_analysis.py --log_file training_logs/distilbert_crf_full.log
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_predictions(file_path: Path):
    """
    Parses a log file or a specific prediction output file.
    Ideally, during training/inference, you dump: Token | Gold | Pred
    If that file exists, we parse it.
    
    For now, this script generates a dummy confusion matrix to demonstrate the visualization
    code structure, as parsing raw logs for token-level predictions requires the raw pred dump.
    """
    # Placeholder: In a real scenario, load 'predictions.txt' containing columns:
    # Token, Gold_Label, Predicted_Label
    return [], []

def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='.2f', 
        xticklabels=labels, 
        yticklabels=labels, 
        cmap='Blues'
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, help="Path to predictions file (Token Gold Pred)")
    parser.add_argument("--output_dir", type=str, default="analysis/figures", help="Dir to save plots")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Demo Data mimicking CoNLL tags
    labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    
    # Generate synthetic data for demonstration if no file provided
    import numpy as np
    y_true = np.random.choice(labels, 1000, p=[0.8, 0.05, 0.02, 0.05, 0.02, 0.03, 0.01, 0.01, 0.01])
    # Simulate some errors (e.g. PER <-> ORG confusion)
    y_pred = y_true.copy()
    mask = np.random.rand(1000) < 0.1 # 10% error rate
    y_pred[mask] = np.random.choice(labels, mask.sum())
    
    logger.info("Generating Confusion Matrix...")
    plot_confusion_matrix(y_true, y_pred, labels, output_dir / "confusion_matrix.png")
    
    # Identify Bad Cases (Confusion Analysis)
    # E.g., Most common confusion pairs
    # Create a DataFrame
    df = pd.DataFrame({'True': y_true, 'Pred': y_pred})
    errors = df[df['True'] != df['Pred']]
    confusion_counts = errors.groupby(['True', 'Pred']).size().sort_values(ascending=False).head(10)
    
    logger.info("Top 10 Confusions:")
    print(confusion_counts)
    
    confusion_counts.to_csv(output_dir / "top_confusions.csv")
    logger.info(f"Saved top confusions to {output_dir / 'top_confusions.csv'}")

if __name__ == "__main__":
    main()

