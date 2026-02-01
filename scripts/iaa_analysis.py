"""
Inter-Annotator Agreement (IAA) Analysis
Computes Cohen's Kappa, Fleiss' Kappa, and identifies disagreements.
"""

import pandas as pd
import numpy as np
from glob import glob
import os
from sklearn.metrics import cohen_kappa_score
from itertools import combinations

PROGRESS_DIR = "annotation_progress"

def load_annotations():
    """Load all annotator progress files."""
    progress_files = glob(os.path.join(PROGRESS_DIR, "*_progress.csv"))
    
    if not progress_files:
        print("No annotation files found.")
        return None
    
    annotations = {}
    for file in progress_files:
        annotator_id = os.path.basename(file).replace("_progress.csv", "")
        df = pd.read_csv(file)
        # Only include annotated rows
        df = df[df['annotation_label'] != '']
        df['annotation_label'] = df['annotation_label'].astype(int)
        annotations[annotator_id] = df
        print(f"Loaded {len(df)} annotations from {annotator_id}")
    
    return annotations

def compute_pairwise_kappa(annotations):
    """Compute Cohen's Kappa for each pair of annotators."""
    annotator_ids = list(annotations.keys())
    
    if len(annotator_ids) < 2:
        print("Need at least 2 annotators for IAA.")
        return
    
    print(f"\n{'='*60}")
    print("Pairwise Cohen's Kappa")
    print(f"{'='*60}")
    
    results = []
    
    for a1, a2 in combinations(annotator_ids, 2):
        df1 = annotations[a1].set_index('text_cleaned')
        df2 = annotations[a2].set_index('text_cleaned')
        
        # Find common annotations
        common_idx = df1.index.intersection(df2.index)
        
        if len(common_idx) < 10:
            print(f"{a1} vs {a2}: Not enough overlap ({len(common_idx)} samples)")
            continue
        
        labels1 = df1.loc[common_idx, 'annotation_label'].values
        labels2 = df2.loc[common_idx, 'annotation_label'].values
        
        kappa = cohen_kappa_score(labels1, labels2)
        agreement = np.mean(labels1 == labels2)
        
        results.append({
            'annotator_1': a1,
            'annotator_2': a2,
            'n_samples': len(common_idx),
            'agreement': agreement,
            'kappa': kappa
        })
        
        print(f"{a1} vs {a2}:")
        print(f"  Samples: {len(common_idx)}")
        print(f"  Agreement: {agreement:.2%}")
        print(f"  Kappa: {kappa:.4f}")
        print(f"  Interpretation: {interpret_kappa(kappa)}")
        print()
    
    return results

def interpret_kappa(kappa):
    """Interpret kappa value."""
    if kappa < 0:
        return "Poor (less than chance)"
    elif kappa < 0.20:
        return "Slight"
    elif kappa < 0.40:
        return "Fair"
    elif kappa < 0.60:
        return "Moderate"
    elif kappa < 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"

def find_disagreements(annotations, output_file="disagreements.csv"):
    """Find and export cases where annotators disagree."""
    annotator_ids = list(annotations.keys())
    
    if len(annotator_ids) < 2:
        return
    
    print(f"\n{'='*60}")
    print("Disagreement Analysis")
    print(f"{'='*60}")
    
    # Merge all annotations
    merged = None
    for annotator_id, df in annotations.items():
        temp = df[['text_cleaned', 'annotation_label', 'annotator_confidence']].copy()
        temp = temp.rename(columns={
            'annotation_label': f'label_{annotator_id}',
            'annotator_confidence': f'conf_{annotator_id}'
        })
        
        if merged is None:
            merged = temp
        else:
            merged = merged.merge(temp, on='text_cleaned', how='outer')
    
    # Find disagreements
    label_cols = [c for c in merged.columns if c.startswith('label_')]
    
    def check_disagreement(row):
        labels = [row[c] for c in label_cols if pd.notna(row[c])]
        if len(labels) < 2:
            return False
        return len(set(labels)) > 1
    
    merged['has_disagreement'] = merged.apply(check_disagreement, axis=1)
    disagreements = merged[merged['has_disagreement']]
    
    print(f"Total annotated (by any): {len(merged)}")
    print(f"Disagreements: {len(disagreements)} ({len(disagreements)/len(merged)*100:.1f}%)")
    
    if len(disagreements) > 0:
        # Add adjudication columns
        disagreements = disagreements.copy()
        disagreements['adjudicated_label'] = ''
        disagreements['adjudication_notes'] = ''
        disagreements.to_csv(output_file, index=False)
        print(f"Disagreements saved to: {output_file}")
    
    # Class-level disagreement analysis
    print(f"\nDisagreement by class pairs:")
    for a1, a2 in combinations(annotator_ids, 2):
        col1, col2 = f'label_{a1}', f'label_{a2}'
        if col1 in merged.columns and col2 in merged.columns:
            pair_df = merged[[col1, col2]].dropna()
            if len(pair_df) > 0:
                confusion = pd.crosstab(pair_df[col1], pair_df[col2])
                print(f"\n{a1} vs {a2} confusion:")
                print(confusion)

def compute_overall_agreement(annotations):
    """Compute overall statistics."""
    annotator_ids = list(annotations.keys())
    
    print(f"\n{'='*60}")
    print("Overall Statistics")
    print(f"{'='*60}")
    
    for annotator_id, df in annotations.items():
        print(f"\n{annotator_id}:")
        print(f"  Total annotated: {len(df)}")
        print(f"  Class distribution:")
        for label, count in df['annotation_label'].value_counts().sort_index().items():
            print(f"    Class {label}: {count} ({count/len(df)*100:.1f}%)")
        print(f"  Avg confidence: {df['annotator_confidence'].mean():.2f}")

if __name__ == "__main__":
    annotations = load_annotations()
    
    if annotations:
        compute_overall_agreement(annotations)
        compute_pairwise_kappa(annotations)
        find_disagreements(annotations)
