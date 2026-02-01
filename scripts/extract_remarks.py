"""
Extract emails with annotator remarks for chief annotator review.
Merges remarks from all annotators for adjudication.
"""

import pandas as pd
import os
from glob import glob

PROGRESS_DIR = "annotation_progress"
OUTPUT_FILE = "emails_for_review.csv"

def extract_emails_with_remarks():
    progress_files = glob(os.path.join(PROGRESS_DIR, "*_progress.csv"))
    
    if not progress_files:
        print("No annotation progress files found.")
        return
    
    print(f"Found {len(progress_files)} annotator file(s)")
    
    all_remarks = []
    
    for file in progress_files:
        annotator_id = os.path.basename(file).replace("_progress.csv", "")
        df = pd.read_csv(file)
        
        # Filter rows with remarks
        has_remarks = df['annotator_remarks'].notna() & (df['annotator_remarks'] != '')
        remarks_df = df[has_remarks].copy()
        
        if len(remarks_df) > 0:
            remarks_df['annotator_id'] = annotator_id
            all_remarks.append(remarks_df)
            print(f"  {annotator_id}: {len(remarks_df)} emails with remarks")
    
    if not all_remarks:
        print("No emails with remarks found.")
        return
    
    # Combine all remarks
    combined = pd.concat(all_remarks, ignore_index=True)
    
    # Group by email (using text_cleaned as key) and aggregate remarks
    review_df = combined.groupby('text_cleaned').agg({
        'source_dataset': 'first',
        'text_length': 'first',
        'annotation_label': lambda x: list(x),
        'annotator_confidence': lambda x: list(x),
        'annotator_remarks': lambda x: ' | '.join([f"[{combined.loc[i, 'annotator_id']}] {r}" 
                                                    for i, r in zip(x.index, x) if pd.notna(r) and r != '']),
        'annotator_id': lambda x: list(x)
    }).reset_index()
    
    review_df.columns = ['text_cleaned', 'source_dataset', 'text_length', 
                         'original_labels', 'original_confidence', 
                         'all_remarks', 'annotators']
    
    # Add columns for chief annotator
    review_df['chief_label'] = ''
    review_df['chief_confidence'] = ''
    review_df['chief_notes'] = ''
    
    # Save
    review_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(review_df)} emails for review to: {OUTPUT_FILE}")
    print("\nColumns for chief annotator:")
    print("  - chief_label: Final class decision")
    print("  - chief_confidence: Confidence in decision")
    print("  - chief_notes: Resolution notes")

if __name__ == "__main__":
    extract_emails_with_remarks()
