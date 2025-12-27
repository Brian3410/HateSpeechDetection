#!/usr/bin/env python3
"""
Example usage of the merged hate speech dataset.

This script demonstrates how to load and use the unified hate speech dataset
created by merge_dataset.py.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_merged_dataset(dataset_path):
    """Load the merged dataset."""
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded: {df.shape[0]:,} samples, {df.shape[1]} columns")
    return df

def explore_dataset(df):
    """Explore the dataset structure and content."""
    print("\n" + "="*50)
    print("DATASET EXPLORATION")
    print("="*50)
    
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nDataset shape: {df.shape}")
    
    print("\nLabel distribution:")
    print(df['label'].value_counts())
    
    print("\nSource distribution:")
    print(df['source'].value_counts())
    
    print("\nSample texts by label:")
    for label in df['label'].unique():
        print(f"\n{label.upper()} examples:")
        samples = df[df['label'] == label]['text'].head(3)
        for i, text in enumerate(samples, 1):
            text_preview = text[:100] + "..." if len(text) > 100 else text
            print(f"  {i}. {text_preview}")
    
    print("\nText length statistics:")
    df['text_length'] = df['text'].str.len()
    print(df['text_length'].describe())

def create_training_splits(df, test_size=0.1, val_size=0.1, random_state=42):
    """Create 80-10-10 train/validation/test splits."""
    from sklearn.model_selection import train_test_split
    
    # Calculate train size (should be 0.8 = 80%)
    train_size = 1.0 - test_size - val_size
    
    print(f"\nCreating splits:")
    print(f"  Training: {train_size*100:.0f}%")
    print(f"  Validation: {val_size*100:.0f}%") 
    print(f"  Test: {test_size*100:.0f}%")
    
    # First split: separate test set (10%)
    train_val, test = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['label']
    )
    
    # Second split: separate validation from training
    # val_size_adjusted accounts for the fact that we're splitting from the remaining 90%
    val_size_adjusted = val_size / (1.0 - test_size)  # 0.1 / 0.9 = 0.111...
    
    train, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=train_val['label']
    )
    
    print(f"\nActual dataset splits:")
    print(f"  Training: {len(train):,} samples ({len(train)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(val):,} samples ({len(val)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test):,} samples ({len(test)/len(df)*100:.1f}%)")
    
    # Verify label distribution in each split
    print(f"\nLabel distribution in splits:")
    print(f"Training - Hate: {(train['label'] == 'hate').sum()}, NoHate: {(train['label'] == 'noHate').sum()}")
    print(f"Validation - Hate: {(val['label'] == 'hate').sum()}, NoHate: {(val['label'] == 'noHate').sum()}")
    print(f"Test - Hate: {(test['label'] == 'hate').sum()}, NoHate: {(test['label'] == 'noHate').sum()}")
    
    return train, val, test

def save_splits(train, val, test, output_dir="training_data"):
    """Save the dataset splits to separate CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "validation.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    
    print(f"\nDataset splits saved to {output_dir}/")
    print(f"  train.csv: {len(train):,} samples")
    print(f"  validation.csv: {len(val):,} samples")
    print(f"  test.csv: {len(test):,} samples")

def analyze_by_source(df):
    """Analyze the dataset by source."""
    print("\n" + "="*50)
    print("SOURCE ANALYSIS")
    print("="*50)
    
    source_analysis = df.groupby(['source', 'label']).size().unstack(fill_value=0)
    print("\nSamples by source and label:")
    print(source_analysis)
    
    print("\nPercentage by source and label:")
    source_pct = source_analysis.div(source_analysis.sum(axis=1), axis=0) * 100
    print(source_pct.round(1))

def main():
    """Main function to demonstrate dataset usage."""
    # Load the balanced dataset
    dataset_path = "../merged_datasets/unified_hate_speech_dataset_balanced.csv"
    
    # Check if path exists, if not try alternative paths
    if not Path(dataset_path).exists():
        # Try from project root
        alt_path = "merged_datasets/unified_hate_speech_dataset_balanced.csv"
        if Path(alt_path).exists():
            dataset_path = alt_path
        else:
            print(f"Dataset not found at {dataset_path} or {alt_path}")
            print("Please run merge_dataset.py first to create the merged dataset.")
            return
    
    # Load and explore
    df = load_merged_dataset(dataset_path)
    explore_dataset(df)
    analyze_by_source(df)
    
    # Create 80-10-10 training splits
    train, val, test = create_training_splits(df, test_size=0.1, val_size=0.1)
    
    # Save splits for training
    save_splits(train, val, test)
    
    print("\n" + "="*50)
    print("READY FOR TRAINING!")
    print("="*50)
    print("Your merged dataset is ready for hate speech detection training.")
    print("The dataset includes:")
    print("- Text content from multiple sources")
    print("- Binary hate/noHate labels")
    print("- Source information for analysis")
    print("- Balanced classes for fair training")
    print("- 80-10-10 train/validation/test split")
    print("\nNext steps:")
    print("1. Use the training splits in training_data/ folder")
    print("2. Train your hate speech detection model")
    print("3. Evaluate on the validation set during training")
    print("4. Final evaluation on the test set")

if __name__ == "__main__":
    main()