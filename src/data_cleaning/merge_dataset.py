#!/usr/bin/env python3
"""
Merge multiple hate speech datasets into a unified format.

This script combines datasets from:
1. Stormfront (text files + annotations)
2. Gab/Reddit (CSV format)
3. Hate Corpus (CSV format with implicit/explicit hate labels)

Output: A unified CSV file with columns: text, label, source, original_label
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

class DatasetMerger:
    def __init__(self, datasets_dir="datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.merged_data = []
        
    def load_stormfront_data(self):
        """Load Stormfront dataset (text files + annotations)."""
        print("Loading Stormfront dataset...")
        
        # Load annotations
        annotations_path = self.datasets_dir / "stormfront" / "annotations_metadata.csv"
        all_files_path = self.datasets_dir / "stormfront" / "all_files"
        
        if not annotations_path.exists():
            print(f"Warning: Stormfront annotations not found at {annotations_path}")
            return []
        
        annotations = pd.read_csv(annotations_path)
        stormfront_data = []
        
        for _, row in tqdm(annotations.iterrows(), total=len(annotations), desc="Processing Stormfront"):
            file_id = row['file_id']
            label = row['label']
            
            # Read text content
            text_file = all_files_path / f"{file_id}.txt"
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                stormfront_data.append({
                    'text': text,
                    'label': 'hate' if label == 'hate' else 'noHate',
                    'source': 'stormfront',
                    'original_label': label,
                    'file_id': file_id
                })
            except FileNotFoundError:
                print(f"Warning: File not found: {text_file}")
            except Exception as e:
                print(f"Error reading {text_file}: {e}")
        
        print(f"Loaded {len(stormfront_data)} samples from Stormfront")
        return stormfront_data
    
    def load_gab_reddit_data(self):
        """Load Gab/Reddit dataset."""
        print("Loading Gab/Reddit dataset...")
        
        gab_reddit_path = self.datasets_dir / "gab_reddit" / "cleaned_gab_reddit.csv"
        
        if not gab_reddit_path.exists():
            print(f"Warning: Gab/Reddit dataset not found at {gab_reddit_path}")
            return []
        
        df = pd.read_csv(gab_reddit_path)
        gab_reddit_data = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Gab/Reddit"):
            text = str(row['text']).strip()
            classification = int(row['Classification'])  # Changed from 'label' to 'Classification'
            
            if text and text != 'nan':
                gab_reddit_data.append({
                    'text': text,
                    'label': 'hate' if classification == 1 else 'noHate',
                    'source': 'gab_reddit',
                    'original_label': classification,
                    'file_id': None
                })
        
        print(f"Loaded {len(gab_reddit_data)} samples from Gab/Reddit")
        return gab_reddit_data
    
    def load_hate_corpus_data(self):
        """Load Hate Corpus dataset (implicit/explicit hate)."""
        print("Loading Hate Corpus dataset...")
        
        hate_corpus_path = self.datasets_dir / "hate_corpus"
        hate_corpus_data = []
        
        # Load different stages of the hate corpus
        corpus_files = [
            "implicit_hate_v1_stg1_posts.csv",
            "implicit_hate_v1_stg2_posts.csv", 
            "implicit_hate_v1_stg3_posts.csv"
        ]
        
        for file_name in corpus_files:
            file_path = hate_corpus_path / file_name
            
            if not file_path.exists():
                print(f"Warning: {file_name} not found")
                continue
            
            df = pd.read_csv(file_path)
            stage = file_name.split('_')[3]  # Extract stage number
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing Hate Corpus {stage}"):
                text = str(row['post']).strip() if 'post' in row else str(row.iloc[0]).strip()
                
                # Determine label based on the file and class column
                if 'class' in row:
                    original_label = row['class']
                elif 'implicit_class' in row:
                    original_label = row['implicit_class']
                else:
                    original_label = 'not_hate'  # Default for stage 3
                
                # Convert to binary hate/noHate
                if original_label in ['implicit_hate', 'explicit_hate']:
                    label = 'hate'
                else:
                    label = 'noHate'
                
                if text and text != 'nan':
                    hate_corpus_data.append({
                        'text': text,
                        'label': label,
                        'source': f'hate_corpus_{stage}',
                        'original_label': original_label,
                        'file_id': None
                    })
        
        print(f"Loaded {len(hate_corpus_data)} samples from Hate Corpus")
        return hate_corpus_data
    
    def merge_all_datasets(self):
        """Merge all datasets into a single dataframe."""
        print("Merging all datasets...")
        
        # Load each dataset
        stormfront_data = self.load_stormfront_data()
        gab_reddit_data = self.load_gab_reddit_data()
        hate_corpus_data = self.load_hate_corpus_data()
        
        # Combine all data
        all_data = stormfront_data + gab_reddit_data + hate_corpus_data
        
        if not all_data:
            print("No data loaded from any dataset!")
            return pd.DataFrame()
        
        # Create dataframe
        df = pd.DataFrame(all_data)
        
        # Clean text data
        df['text'] = df['text'].astype(str)
        df['text'] = df['text'].str.strip()
        
        # Remove empty or very short texts
        df = df[df['text'].str.len() > 5]
        
        # Remove duplicates based on text content
        initial_count = len(df)
        df = df.drop_duplicates(subset=['text'], keep='first')
        print(f"Removed {initial_count - len(df)} duplicate texts")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def get_dataset_statistics(self, df):
        """Print statistics about the merged dataset."""
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        
        print(f"Total samples: {len(df):,}")
        print(f"Total unique texts: {df['text'].nunique():,}")
        
        print("\nLabel distribution:")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {label}: {count:,} ({percentage:.1f}%)")
        
        print("\nSource distribution:")
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {source}: {count:,} ({percentage:.1f}%)")
        
        print("\nOriginal label distribution:")
        original_label_counts = df['original_label'].value_counts()
        for label, count in original_label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {label}: {count:,} ({percentage:.1f}%)")
        
        print("\nText length statistics:")
        df['text_length'] = df['text'].str.len()
        print(f"  Mean length: {df['text_length'].mean():.1f} characters")
        print(f"  Median length: {df['text_length'].median():.1f} characters")
        print(f"  Min length: {df['text_length'].min()} characters")
        print(f"  Max length: {df['text_length'].max()} characters")
    
    def save_merged_dataset(self, df, output_path="merged_hate_speech_dataset.csv"):
        """Save the merged dataset to CSV."""
        output_path = Path(output_path)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\nMerged dataset saved to: {output_path}")
        print(f"File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Also save a sample for inspection
        sample_path = output_path.parent / f"sample_{output_path.name}"
        df.sample(min(1000, len(df))).to_csv(sample_path, index=False, encoding='utf-8')
        print(f"Sample dataset saved to: {sample_path}")
    
    def create_balanced_subset(self, df, max_samples_per_class=None):
        """Create a balanced subset of the dataset."""
        if max_samples_per_class is None:
            # Use the minority class size
            class_counts = df['label'].value_counts()
            max_samples_per_class = class_counts.min()
        
        print(f"\nCreating balanced subset with {max_samples_per_class} samples per class...")
        
        balanced_data = []
        for label in df['label'].unique():
            class_data = df[df['label'] == label].sample(
                n=min(max_samples_per_class, len(df[df['label'] == label])),
                random_state=42
            )
            balanced_data.append(class_data)
        
        balanced_df = pd.concat(balanced_data, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Balanced dataset created with {len(balanced_df)} samples")
        return balanced_df

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Merge hate speech datasets')
    parser.add_argument('--datasets_dir', '-d', type=str, default='../datasets',
                        help='Directory containing datasets (default: ../datasets)')
    parser.add_argument('--output', '-o', type=str, default='merged_hate_speech_dataset.csv',
                        help='Output CSV file path')
    parser.add_argument('--balanced', '-b', action='store_true',
                        help='Create balanced subset of the dataset')
    parser.add_argument('--max_samples', '-m', type=int,
                        help='Maximum samples per class for balanced subset')
    parser.add_argument('--sample_only', '-s', action='store_true',
                        help='Create only a sample dataset (1000 samples)')
    parser.add_argument('--save_individual', '-i', action='store_true',
                        help='Save each dataset individually (e.g., merged_reddit_gab.csv, merged_corpus.csv, merged_stormfront.csv)')
    
    args = parser.parse_args()
    
    # Initialize merger
    merger = DatasetMerger(args.datasets_dir)

    # Load each dataset separately for individual saving
    stormfront_data = merger.load_stormfront_data()
    gab_reddit_data = merger.load_gab_reddit_data()
    hate_corpus_data = merger.load_hate_corpus_data()

    # Convert to DataFrames for individual saving
    stormfront_df = pd.DataFrame(stormfront_data) if stormfront_data else pd.DataFrame()
    gab_reddit_df = pd.DataFrame(gab_reddit_data) if gab_reddit_data else pd.DataFrame()
    hate_corpus_df = pd.DataFrame(hate_corpus_data) if hate_corpus_data else pd.DataFrame()

    # Save individual datasets if requested
    if args.save_individual:
        print("\nSaving individual datasets...")
        if not stormfront_df.empty:
            merger.save_merged_dataset(stormfront_df, "merged_stormfront.csv")
        if not gab_reddit_df.empty:
            merger.save_merged_dataset(gab_reddit_df, "merged_reddit_gab.csv")
        if not hate_corpus_df.empty:
            merger.save_merged_dataset(hate_corpus_df, "merged_corpus.csv")
    
    # Merge all datasets for full merge
    all_data = stormfront_data + gab_reddit_data + hate_corpus_data
    if not all_data:
        print("No data loaded from any dataset! Exiting.")
        return
    
    merged_df = pd.DataFrame(all_data)
    merged_df['text'] = merged_df['text'].astype(str).str.strip()
    merged_df = merged_df[merged_df['text'].str.len() > 5]
    initial_count = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['text'], keep='first')
    print(f"Removed {initial_count - len(merged_df)} duplicate texts")
    merged_df = merged_df.reset_index(drop=True)
    
    # Show statistics
    merger.get_dataset_statistics(merged_df)
    
    # Create balanced subset if requested
    if args.balanced:
        balanced_df = merger.create_balanced_subset(merged_df, args.max_samples)
        merger.get_dataset_statistics(balanced_df)
        
        # Save balanced dataset
        balanced_output = args.output.replace('.csv', '_balanced.csv')
        merger.save_merged_dataset(balanced_df, balanced_output)
    
    # Create sample dataset if requested
    if args.sample_only:
        sample_df = merged_df.sample(min(1000, len(merged_df)), random_state=42)
        sample_output = args.output.replace('.csv', '_sample.csv')
        merger.save_merged_dataset(sample_df, sample_output)
    else:
        # Save full dataset
        merger.save_merged_dataset(merged_df, args.output)

if __name__ == "__main__":
    main()
