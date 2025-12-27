import os
import pandas as pd
import torch
import argparse
from tqdm import tqdm  # Changed from tqdm.notebook for command-line usage
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define constants
MODEL_OUTPUT_DIR = os.path.join("hate_speech_model")  # Path to saved model

def classify_texts_in_folder(input_folder, output_excel=None, model=None, tokenizer=None):
    """
    Classify all text files in a folder and export results to Excel.
    
    Args:
        input_folder (str): Path to folder containing text files
        output_excel (str, optional): Path for output Excel file. Defaults to "classification_results.xlsx"
        model: Pre-trained model (loads from MODEL_OUTPUT_DIR if None)
        tokenizer: Tokenizer (loads from MODEL_OUTPUT_DIR if None)
    
    Returns:
        pd.DataFrame: Classification results
    """
    # Set default output file if not provided
    if output_excel is None:
        output_excel = "classification_results.xlsx"
    
    # Load model and tokenizer if not provided
    if model is None or tokenizer is None:
        print(f"Loading model from {MODEL_OUTPUT_DIR}...")
        try:
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_OUTPUT_DIR)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_OUTPUT_DIR)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure the model has been trained and saved correctly.")
            return None
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Verify input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist")
        return None
    
    # Get all text files in the folder
    text_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    
    if not text_files:
        print(f"No text files found in {input_folder}")
        return None
    
    print(f"Found {len(text_files)} text files to process")
    
    # Prepare results container
    results = []
    
    # Process each file
    for filename in tqdm(text_files, desc="Processing files"):
        file_path = os.path.join(input_folder, filename)
        
        try:
            # Read text file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding="max_length"
            )
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
            
            # Get prediction
            hate_prob = probabilities[1]
            predicted_class = 1 if hate_prob >= 0.5 else 0
            predicted_label = "hate" if predicted_class == 1 else "noHate"
            confidence = max(probabilities)
            
            # Add to results
            results.append({
                "filename": filename,
                "text": text[:100] + "..." if len(text) > 100 else text,  # Truncate long texts
                "prediction": predicted_label,
                "hate_probability": hate_prob,
                "confidence": confidence,
                "file_path": file_path
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            results.append({
                "filename": filename,
                "text": "ERROR: Could not process file",
                "prediction": "ERROR",
                "hate_probability": None,
                "confidence": None,
                "file_path": file_path,
                "error": str(e)
            })
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Sort by hate probability (descending)
    df_results = df_results.sort_values(by="hate_probability", ascending=False, na_position='last')
    
    # Export to Excel
    df_results.to_excel(output_excel, index=False)
    print(f"Results exported to {output_excel}")
    
    # Summary statistics
    if df_results["prediction"].value_counts().get("hate", 0) > 0:
        print("\nSummary:")
        print(f"Total files: {len(df_results)}")
        print(f"Hate speech detected: {df_results['prediction'].value_counts().get('hate', 0)} files")
        print(f"Non-hate content: {df_results['prediction'].value_counts().get('noHate', 0)} files")
        print(f"Error processing: {df_results['prediction'].value_counts().get('ERROR', 0)} files")
        
        # Print top 5 most hateful texts
        print("\nTop 5 texts with highest hate probability:")
        for i, row in df_results.head(5).iterrows():
            print(f"- {row['filename']}: {row['hate_probability']:.4f} - \"{row['text']}\"")
    
    return df_results

def main():
    """Command line interface for text classification"""
    parser = argparse.ArgumentParser(description='Classify text files for hate speech')
    parser.add_argument('input_folder', type=str, help='Path to folder containing text files')
    parser.add_argument('--output', '-o', type=str, default='classification_results.xlsx', 
                        help='Output Excel file path (default: classification_results.xlsx)')
    parser.add_argument('--model_dir', '-m', type=str, default=MODEL_OUTPUT_DIR,
                        help=f'Directory containing the model (default: {MODEL_OUTPUT_DIR})')
    
    args = parser.parse_args()
    
    # Update model directory if specified
    global MODEL_OUTPUT_DIR
    if args.model_dir != MODEL_OUTPUT_DIR:
        MODEL_OUTPUT_DIR = args.model_dir
    
    # Run classification
    results = classify_texts_in_folder(args.input_folder, args.output)
    if results is not None:
        print("Classification completed successfully.")
    else:
        print("Classification failed.")

if __name__ == "__main__":
    main()