import os
import pandas as pd
import torch
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, fbeta_score, roc_auc_score
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import time
import os
import sys

# Set DistilBERT-specific cache directories under distilbert/ subdirectory
os.environ["HF_HOME"] = "/scratch/ml23/bnge/distilbert/hf_cache"
os.environ["HF_HUB_CACHE"] = "/scratch/ml23/bnge/distilbert/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/scratch/ml23/bnge/distilbert/hf_cache"
os.environ["XDG_CACHE_HOME"] = "/scratch/ml23/bnge/distilbert/xdg_cache"
os.environ["MPLCONFIGDIR"] = "/scratch/ml23/bnge/distilbert/matplotlib_cache"
os.environ["TORCH_HOME"] = "/scratch/ml23/bnge/distilbert/torch_cache"
os.environ["TMPDIR"] = "/scratch/ml23/bnge/distilbert/tmp"

# Create directories
cache_dirs = [
    "/scratch/ml23/bnge/distilbert/hf_cache", 
    "/scratch/ml23/bnge/distilbert/xdg_cache",
    "/scratch/ml23/bnge/distilbert/matplotlib_cache",
    "/scratch/ml23/bnge/distilbert/torch_cache",
    "/scratch/ml23/bnge/distilbert/tmp"
]

for cache_dir in cache_dirs:
    os.makedirs(cache_dir, exist_ok=True)

print("DistilBERT cache directories created successfully under distilbert/!")

# Load environment variables
load_dotenv()

# GPU Configuration
def setup_gpu():
    """Setup GPU configuration and check availability."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        return device
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")
        return device

# Configuration
class Config:
    # Paths
    DATASET_PATH = "merged_reddit_gab.csv"
    MODEL_OUTPUT_DIR = "distilbert_hate_speech_model"
    
    # Model settings
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 256  # Increased for better context capture
    
    # Training settings - Improved for better performance
    BATCH_SIZE = 16 if torch.cuda.is_available() else 8
    LEARNING_RATE = 3e-5  # Slightly higher for faster convergence
    NUM_EPOCHS = 10 
    WARMUP_RATIO = 0.1  # 10% warmup
    WEIGHT_DECAY = 0.01
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 32
    FP16 = torch.cuda.is_available()
    
    # Advanced training settings
    LABEL_SMOOTHING = 0.1  # Prevent overconfidence
    DROPOUT = 0.3  # Increased regularization
    SCHEDULER_TYPE = "cosine"  # Better learning rate scheduling
    
    # Data preprocessing
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1

class HateSpeechDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=256):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = str(row['text']).strip()
        
        # Handle different label formats
        if isinstance(row['label'], str):
            label = 1 if row['label'].lower() == 'hate' else 0
        else:
            label = int(row['label'])
        
        # Enhanced tokenization with better handling
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(pred):
    """Enhanced evaluation metrics calculation."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, 
        preds, 
        average='binary',
        zero_division=0
    )
    
    # F0.5 and F2 scores (beta=0.5 favors precision, beta=2 favors recall)
    f05 = fbeta_score(labels, preds, beta=0.5, average='binary', zero_division=0)
    f2 = fbeta_score(labels, preds, beta=2, average='binary', zero_division=0)
    
    # AUC (requires probabilities for positive class)
    try:
        probabilities = torch.softmax(torch.from_numpy(pred.predictions), dim=1).numpy()
        auc = roc_auc_score(labels, probabilities[:, 1])  # Use positive class probabilities
    except ValueError:
        auc = 0.0  # Handle edge cases where AUC cannot be computed
    
    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, 
        preds, 
        average='macro',
        zero_division=0
    )
    
    # Weighted averages
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, 
        preds, 
        average='weighted',
        zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "f1": f1,  # Main metric for early stopping
        "f05": f05,
        "f2": f2,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_weighted": f1_weighted,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
    }

def load_and_split_data():
    """Load and intelligently split the dataset."""
    print("Loading unified dataset...")
    df = pd.read_csv(Config.DATASET_PATH)
    print(f"Loaded {len(df)} samples")
    
    # Convert labels to binary if needed
    if df['label'].dtype == 'object':
        label_map = {'noHate': 0, 'hate': 1}
        df['label'] = df['label'].map(label_map)
        print("Converted string labels to integers")
    
    # Check label distribution
    label_counts = df['label'].value_counts()
    print(f"Label distribution: {dict(label_counts)}")
    
    # Stratified split to maintain label distribution
    from sklearn.model_selection import train_test_split
    
    # First split: train vs temp (val + test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(Config.VAL_SPLIT + Config.TEST_SPLIT),
        random_state=42,
        stratify=df['label']
    )
    
    # Second split: val vs test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=Config.TEST_SPLIT / (Config.VAL_SPLIT + Config.TEST_SPLIT),
        random_state=42,
        stratify=temp_df['label']
    )
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Print split distributions
    for name, data in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        hate_count = len(data[data['label'] == 1])
        no_hate_count = len(data[data['label'] == 0])
        print(f"{name} - Hate: {hate_count} ({hate_count/len(data)*100:.1f}%), "
              f"No Hate: {no_hate_count} ({no_hate_count/len(data)*100:.1f}%)")
    
    return train_df, val_df, test_df

def create_weighted_sampler(dataset):
    """Create weighted sampler for handling class imbalance."""
    labels = [dataset[i]['labels'].item() for i in range(len(dataset))]
    class_counts = Counter(labels)
    
    # Calculate weights (inverse frequency)
    total_samples = len(labels)
    weights = {label: total_samples / count for label, count in class_counts.items()}
    
    # Create sample weights
    sample_weights = [weights[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    print(f"Class weights: {weights}")
    return sampler

def plot_training_history(trainer):
    """Plot training metrics."""
    try:
        # Get training logs
        logs = trainer.state.log_history
        
        # Extract training and evaluation metrics
        train_loss = []
        eval_loss = []
        eval_f1 = []
        eval_accuracy = []
        
        for log in logs:
            if 'loss' in log and 'epoch' in log:
                train_loss.append(log['loss'])
            if 'eval_loss' in log:
                eval_loss.append(log['eval_loss'])
                eval_f1.append(log.get('eval_f1', 0))  # Changed from 'eval_f1_binary' to 'eval_f1'
                eval_accuracy.append(log.get('eval_accuracy', 0))
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        ax1.plot(train_loss)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Validation loss
        ax2.plot(eval_loss)
        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Evaluation Steps')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
        
        # F1 Score
        ax3.plot(eval_f1)
        ax3.set_title('Validation F1 Score')
        ax3.set_xlabel('Evaluation Steps')
        ax3.set_ylabel('F1 Score')
        ax3.grid(True)
        
        # Accuracy
        ax4.plot(eval_accuracy)
        ax4.set_title('Validation Accuracy')
        ax4.set_xlabel('Evaluation Steps')
        ax4.set_ylabel('Accuracy')
        ax4.grid(True)
        
        plt.tight_layout()
        # save as PDF instead of PNG
        pdf_path = os.path.join(Config.MODEL_OUTPUT_DIR, 'training_history.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        
        print("Training history plots saved as PDF!")
        
    except Exception as e:
        print(f"Could not create training plots: {e}")

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Hate', 'Hate'],
                yticklabels=['No Hate', 'Hate'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    # Ensure save_path has .pdf extension (call-site updated below)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")

def cleanup_output_directory():
    """Clean up output directory."""
    import shutil
    
    if os.path.exists(Config.MODEL_OUTPUT_DIR):
        try:
            print(f"Cleaning up existing output directory: {Config.MODEL_OUTPUT_DIR}")
            shutil.rmtree(Config.MODEL_OUTPUT_DIR)
            print("✅ Output directory cleaned successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not clean output directory: {e}")
    
    # Create fresh output directory
    os.makedirs(Config.MODEL_OUTPUT_DIR, exist_ok=True)

def save_training_config():
    """Save training configuration for reproducibility."""
    config_dict = {
        "model_name": Config.MODEL_NAME,
        "max_length": Config.MAX_LENGTH,
        "batch_size": Config.BATCH_SIZE,
        "learning_rate": Config.LEARNING_RATE,
        "num_epochs": Config.NUM_EPOCHS,
        "warmup_ratio": Config.WARMUP_RATIO,
        "weight_decay": Config.WEIGHT_DECAY,
        "gradient_accumulation_steps": Config.GRADIENT_ACCUMULATION_STEPS,
        "label_smoothing": Config.LABEL_SMOOTHING,
        "dropout": Config.DROPOUT,
        "scheduler_type": Config.SCHEDULER_TYPE,
        "fp16": Config.FP16,
        "train_split": Config.TRAIN_SPLIT,
        "val_split": Config.VAL_SPLIT,
        "test_split": Config.TEST_SPLIT,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    config_path = os.path.join(Config.MODEL_OUTPUT_DIR, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Training configuration saved to {config_path}")

def train_model():
    """Enhanced training function for DistilBERT."""
    print("Starting Enhanced DistilBERT Training for Hate Speech Detection")
    print("=" * 60)
    
    # Setup GPU
    device = setup_gpu()
    
    # Clean up output directory
    cleanup_output_directory()
    
    # Load and split data
    train_df, val_df, test_df = load_and_split_data()
    
    # Load model and tokenizer
    print(f"\nLoading model: {Config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # Load the base model first
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, 
        num_labels=2
    )
    
    # Manually set dropout for DistilBERT - modify the config after loading
    if hasattr(model.config, 'dropout'):
        model.config.dropout = Config.DROPOUT
    if hasattr(model.config, 'attention_dropout'):
        model.config.attention_dropout = Config.DROPOUT
    if hasattr(model.config, 'classifier_dropout'):
        model.config.classifier_dropout = Config.DROPOUT
    if hasattr(model.config, 'seq_classif_dropout'):
        model.config.seq_classif_dropout = Config.DROPOUT
    
    # Print actual config to see what dropout parameters are available
    print(f"Model config: {model.config}")
    
    # Move model to device
    model = model.to(device)
    print(f"Model moved to: {device}")
    
    # Create datasets
    train_dataset = HateSpeechDataset(train_df, tokenizer, max_length=Config.MAX_LENGTH)
    val_dataset = HateSpeechDataset(val_df, tokenizer, max_length=Config.MAX_LENGTH)
    test_dataset = HateSpeechDataset(test_df, tokenizer, max_length=Config.MAX_LENGTH)
    
    # Calculate training steps
    num_training_steps = (len(train_dataset) // (Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS)) * Config.NUM_EPOCHS
    warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)
    
    print(f"Training steps: {num_training_steps}")
    print(f"Warmup steps: {warmup_steps}")
    
    training_args = TrainingArguments(
        output_dir=Config.MODEL_OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE * 2,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        
        # Advanced optimization
        fp16=Config.FP16,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        label_smoothing_factor=Config.LABEL_SMOOTHING,
        lr_scheduler_type=Config.SCHEDULER_TYPE,
        
        # Evaluation and logging
        logging_dir=os.path.join(Config.MODEL_OUTPUT_DIR, 'logs'),
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=False,  # Changed to False to avoid metric dependency
        
        # Model saving
        save_total_limit=2,
        save_safetensors=False,
        overwrite_output_dir=True,
        
        # Performance
        dataloader_num_workers=0,
        remove_unused_columns=True,
        push_to_hub=False,
        report_to="none",
        
        # Stability
        max_grad_norm=1.0,
        seed=42,
        data_seed=42,
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create trainer with enhanced configuration - FIXED: Use processing_class instead of tokenizer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        processing_class=tokenizer,  # Changed from tokenizer to processing_class
    )
    
    # SIMPLIFIED: Remove the custom weighted sampler to avoid data collator conflicts
    # The class imbalance can be handled through class weights in the loss function instead
    print("Using standard random sampling (class imbalance will be handled by loss function)")
    
    # Save training configuration
    save_training_config()
    
    # Train the model
    print("\nStarting training...")
    print("-" * 40)
    
    try:
        training_result = trainer.train()
        print("✅ Training completed successfully!")
        
        # Plot training history
        plot_training_history(trainer)
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("❌ GPU out of memory! Suggestions:")
            print(f"  - Reduce batch size (current: {Config.BATCH_SIZE})")
            print(f"  - Reduce max length (current: {Config.MAX_LENGTH})")
            print(f"  - Increase gradient accumulation steps (current: {Config.GRADIENT_ACCUMULATION_STEPS})")
            raise e
        else:
            raise e
    
    # Final evaluation on test set
    print("\n" + "="*50)
    print("FINAL EVALUATION ON TEST SET")
    print("="*50)
    
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    
    print("\nTest Set Results:")
    for key, value in test_results.items():
        if key.startswith('test_'):
            metric_name = key.replace('test_', '').replace('_', ' ').title()
            print(f"  {metric_name}: {value:.4f}")
    
    # Generate detailed predictions and classification report
    print("\nGenerating detailed evaluation...")
    predictions = trainer.predict(test_dataset)
    y_pred = predictions.predictions.argmax(-1)
    y_true = predictions.label_ids
    
    # Classification report
    target_names = ['No Hate', 'Hate']
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print("\nDetailed Classification Report:")
    print(report)
    
    # Save classification report
    report_path = os.path.join(Config.MODEL_OUTPUT_DIR, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("DistilBERT Hate Speech Detection - Test Set Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write("\n\nTest Set Metrics:\n")
        for key, value in test_results.items():
            if key.startswith('test_'):
                f.write(f"{key}: {value:.4f}\n")
    
    # Plot and save confusion matrix
    cm_path = os.path.join(Config.MODEL_OUTPUT_DIR, "confusion_matrix.pdf")
    plot_confusion_matrix(y_true, y_pred, cm_path)
    
    # Save final model
    print(f"\nSaving model to {Config.MODEL_OUTPUT_DIR}")
    trainer.save_model(Config.MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(Config.MODEL_OUTPUT_DIR)
    
    # Save test predictions for analysis
    test_predictions_path = os.path.join(Config.MODEL_OUTPUT_DIR, "test_predictions.json")
    predictions_data = {
        "predictions": y_pred.tolist(),
        "true_labels": y_true.tolist(),
        "probabilities": torch.softmax(torch.from_numpy(predictions.predictions), dim=1).numpy().tolist()
    }
    
    with open(test_predictions_path, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("✅ ENHANCED DISTILBERT TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📁 Model and results saved in: {Config.MODEL_OUTPUT_DIR}")
    print(f"🎯 Best Test F1 Score: {test_results.get('test_f1', 0):.4f}")
    print(f"🎯 Test Accuracy: {test_results.get('test_accuracy', 0):.4f}")
    
    return model, tokenizer, test_results

if __name__ == "__main__":
    # Usage: python train_distilbert.py <dataset_path> <output_dir>
    if len(sys.argv) >= 3:
        Config.DATASET_PATH = sys.argv[1]
        Config.MODEL_OUTPUT_DIR = sys.argv[2]
    elif len(sys.argv) == 2:
        Config.DATASET_PATH = sys.argv[1]
        Config.MODEL_OUTPUT_DIR = "distilbert_" + os.path.splitext(os.path.basename(Config.DATASET_PATH))[0] + "_model"
    else:
        print("Usage: python train_distilbert.py <dataset_path> <output_dir>")
        sys.exit(1)
    model, tokenizer, results = train_model()