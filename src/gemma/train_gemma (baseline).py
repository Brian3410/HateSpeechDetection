import os
import pandas as pd
import torch
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, fbeta_score, roc_auc_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import time
import sys

# Set Gemma-specific cache directories under gemma/ subdirectory
os.environ["HF_HOME"] = "/scratch/ml23/bnge/gemma/hf_cache"
os.environ["HF_HUB_CACHE"] = "/scratch/ml23/bnge/gemma/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/scratch/ml23/bnge/gemma/hf_cache"
os.environ["XDG_CACHE_HOME"] = "/scratch/ml23/bnge/gemma/xdg_cache"
os.environ["MPLCONFIGDIR"] = "/scratch/ml23/bnge/gemma/matplotlib_cache"
os.environ["TORCH_HOME"] = "/scratch/ml23/bnge/gemma/torch_cache"
os.environ["TMPDIR"] = "/scratch/ml23/bnge/gemma/tmp"

# Create directories
cache_dirs = [
    "/scratch/ml23/bnge/gemma/hf_cache", 
    "/scratch/ml23/bnge/gemma/xdg_cache",
    "/scratch/ml23/bnge/gemma/matplotlib_cache",
    "/scratch/ml23/bnge/gemma/torch_cache",
    "/scratch/ml23/bnge/gemma/tmp"
]

for cache_dir in cache_dirs:
    os.makedirs(cache_dir, exist_ok=True)

print("Gemma cache directories created successfully under gemma/!")

# Load environment variables
load_dotenv()

def setup_gpu():
    """Setup GPU configuration and check availability."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()
        return device
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")
        return device

class Config:
    # Paths
    DATASET_PATH = "merged_reddit_gab.csv"
    MODEL_OUTPUT_DIR = "gemma_hate_speech_model"
    
    # Model settings
    MODEL_NAME = "google/gemma-7b"
    MAX_LENGTH = 256
    
    # Training settings - Consistent with DistilBERT baseline
    BATCH_SIZE = 8 if torch.cuda.is_available() else 4  # Reduced for 7B model
    LEARNING_RATE = 2e-4  # Higher for LoRA fine-tuning
    NUM_EPOCHS = 10 
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    GRADIENT_ACCUMULATION_STEPS = 4  # Increased to compensate for smaller batch
    FP16 = False
    BF16 = torch.cuda.is_available()
    
    # Advanced training settings
    LABEL_SMOOTHING = 0.1
    SCHEDULER_TYPE = "cosine"
    
    # Data preprocessing - Same as DistilBERT baseline
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # LoRA settings for efficient fine-tuning
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    
    # Quantization settings
    USE_4BIT = True
    BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
    BNB_4BIT_QUANT_TYPE = "nf4"
    USE_NESTED_QUANT = False

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
        
        if isinstance(row['label'], str):
            label = 1 if row['label'].lower() == 'hate' else 0
        else:
            label = int(row['label'])
        
        # Simple instruction format for Gemma
        formatted_text = f"Classify this text as hate speech or not: {text}"
        
        encoding = self.tokenizer(
            formatted_text,
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
    
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)

    f05 = fbeta_score(labels, preds, beta=0.5, average='binary', zero_division=0)
    f2 = fbeta_score(labels, preds, beta=2, average='binary', zero_division=0)
    
    try:
        probabilities = torch.softmax(torch.from_numpy(pred.predictions), dim=1).numpy()
        auc = roc_auc_score(labels, probabilities[:, 1])
    except ValueError:
        auc = 0.0
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
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
    print("📂 Loading dataset...")
    df = pd.read_csv(Config.DATASET_PATH)
    print(f"✅ Loaded {len(df)} samples")
    
    if df['label'].dtype == 'object':
        label_map = {'noHate': 0, 'hate': 1}
        df['label'] = df['label'].map(label_map)
        print("✅ Converted string labels to integers")
    
    label_counts = df['label'].value_counts()
    print(f"📊 Label distribution: {dict(label_counts)}")
    
    from sklearn.model_selection import train_test_split
    
    print("🔄 Splitting data...")
    train_df, temp_df = train_test_split(
        df, 
        test_size=(Config.VAL_SPLIT + Config.TEST_SPLIT),
        random_state=42,
        stratify=df['label']
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=Config.TEST_SPLIT / (Config.VAL_SPLIT + Config.TEST_SPLIT),
        random_state=42,
        stratify=temp_df['label']
    )
    
    print(f"📊 Training samples: {len(train_df)}")
    print(f"📊 Validation samples: {len(val_df)}")
    print(f"📊 Test samples: {len(test_df)}")
    
    for name, data in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        hate_count = len(data[data['label'] == 1])
        no_hate_count = len(data[data['label'] == 0])
        print(f"  {name} - Hate: {hate_count} ({hate_count/len(data)*100:.1f}%), "
              f"No Hate: {no_hate_count} ({no_hate_count/len(data)*100:.1f}%)")
    
    return train_df, val_df, test_df

def plot_training_history(trainer):
    """Plot training metrics."""
    try:
        logs = trainer.state.log_history
        train_loss = []
        eval_loss = []
        eval_f1 = []
        eval_accuracy = []
        
        for log in logs:
            if 'loss' in log and 'epoch' in log:
                train_loss.append(log['loss'])
            if 'eval_loss' in log:
                eval_loss.append(log['eval_loss'])
                eval_f1.append(log.get('eval_f1', 0))
                eval_accuracy.append(log.get('eval_accuracy', 0))
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        ax1.plot(train_loss)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        ax2.plot(eval_loss)
        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Evaluation Steps')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
        
        ax3.plot(eval_f1)
        ax3.set_title('Validation F1 Score')
        ax3.set_xlabel('Evaluation Steps')
        ax3.set_ylabel('F1 Score')
        ax3.grid(True)
        
        ax4.plot(eval_accuracy)
        ax4.set_title('Validation Accuracy')
        ax4.set_xlabel('Evaluation Steps')
        ax4.set_ylabel('Accuracy')
        ax4.grid(True)
        
        plt.tight_layout()
        pdf_path = os.path.join(Config.MODEL_OUTPUT_DIR, 'training_history.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        print("📊 Training history plots saved as PDF!")
        
    except Exception as e:
        print(f"⚠️ Could not create training plots: {e}")

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
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"📊 Confusion matrix saved to {save_path}")

def cleanup_output_directory():
    """Clean up output directory."""
    import shutil
    if os.path.exists(Config.MODEL_OUTPUT_DIR):
        try:
            print(f"🧹 Cleaning up existing output directory: {Config.MODEL_OUTPUT_DIR}")
            shutil.rmtree(Config.MODEL_OUTPUT_DIR)
            print("✅ Output directory cleaned successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not clean output directory: {e}")
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
        "scheduler_type": Config.SCHEDULER_TYPE,
        "fp16": Config.FP16,
        "bf16": Config.BF16,
        "train_split": Config.TRAIN_SPLIT,
        "val_split": Config.VAL_SPLIT,
        "test_split": Config.TEST_SPLIT,
        "lora_r": Config.LORA_R,
        "lora_alpha": Config.LORA_ALPHA,
        "use_4bit": Config.USE_4BIT,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    config_path = os.path.join(Config.MODEL_OUTPUT_DIR, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"💾 Training configuration saved to {config_path}")

def train_model():
    """Baseline training function for Gemma with LoRA (no SMOTE, no class weights)."""
    print("🚀 Starting Baseline Gemma-7B Training for Hate Speech Detection")
    print("=" * 80)
    
    device = setup_gpu()
    cleanup_output_directory()
    train_df, val_df, test_df = load_and_split_data()
    
    print(f"\n🤖 Loading model: {Config.MODEL_NAME}")
    
    # Configure quantization
    if Config.USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=Config.USE_4BIT,
            bnb_4bit_quant_type=Config.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=getattr(torch, Config.BNB_4BIT_COMPUTE_DTYPE),
            bnb_4bit_use_double_quant=Config.USE_NESTED_QUANT,
        )
    else:
        bnb_config = None
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, trust_remote_code=True)
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with quantization
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, 
        num_labels=2,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if Config.BF16 else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    print("✅ Applied LoRA configuration")
    model.print_trainable_parameters()
    
    train_dataset = HateSpeechDataset(train_df, tokenizer, max_length=Config.MAX_LENGTH)
    val_dataset = HateSpeechDataset(val_df, tokenizer, max_length=Config.MAX_LENGTH)
    test_dataset = HateSpeechDataset(test_df, tokenizer, max_length=Config.MAX_LENGTH)
    
    num_training_steps = (len(train_dataset) // (Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS)) * Config.NUM_EPOCHS
    warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)
    print(f"📊 Training steps: {num_training_steps}")
    print(f"📊 Warmup steps: {warmup_steps}")
    
    training_args = TrainingArguments(
        output_dir=Config.MODEL_OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE * 2,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        bf16=Config.BF16,
        fp16=Config.FP16,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        label_smoothing_factor=Config.LABEL_SMOOTHING,
        lr_scheduler_type=Config.SCHEDULER_TYPE,
        logging_dir=os.path.join(Config.MODEL_OUTPUT_DIR, 'logs'),
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=False,  # Consistent with DistilBERT baseline
        save_total_limit=2,
        save_safetensors=False,
        overwrite_output_dir=True,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        push_to_hub=False,
        report_to="none",
        max_grad_norm=1.0,
        seed=42,
        data_seed=42,
        optim="paged_adamw_32bit",
        group_by_length=True,
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Standard trainer (no class weights, no special sampling)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    
    print("🎯 Using standard Trainer (baseline - no imbalance handling)")
    
    save_training_config()
    
    print("\n🏃 Starting training...")
    print("-" * 40)
    
    try:
        training_result = trainer.train()
        print("✅ Training completed successfully!")
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
    
    print("\n" + "="*50)
    print("🧪 FINAL EVALUATION ON TEST SET")
    print("="*50)
    
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    print("\n📊 Test Set Results:")
    for key, value in test_results.items():
        if key.startswith('test_'):
            metric_name = key.replace('test_', '').replace('_', ' ').title()
            print(f"  {metric_name}: {value:.4f}")
    
    print("\n📋 Generating detailed evaluation...")
    predictions = trainer.predict(test_dataset)
    y_pred = predictions.predictions.argmax(-1)
    y_true = predictions.label_ids
    
    target_names = ['No Hate', 'Hate']
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print("\n📊 Detailed Classification Report:")
    print(report)
    
    report_path = os.path.join(Config.MODEL_OUTPUT_DIR, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("Gemma-7B Hate Speech Detection - Baseline Test Set Results\n")
        f.write("=" * 60 + "\n\n")
        f.write("Baseline Configuration (No SMOTE, No Class Weights)\n")
        f.write(f"- LoRA Fine-tuning: True\n")
        f.write(f"- 4-bit Quantization: {Config.USE_4BIT}\n\n")
        f.write(report)
        f.write("\n\nTest Set Metrics:\n")
        for key, value in test_results.items():
            if key.startswith('test_'):
                f.write(f"{key}: {value:.4f}\n")
    
    cm_path = os.path.join(Config.MODEL_OUTPUT_DIR, "confusion_matrix.pdf")
    plot_confusion_matrix(y_true, y_pred, cm_path)
    
    print(f"\n💾 Saving model to {Config.MODEL_OUTPUT_DIR}")
    trainer.save_model(Config.MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(Config.MODEL_OUTPUT_DIR)
    
    test_predictions_path = os.path.join(Config.MODEL_OUTPUT_DIR, "test_predictions.json")
    predictions_data = {
        "predictions": y_pred.tolist(),
        "true_labels": y_true.tolist(),
        "probabilities": torch.softmax(torch.from_numpy(predictions.predictions), dim=1).numpy().tolist()
    }
    
    with open(test_predictions_path, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("🎉 BASELINE GEMMA-7B TRAINING COMPLETED!")
    print("="*60)
    print(f"📁 Model and results saved in: {Config.MODEL_OUTPUT_DIR}")
    print(f"🎯 Test F1 Score: {test_results.get('test_f1', 0):.4f}")
    print(f"🎯 Test Accuracy: {test_results.get('test_accuracy', 0):.4f}")
    print(f"🔧 LoRA Fine-tuning: True")
    print(f"🔧 4-bit Quantization: {Config.USE_4BIT}")
    print(f"⚖️ Class Imbalance Handling: None (Baseline)")
    
    return model, tokenizer, test_results

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        Config.DATASET_PATH = sys.argv[1]
        Config.MODEL_OUTPUT_DIR = sys.argv[2]
    elif len(sys.argv) == 2:
        Config.DATASET_PATH = sys.argv[1]
        Config.MODEL_OUTPUT_DIR = "gemma_baseline_" + os.path.splitext(os.path.basename(Config.DATASET_PATH))[0] + "_model"
    else:
        print("Usage: python train_gemma_baseline.py <dataset_path> <output_dir>")
        sys.exit(1)
    model, tokenizer, results = train_model()