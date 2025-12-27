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
import spacy
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

# Load spaCy model for POS tagging
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    raise

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
    MODEL_OUTPUT_DIR = "gemma_pos_only_model"
    
    # Model settings
    MODEL_NAME = "google/gemma-7b"
    MAX_LENGTH = 256
    
    # Training settings - Reduced for 7B model
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
    
    # Data preprocessing
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # POS Tagging settings only - no SMOTE or class weights
    USE_POS_FEATURES = True
    POS_FEATURE_DIM = 64  # Dimension for POS feature embeddings
    
    # LoRA settings for efficient fine-tuning
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    
    # Quantization settings
    USE_4BIT = True
    BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
    BNB_4BIT_QUANT_TYPE = "nf4"
    USE_NESTED_QUANT = False

def extract_pos_features(text):
    """Extract POS tag features from text using spaCy."""
    try:
        doc = nlp(text)
        pos_tags = [token.pos_ for token in doc]
        
        # Create POS tag statistics
        pos_counts = Counter(pos_tags)
        total_tokens = len(pos_tags)
        
        if total_tokens == 0:
            return {
                'noun_ratio': 0.0, 'verb_ratio': 0.0, 'adj_ratio': 0.0,
                'adv_ratio': 0.0, 'pronoun_ratio': 0.0, 'intj_ratio': 0.0,
                'profanity_ratio': 0.0, 'caps_ratio': 0.0
            }
        
        # Calculate linguistic ratios
        features = {
            'noun_ratio': pos_counts.get('NOUN', 0) / total_tokens,
            'verb_ratio': pos_counts.get('VERB', 0) / total_tokens,
            'adj_ratio': pos_counts.get('ADJ', 0) / total_tokens,
            'adv_ratio': pos_counts.get('ADV', 0) / total_tokens,
            'pronoun_ratio': pos_counts.get('PRON', 0) / total_tokens,
            'intj_ratio': pos_counts.get('INTJ', 0) / total_tokens,  # Interjections often in hate speech
        }
        
        # Additional linguistic features
        features['profanity_ratio'] = sum(1 for token in doc if token.is_stop) / total_tokens
        features['caps_ratio'] = sum(1 for token in doc if token.text.isupper()) / total_tokens
        
        return features
        
    except Exception as e:
        print(f"Error in POS tagging: {e}")
        return {
            'noun_ratio': 0.0, 'verb_ratio': 0.0, 'adj_ratio': 0.0,
            'adv_ratio': 0.0, 'pronoun_ratio': 0.0, 'intj_ratio': 0.0,
            'profanity_ratio': 0.0, 'caps_ratio': 0.0
        }

class GemmaWithPOS(nn.Module):
    """Gemma model with POS feature integration."""
    
    def __init__(self, model, pos_feature_dim=64):
        super().__init__()
        self.gemma = model
        self.config = model.config
        
        # POS feature processing
        self.pos_feature_dim = 8  # Number of POS features
        self.pos_projection = nn.Linear(self.pos_feature_dim, pos_feature_dim)
        self.pos_dropout = nn.Dropout(0.1)
        
        # Get the original classifier
        original_classifier = self.gemma.classifier if hasattr(self.gemma, 'classifier') else self.gemma.score
        
        # Combined classifier
        gemma_hidden_size = self.config.hidden_size
        self.combined_classifier = nn.Sequential(
            nn.Linear(gemma_hidden_size + pos_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.config.num_labels)
        )
        
        # Remove original classifier
        if hasattr(self.gemma, 'classifier'):
            self.gemma.classifier = nn.Identity()
        elif hasattr(self.gemma, 'score'):
            self.gemma.score = nn.Identity()
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the underlying Gemma model."""
        if hasattr(self.gemma, 'gradient_checkpointing_enable'):
            self.gemma.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        elif hasattr(self.gemma.base_model, 'gradient_checkpointing_enable'):
            self.gemma.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the underlying Gemma model."""
        if hasattr(self.gemma, 'gradient_checkpointing_disable'):
            self.gemma.gradient_checkpointing_disable()
        elif hasattr(self.gemma.base_model, 'gradient_checkpointing_disable'):
            self.gemma.base_model.gradient_checkpointing_disable()
    
    def get_input_embeddings(self):
        """Get input embeddings from the underlying Gemma model."""
        if hasattr(self.gemma, 'get_input_embeddings'):
            return self.gemma.get_input_embeddings()
        elif hasattr(self.gemma.base_model, 'get_input_embeddings'):
            return self.gemma.base_model.get_input_embeddings()
        return None
    
    def set_input_embeddings(self, embeddings):
        """Set input embeddings for the underlying Gemma model."""
        if hasattr(self.gemma, 'set_input_embeddings'):
            self.gemma.set_input_embeddings(embeddings)
        elif hasattr(self.gemma.base_model, 'set_input_embeddings'):
            self.gemma.base_model.set_input_embeddings(embeddings)
    
    def resize_token_embeddings(self, new_num_tokens):
        """Resize token embeddings for the underlying Gemma model."""
        if hasattr(self.gemma, 'resize_token_embeddings'):
            return self.gemma.resize_token_embeddings(new_num_tokens)
        elif hasattr(self.gemma.base_model, 'resize_token_embeddings'):
            return self.gemma.base_model.resize_token_embeddings(new_num_tokens)
        return None
    
    def get_output_embeddings(self):
        """Get output embeddings from the underlying Gemma model."""
        if hasattr(self.gemma, 'get_output_embeddings'):
            return self.gemma.get_output_embeddings()
        elif hasattr(self.gemma.base_model, 'get_output_embeddings'):
            return self.gemma.base_model.get_output_embeddings()
        return None
    
    def set_output_embeddings(self, embeddings):
        """Set output embeddings for the underlying Gemma model."""
        if hasattr(self.gemma, 'set_output_embeddings'):
            self.gemma.set_output_embeddings(embeddings)
        elif hasattr(self.gemma.base_model, 'set_output_embeddings'):
            self.gemma.base_model.set_output_embeddings(embeddings)
    
    def tie_weights(self):
        """Tie weights for the underlying Gemma model."""
        if hasattr(self.gemma, 'tie_weights'):
            self.gemma.tie_weights()
        elif hasattr(self.gemma.base_model, 'tie_weights'):
            self.gemma.base_model.tie_weights()
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Prepare inputs for generation."""
        if hasattr(self.gemma, 'prepare_inputs_for_generation'):
            return self.gemma.prepare_inputs_for_generation(*args, **kwargs)
        return None
        
    def forward(self, input_ids, attention_mask, pos_features, labels=None, **kwargs):
        # Get Gemma outputs - handle both PEFT and regular models
        if hasattr(self.gemma, 'base_model'):
            # PEFT model - we need to call the full model, not just the base
            outputs = self.gemma(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,  # Ensure we get hidden states
                **kwargs
            )
        else:
            # Regular model
            outputs = self.gemma(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,  # Ensure we get hidden states
                **kwargs
            )
        
        # Get the hidden states - handle different output formats
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # Use the last layer's hidden states
            hidden_states = outputs.hidden_states[-1]
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            # Fallback: try to get from the base model
            base_model = self.gemma.base_model.model if hasattr(self.gemma, 'base_model') else self.gemma.model
            base_outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs
            )
            hidden_states = base_outputs.hidden_states[-1] if hasattr(base_outputs, 'hidden_states') else base_outputs.last_hidden_state
        
        # Use mean pooling for sequence representation
        # Apply attention mask to avoid pooling over padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        sequence_output = sum_embeddings / sum_mask
        
        # Process POS features
        pos_embedded = self.pos_projection(pos_features)
        pos_embedded = self.pos_dropout(pos_embedded)
        
        # Combine features
        combined_features = torch.cat([sequence_output, pos_embedded], dim=1)
        logits = self.combined_classifier(combined_features)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # Standard cross-entropy loss (no class weights)
            loss = loss_fct(logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=getattr(outputs, 'attentions', None)
        )

class HateSpeechDatasetWithPOS(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=256, use_pos=True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_pos = use_pos
        
        # Pre-compute POS features if enabled
        if self.use_pos:
            print("🔄 Extracting POS features...")
            self.pos_features = []
            for idx, row in self.dataframe.iterrows():
                text = str(row['text']).strip()
                pos_feat = extract_pos_features(text)
                # Convert to list in consistent order
                feat_vector = [
                    pos_feat['noun_ratio'], pos_feat['verb_ratio'], pos_feat['adj_ratio'],
                    pos_feat['adv_ratio'], pos_feat['pronoun_ratio'], pos_feat['intj_ratio'],
                    pos_feat['profanity_ratio'], pos_feat['caps_ratio']
                ]
                self.pos_features.append(feat_vector)
                
                if (idx + 1) % 1000 == 0:
                    print(f"  Processed {idx + 1}/{len(self.dataframe)} samples...")
            
            print("✅ POS feature extraction completed!")
        
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
        
        result = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        
        # Add POS features if enabled
        if self.use_pos:
            result['pos_features'] = torch.tensor(self.pos_features[idx], dtype=torch.float32)
        
        return result

# Custom Trainer for POS features (no class weights)
class TrainerWithPOS(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        
        # Forward pass with POS features
        if 'pos_features' in inputs:
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                pos_features=inputs['pos_features'],
                labels=labels
            )
        else:
            outputs = model(**inputs)
        
        # Use standard cross-entropy loss (no class weights)
        loss = outputs.loss
            
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    """Enhanced evaluation metrics calculation."""
    labels = pred.label_ids
    
    # Handle case where predictions is a tuple (multiple outputs)
    if isinstance(pred.predictions, tuple):
        preds = pred.predictions[0].argmax(-1)  # Use first element (logits)
        predictions_for_auc = pred.predictions[0]  # Use logits for AUC calculation
    else:
        preds = pred.predictions.argmax(-1)
        predictions_for_auc = pred.predictions
    
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)

    f05 = fbeta_score(labels, preds, beta=0.5, average='binary', zero_division=0)
    f2 = fbeta_score(labels, preds, beta=2, average='binary', zero_division=0)
    
    try:
        probabilities = torch.softmax(torch.from_numpy(predictions_for_auc), dim=1).numpy()
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
    """Load and intelligently split the dataset with POS feature extraction."""
    print("📂 Loading dataset...")
    df = pd.read_csv(Config.DATASET_PATH)
    print(f"✅ Loaded {len(df)} samples")
    
    if df['label'].dtype == 'object':
        label_map = {'noHate': 0, 'hate': 1}
        df['label'] = df['label'].map(label_map)
        print("✅ Converted string labels to integers")
    
    label_counts = df['label'].value_counts()
    print(f"📊 Label distribution: {dict(label_counts)}")
    
    minority_class_count = min(label_counts.values)
    majority_class_count = max(label_counts.values)
    imbalance_ratio = majority_class_count / minority_class_count
    print(f"⚖️ Imbalance ratio: {imbalance_ratio:.2f}:1")
    print("🚫 No SMOTE or class weights applied - using original data distribution")
    
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
        "use_smote": False,  # Explicitly disabled
        "use_class_weights": False,  # Explicitly disabled
        "use_pos_features": Config.USE_POS_FEATURES,
        "pos_feature_dim": Config.POS_FEATURE_DIM,
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
    """Enhanced training function for Gemma with POS features only (no SMOTE, no class weights)."""
    print("🚀 Starting Gemma-7B Training with POS Features Only (No SMOTE, No Class Weights)")
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
    
    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, 
        num_labels=2,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if Config.BF16 else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    # Apply LoRA to the base model first
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
    
    model = get_peft_model(base_model, lora_config)
    print("✅ Applied LoRA configuration")
    model.print_trainable_parameters()
    
    # Always use POS features in this version
    print("✅ Wrapping Gemma with POS feature integration")
    model = GemmaWithPOS(model, pos_feature_dim=Config.POS_FEATURE_DIM)
    
    print("🚫 Class weights disabled - using standard cross-entropy loss")
    
    # Create datasets with POS features
    train_dataset = HateSpeechDatasetWithPOS(
        train_df, tokenizer, max_length=Config.MAX_LENGTH, use_pos=True
    )
    val_dataset = HateSpeechDatasetWithPOS(
        val_df, tokenizer, max_length=Config.MAX_LENGTH, use_pos=True
    )
    test_dataset = HateSpeechDatasetWithPOS(
        test_df, tokenizer, max_length=Config.MAX_LENGTH, use_pos=True
    )
    
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
        load_best_model_at_end=False,
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
        optim="paged_adamw_32bit",  # Memory efficient optimizer
        group_by_length=True,
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Use custom trainer for POS features (no class weights)
    trainer = TrainerWithPOS(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    print("🎯 Using TrainerWithPOS (no class weights)")
    
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
    
    # Handle case where predictions is a tuple (multiple outputs)
    if isinstance(predictions.predictions, tuple):
        y_pred = predictions.predictions[0].argmax(-1)  # Use first element (logits)
        predictions_for_probabilities = predictions.predictions[0]  # Use logits for probability calculation
    else:
        y_pred = predictions.predictions.argmax(-1)
        predictions_for_probabilities = predictions.predictions
    
    y_true = predictions.label_ids
    
    target_names = ['No Hate', 'Hate']
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print("\n📊 Detailed Classification Report:")
    print(report)
    
    report_path = os.path.join(Config.MODEL_OUTPUT_DIR, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("Gemma-7B with POS Features Only - Hate Speech Detection Test Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Enhancement Techniques Used:\n")
        f.write(f"- SMOTE: False (disabled)\n")
        f.write(f"- Class Weights: False (disabled)\n")
        f.write(f"- POS Features: {Config.USE_POS_FEATURES}\n")
        f.write(f"- POS Feature Dimension: {Config.POS_FEATURE_DIM}\n")
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
        "probabilities": torch.softmax(torch.from_numpy(predictions_for_probabilities), dim=1).numpy().tolist()
    }
    
    with open(test_predictions_path, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("🎉 GEMMA-7B WITH POS FEATURES ONLY TRAINING COMPLETED!")
    print("="*60)
    print(f"📁 Model and results saved in: {Config.MODEL_OUTPUT_DIR}")
    print(f"🎯 Test F1 Score: {test_results.get('test_f1', 0):.4f}")
    print(f"🎯 Test Accuracy: {test_results.get('test_accuracy', 0):.4f}")
    print(f"🚫 SMOTE Applied: False")
    print(f"🚫 Class Weights Used: False")
    print(f"🏷️ POS Features Used: {Config.USE_POS_FEATURES}")
    print(f"🔧 LoRA Fine-tuning: True")
    print(f"🔧 4-bit Quantization: {Config.USE_4BIT}")
    
    return model, tokenizer, test_results

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        Config.DATASET_PATH = sys.argv[1]
        Config.MODEL_OUTPUT_DIR = sys.argv[2]
    elif len(sys.argv) == 2:
        Config.DATASET_PATH = sys.argv[1]
        Config.MODEL_OUTPUT_DIR = "gemma_pos_only_" + os.path.splitext(os.path.basename(Config.DATASET_PATH))[0] + "_model"
    else:
        print("Usage: python train_gemma_pos_only.py <dataset_path> <output_dir>")
        sys.exit(1)
    model, tokenizer, results = train_model()