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
    DataCollatorWithPadding
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import time
import sys
import random
import re
import nltk
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Set DistilBERT-specific cache directories
os.environ["HF_HOME"] = "/scratch/ml23/bnge/distilbert/hf_cache"
os.environ["HF_HUB_CACHE"] = "/scratch/ml23/bnge/distilbert/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/scratch/ml23/bnge/distilbert/hf_cache"
os.environ["XDG_CACHE_HOME"] = "/scratch/ml23/bnge/distilbert/xdg_cache"
os.environ["MPLCONFIGDIR"] = "/scratch/ml23/bnge/distilbert/matplotlib_cache"
os.environ["TORCH_HOME"] = "/scratch/ml23/bnge/distilbert/torch_cache"
os.environ["TMPDIR"] = "/scratch/ml23/bnge/distilbert/tmp"

# Set NLTK data path
nltk_data_path = "/scratch/ml23/bnge/distilbert/nltk_data"
os.environ["NLTK_DATA"] = nltk_data_path
nltk.data.path.append(nltk_data_path)

# Create directories
cache_dirs = [
    "/scratch/ml23/bnge/distilbert/hf_cache", 
    "/scratch/ml23/bnge/distilbert/xdg_cache",
    "/scratch/ml23/bnge/distilbert/matplotlib_cache",
    "/scratch/ml23/bnge/distilbert/torch_cache",
    "/scratch/ml23/bnge/distilbert/tmp",
    nltk_data_path
]

for cache_dir in cache_dirs:
    os.makedirs(cache_dir, exist_ok=True)

print("DistilBERT cache directories created successfully!")

# Setup NLTK
def setup_nltk():
    """Setup NLTK data with proper error handling."""
    try:
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('averaged_perceptron_tagger')
        print("✅ NLTK data already available")
        return True
    except LookupError:
        print("📥 Downloading NLTK data...")
        try:
            nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)
            nltk.download('omw-1.4', download_dir=nltk_data_path, quiet=True)
            nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
            nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path, quiet=True)
            print("✅ NLTK data downloaded successfully")
            return True
        except Exception as e:
            print(f"⚠️ Error downloading NLTK data: {e}")
            return False

NLTK_AVAILABLE = setup_nltk()

if NLTK_AVAILABLE:
    try:
        from nltk.corpus import wordnet, stopwords
        from nltk.tag import pos_tag
        from nltk.tokenize import word_tokenize
        print("✅ NLTK corpus modules imported successfully")
    except ImportError as e:
        print(f"⚠️ NLTK import error: {e}")
        NLTK_AVAILABLE = False

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
    MODEL_OUTPUT_DIR = "distilbert_survey_augmentation_model"
    
    # Model settings
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 256
    
    # Training settings
    BATCH_SIZE = 16 if torch.cuda.is_available() else 8
    LEARNING_RATE = 3e-5
    NUM_EPOCHS = 10
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    GRADIENT_ACCUMULATION_STEPS = 2
    FP16 = torch.cuda.is_available()
    
    # Advanced training settings
    LABEL_SMOOTHING = 0.1
    DROPOUT = 0.3
    SCHEDULER_TYPE = "cosine"
    
    # Data preprocessing
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Survey-based TDA settings (Based on Applied Soft Computing 2023 paper)
    USE_SURVEY_AUGMENTATION = True
    AUGMENTATION_TARGET_RATIO = 1.2  # Perfect balance (1:1 ratio)
    
    # Character-Level Augmentation (Section 3.2.1 - Conservative approach)
    USE_CHARACTER_NOISE = True
    CHAR_SWITCH_PROB = 0.01          # Further reduced - paper shows diminishing returns
    CHAR_INSERT_PROB = 0.005         # Minimal insertion
    CHAR_DELETE_PROB = 0.005         # Minimal deletion  
    KEYBOARD_NOISE_PROB = 0.02       # Slightly increased for realism
    SPELLING_ERROR_PROB = 0.01       # Conservative spelling errors
    
    # Word-Level Augmentation (Section 3.2.2 - EDA + Enhanced methods)
    USE_WORD_LEVEL_AUG = True
    # EDA parameters (paper shows EDA is most robust across datasets)
    EDA_SYNONYM_REPLACE_PROB = 0.15   # Increased - most effective method
    EDA_RANDOM_SWAP_PROB = 0.12       # Increased swap frequency
    EDA_RANDOM_DELETE_PROB = 0.08     # Reduced deletion to preserve meaning
    EDA_RANDOM_INSERT_PROB = 0.12     # Increased insertion for diversity
    EDA_ALPHA = 0.15                  # Increased strength for more changes
    
    # BERT-based augmentation (paper shows BERT Aug performs well)
    USE_BERT_AUGMENTATION = True
    BERT_MASK_PROB = 0.25             # Increased for more contextual changes
    BERT_REPLACE_PROB = 0.15          # Increased replacement probability
    
    # WordNet synonym replacement (complementary to EDA)
    WORDNET_SYNONYM_PROB = 0.12       # Increased for more diversity
    
    # Sentence-Level Augmentation (Section 3.2.4 - Safest transformations)
    USE_SENTENCE_LEVEL_AUG = True
    CONTRACTION_TRANSFORM_PROB = 0.5  # High - very safe transformation
    
    # Back-translation (paper shows good semantic fidelity)
    USE_BACK_TRANSLATION = True
    BACK_TRANSLATION_PROB = 0.3       # Increased for semantic diversity
    
    # Quality control (Enhanced based on paper findings)
    MAX_AUGMENTATIONS_PER_SAMPLE = 6  # Increased for more diversity
    MIN_WORDS_THRESHOLD = 3           # Lowered threshold
    MAX_WORDS_THRESHOLD = 100         # Increased for longer contexts
    
    # Paper-based quality metrics
    MIN_SEMANTIC_SIMILARITY = 0.65    # Slightly relaxed for more diversity
    MAX_LEXICAL_DIVERSITY = 0.7       # Increased for more diverse augmentations
    
    # Word-level augmentation probabilities (Survey Section 3.1.2)
    SYNONYM_REPLACE_PROB = 0.12       # Increased WordNet usage
    RANDOM_SWAP_PROB = 0.12           # Balanced swap
    RANDOM_DELETE_PROB = 0.08         # Conservative deletion
    RANDOM_INSERT_PROB = 0.12         # Increased insertion

    # NEW: Advanced augmentation settings for performance
    USE_CONTEXTUAL_AUGMENTATION = True  # Use context-aware methods
    USE_PARAPHRASE_AUGMENTATION = True  # Add paraphrasing
    USE_ENSEMBLE_AUGMENTATION = True    # Combine multiple methods
    
    # NEW: Class-specific augmentation
    MINORITY_CLASS_AUG_MULTIPLIER = 2.0  # Extra augmentation for minority class
    AUGMENTATION_ROUNDS = 3              # Multiple rounds of augmentation

class SurveyBasedAugmenter:
    """Enhanced survey-based text data augmentation optimized for maximum performance."""
    
    def __init__(self):
        print("🚀 Initializing ENHANCED Survey-Based Text Augmenter for MAXIMUM PERFORMANCE...")
        
        # Initialize NLTK components
        if NLTK_AVAILABLE and stopwords:
            try:
                self.stop_words = set(stopwords.words('english'))
                print("✅ Stopwords loaded")
            except:
                self.stop_words = set([
                    'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'on', 'with', 
                    'as', 'by', 'at', 'from', 'it', 'an', 'be', 'are', 'was', 'been'
                ])
        else:
            self.stop_words = set([
                'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'on', 'with', 
                'as', 'by', 'at', 'from', 'it', 'an', 'be', 'are', 'was', 'been'
            ])
        
        # ENHANCED preserve words (CRITICAL for hate speech detection performance)
        self.preserve_words = {
            # Core hate indicators (NEVER modify these)
            'hate', 'kill', 'die', 'death', 'murder', 'terrorist', 'violence', 
            'attack', 'threat', 'bomb', 'shoot', 'gun', 'weapon', 'fight',
            'racist', 'sexist', 'discrimination', 'prejudice', 'evil', 'destroy',
            'nazi', 'fascist', 'supremacist', 'genocide', 'lynch', 'exterminate',
            'war', 'blood', 'brutal', 'savage', 'vicious', 'cruel', 'torture',
            
            # CRITICAL negation words (paper emphasizes extreme importance)
            'not', 'never', 'no', 'dont', 'cant', 'wont', 'shouldnt', 'isnt',
            'arent', 'wasnt', 'werent', 'havent', 'hasnt', 'hadnt', 'wouldnt',
            'couldnt', 'didnt', 'doesnt', 'neither', 'nor', 'none', 'nothing',
            'nowhere', 'nobody', 'barely', 'hardly', 'scarcely', 'without',
            
            # Emotional intensity words (preserve for sentiment strength)
            'very', 'really', 'extremely', 'totally', 'completely', 'absolutely',
            'fucking', 'damn', 'hell', 'shit', 'bloody', 'goddamn', 'bastard',
            'amazing', 'terrible', 'awful', 'horrible', 'wonderful', 'excellent',
            
            # Target groups (preserve for context - critical for hate speech)
            'black', 'white', 'asian', 'hispanic', 'latino', 'jewish', 'muslim',
            'christian', 'gay', 'lesbian', 'trans', 'transgender', 'women', 'men',
            'immigrant', 'immigrants', 'refugees', 'foreigners', 'minorities',
            'african', 'american', 'european', 'mexican', 'chinese', 'indian',
            'female', 'male', 'girl', 'boy', 'woman', 'man'
        }
        
        # Enhanced QWERTY keyboard layout for realistic typos
        self.keyboard_neighbors = {
            'a': ['q', 'w', 's', 'z'], 'b': ['v', 'g', 'h', 'n'],
            'c': ['x', 'd', 'f', 'v'], 'd': ['s', 'e', 'r', 'f', 'c', 'x'],
            'e': ['w', 's', 'd', 'r'], 'f': ['d', 'r', 't', 'g', 'v', 'c'],
            'g': ['f', 't', 'y', 'h', 'b', 'v'], 'h': ['g', 'y', 'u', 'j', 'n', 'b'],
            'i': ['u', 'j', 'k', 'o'], 'j': ['h', 'u', 'i', 'k', 'm', 'n'],
            'k': ['j', 'i', 'o', 'l', 'm'], 'l': ['k', 'o', 'p'],
            'm': ['n', 'j', 'k'], 'n': ['b', 'h', 'j', 'm'],
            'o': ['i', 'k', 'l', 'p'], 'p': ['o', 'l'],
            'q': ['w', 'a', 's'], 'r': ['e', 'd', 'f', 't'],
            's': ['a', 'w', 'e', 'd', 'x', 'z'], 't': ['r', 'f', 'g', 'y'],
            'u': ['y', 'h', 'j', 'i'], 'v': ['c', 'f', 'g', 'b'],
            'w': ['q', 'a', 's', 'e'], 'x': ['z', 's', 'd', 'c'],
            'y': ['t', 'g', 'h', 'u'], 'z': ['a', 's', 'x']
        }
        
        # Comprehensive contractions for sentence-level augmentation
        self.contractions = {
            # Basic contractions
            "I'm": "I am", "you're": "you are", "he's": "he is", "she's": "she is",
            "it's": "it is", "we're": "we are", "they're": "they are",
            "I've": "I have", "you've": "you have", "we've": "we have",
            "they've": "they have", "I'd": "I would", "you'd": "you would",
            "he'd": "he would", "she'd": "she would", "we'd": "we would",
            "they'd": "they would", "I'll": "I will", "you'll": "you will",
            "he'll": "he will", "she'll": "she will", "we'll": "we will",
            "they'll": "they will", "can't": "cannot", "won't": "will not",
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
            "wouldn't": "would not", "shouldn't": "should not", "couldn't": "could not",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
            
            # Extended contractions for more diversity
            "let's": "let us", "that's": "that is", "what's": "what is",
            "who's": "who is", "where's": "where is", "here's": "here is",
            "there's": "there is", "how's": "how is", "why's": "why is",
            "ain't": "is not", "gonna": "going to", "wanna": "want to",
            "gotta": "got to", "kinda": "kind of", "sorta": "sort of"
        }
        
        # Initialize enhanced semantic similarity model
        try:
            from sentence_transformers import SentenceTransformer
            # Try best model first, fallback to lighter model
            try:
                self.similarity_model = SentenceTransformer('all-mpnet-base-v2')
                print("✅ High-quality semantic similarity model loaded")
            except:
                self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Standard semantic similarity model loaded")
        except:
            self.similarity_model = None
            print("⚠️ Semantic similarity model not available")
        
        # Initialize paraphrase model for advanced augmentation
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # Updated imports
            
            # Set seed for reproducibility
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            
            # Force paraphrase model to CPU to save GPU memory (as before)
            self.paraphrase_device = torch.device("cpu")
            print(f"📝 Loading paraphrase model on CPU to conserve GPU memory")
            
            # Use BART instead of Pegasus
            self.paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("eugenesiow/bart-paraphrase")
            self.paraphrase_tokenizer = AutoTokenizer.from_pretrained("eugenesiow/bart-paraphrase")
            
            # Move to CPU
            self.paraphrase_model = self.paraphrase_model.to(self.paraphrase_device)
            
            print("✅ Paraphrase model loaded for advanced augmentation")
        except Exception as e:
            print(f"⚠️ Paraphrase model failed to load: {e}")
            self.paraphrase_model = None
            self.paraphrase_tokenizer = None
            self.paraphrase_device = None
            print("⚠️ Paraphrase model not available - using standard methods only")
    
    # ==== ENHANCED WORDNET METHODS ====
    
    def get_enhanced_wordnet_synonyms(self, word):
        """Get high-quality synonyms from WordNet without POS tagging."""
        if not NLTK_AVAILABLE or not wordnet:
            return []
        
        synonyms = set()
        try:
            # Get all synsets for the word (no POS restriction)
            synsets = wordnet.synsets(word)
            
            for syn in synsets:
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if (synonym != word and 
                        synonym.isalpha() and 
                        len(synonym) > 2 and
                        len(synonym) < 15 and  # Avoid very long synonyms
                        synonym.lower() not in self.stop_words and
                        synonym.lower() not in self.preserve_words):
                        synonyms.add(synonym)
        except Exception:
            pass
        
        return list(synonyms)[:8]  # More synonyms for better selection
    
    def select_best_synonym(self, original_word, synonyms, context_words, word_idx):
        """Select the best synonym based on semantic similarity without POS."""
        if not synonyms:
            return None
        
        # If no similarity model, use simple heuristics + random selection
        if not self.similarity_model:
            # Prefer synonyms with similar length
            length_similar = [s for s in synonyms if abs(len(s) - len(original_word)) <= 2]
            if length_similar:
                return random.choice(length_similar)
            return random.choice(synonyms)
        
        # Create context for comparison
        context = ' '.join(context_words)
        best_synonym = None
        best_score = -1
        
        # Test top 5 synonyms for performance
        test_synonyms = synonyms[:5]
        
        for synonym in test_synonyms:
            # Create augmented context
            test_words = context_words.copy()
            test_words[word_idx] = synonym
            test_context = ' '.join(test_words)
            
            # Calculate similarity
            try:
                similarity = self.calculate_semantic_similarity(context, test_context)
                if similarity > best_score and similarity > 0.75:  # High threshold
                    best_score = similarity
                    best_synonym = synonym
            except:
                continue
        
        # If no good semantic match, use length-based heuristic
        if best_score <= 0.75:
            length_similar = [s for s in synonyms if abs(len(s) - len(original_word)) <= 2]
            if length_similar:
                return random.choice(length_similar)
            return random.choice(synonyms[:3])
        
        return best_synonym
    
    # ==== ENHANCED EDA METHODS (BEST PERFORMING) ====
    
    def enhanced_eda_synonym_replacement(self, text, n):
        """Enhanced EDA synonym replacement without POS awareness."""
        if not NLTK_AVAILABLE:
            return text
        
        words = text.split()
        new_words = words.copy()
        
        # Select words for replacement based on simple criteria
        replacement_candidates = []
        for i, word in enumerate(words):
            word_clean = word.lower().strip('.,!?;:"\'()[]{}')
            if (word_clean not in self.preserve_words and 
                word_clean not in self.stop_words and 
                len(word) > 2 and 
                word.isalpha() and
                len(word) < 15):  # Avoid very long words
                replacement_candidates.append((i, word))
        
        # Perform enhanced replacements
        num_replaced = 0
        random.shuffle(replacement_candidates)
        
        for idx, word in replacement_candidates:
            if num_replaced >= n:
                break
            
            synonyms = self.get_enhanced_wordnet_synonyms(word)
            
            if synonyms:
                # Select best synonym based on context
                best_synonym = self.select_best_synonym(word, synonyms, words, idx)
                if best_synonym:
                    new_words[idx] = best_synonym
                    num_replaced += 1
        
        return ' '.join(new_words)
    
    def enhanced_eda_random_swap(self, text, n):
        """Enhanced EDA random swap without grammatical awareness."""
        words = text.split()
        if len(words) < 2:
            return text
        
        new_words = words.copy()
        
        # Get swappable word indices (avoid preserve words)
        swappable_indices = []
        for i, word in enumerate(words):
            word_clean = word.lower().strip('.,!?;:"\'()[]{}')
            if word_clean not in self.preserve_words:
                swappable_indices.append(i)
        
        if len(swappable_indices) < 2:
            # Fallback to all indices if no swappable words found
            swappable_indices = list(range(len(words)))
        
        num_swaps = min(n, len(words) // 3)
        
        for _ in range(num_swaps):
            if len(swappable_indices) >= 2:
                idx1, idx2 = random.sample(swappable_indices, 2)
                new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return ' '.join(new_words)
    
    def enhanced_eda_random_deletion(self, text, p):
        """Enhanced EDA random deletion with critical word preservation."""
        words = text.split()
        if len(words) <= 3:  # Preserve very short sentences
            return text
        
        new_words = []
        preserved_count = 0
        
        for word in words:
            word_lower = word.lower().strip('.,!?;:"\'()[]{}')
            
            # NEVER delete critical words
            if (word_lower in self.preserve_words or 
                word_lower in ['not', 'never', 'no', 'nothing', 'none'] or  # Extra negation protection
                len(word) <= 2):
                new_words.append(word)
                preserved_count += 1
            elif random.uniform(0, 1) > p:
                new_words.append(word)
        
        # Ensure minimum length and preserved words
        if len(new_words) < max(3, len(words) * 0.7) or preserved_count == 0:
            return text
        
        return ' '.join(new_words)
    
    def enhanced_eda_random_insertion(self, text, n):
        """Enhanced EDA random insertion without POS awareness."""
        if not NLTK_AVAILABLE:
            return text
        
        words = text.split()
        new_words = words.copy()
        
        insertions_made = 0
        
        for _ in range(n):
            if insertions_made >= n:
                break
            
            # Select suitable words for synonym insertion
            suitable_words = []
            for i, word in enumerate(words):
                word_clean = word.lower().strip('.,!?;:"\'()[]{}')
                if (word_clean not in self.preserve_words and 
                    word_clean not in self.stop_words and 
                    len(word) > 2 and 
                    word.isalpha() and
                    len(word) < 12):
                    suitable_words.append((i, word))
            
            if not suitable_words:
                continue
            
            word_idx, word = random.choice(suitable_words)
            synonyms = self.get_enhanced_wordnet_synonyms(word)
            
            if synonyms:
                # Select appropriate synonym with context
                synonym = self.select_best_synonym(word, synonyms, new_words, word_idx)
                if synonym:
                    # Insert near the original word for better context
                    insert_positions = [
                        max(0, word_idx - 1),
                        word_idx,
                        min(len(new_words), word_idx + 1)
                    ]
                    insert_pos = random.choice(insert_positions)
                    new_words.insert(insert_pos, synonym)
                    insertions_made += 1
        
        return ' '.join(new_words)
    
    def apply_enhanced_eda(self, text):
        """Apply enhanced EDA methods - highest performing according to survey."""
        if not Config.USE_WORD_LEVEL_AUG:
            return []
        
        variations = []
        words = text.split()
        
        if len(words) < Config.MIN_WORDS_THRESHOLD:
            return []
        
        # Calculate dynamic number of changes based on text length
        n_changes = max(1, int(Config.EDA_ALPHA * len(words)))
        
        # Enhanced EDA Synonym Replacement (highest priority - best performance)
        if random.random() < Config.EDA_SYNONYM_REPLACE_PROB:
            syn_aug = self.enhanced_eda_synonym_replacement(text, n_changes)
            if syn_aug != text and self.enhanced_quality_check(syn_aug, text):
                variations.append(syn_aug)
        
        # Enhanced EDA Random Swap
        if random.random() < Config.EDA_RANDOM_SWAP_PROB:
            swap_aug = self.enhanced_eda_random_swap(text, n_changes)
            if swap_aug != text and self.enhanced_quality_check(swap_aug, text):
                variations.append(swap_aug)
        
        # Enhanced EDA Random Deletion (more conservative)
        if random.random() < Config.EDA_RANDOM_DELETE_PROB:
            del_aug = self.enhanced_eda_random_deletion(text, Config.EDA_ALPHA * 0.8)
            if del_aug != text and self.enhanced_quality_check(del_aug, text):
                variations.append(del_aug)
        
        # Enhanced EDA Random Insertion
        if random.random() < Config.EDA_RANDOM_INSERT_PROB:
            ins_aug = self.enhanced_eda_random_insertion(text, n_changes)
            if ins_aug != text and self.enhanced_quality_check(ins_aug, text):
                variations.append(ins_aug)
        
        return variations
    
    # ==== ENHANCED BERT-BASED AUGMENTATION ====
    
    def apply_enhanced_bert_augmentation(self, text):
        """Enhanced BERT-based contextual augmentation."""
        if not Config.USE_BERT_AUGMENTATION:
            return []
        
        variations = []
        
        # Enhanced BERT masked language modeling
        if random.random() < Config.BERT_MASK_PROB:
            bert_aug = self.enhanced_bert_mask_and_replace(text)
            if bert_aug != text and self.enhanced_quality_check(bert_aug, text):
                variations.append(bert_aug)
        
        # Additional BERT-based contextual replacement
        if random.random() < Config.BERT_REPLACE_PROB:
            contextual_aug = self.contextual_word_replacement(text)
            if contextual_aug != text and self.enhanced_quality_check(contextual_aug, text):
                variations.append(contextual_aug)
        
        return variations
    
    def enhanced_bert_mask_and_replace(self, text):
        """Enhanced BERT masking with better word selection (no POS tagging)."""
        words = text.split()
        if len(words) < 3:
            return text
        
        new_words = words.copy()
        
        # Enhanced word selection for masking without POS tagging
        maskable_indices = []
        for i, word in enumerate(words):
            word_clean = word.lower().strip('.,!?;:"\'()[]{}')
            if (word_clean not in self.preserve_words and 
                word_clean not in self.stop_words and 
                word.isalpha() and 
                len(word) > 2 and
                len(word) < 15):  # Avoid very long words
                maskable_indices.append(i)
        
        if not maskable_indices:
            return text
        
        # Mask 1-3 words intelligently
        num_masks = min(3, max(1, len(maskable_indices) // 4))
        mask_indices = random.sample(maskable_indices, min(num_masks, len(maskable_indices)))
        
        for idx in mask_indices:
            if random.random() < Config.BERT_REPLACE_PROB:
                # Enhanced contextual replacement
                original_word = words[idx]
                synonyms = self.get_enhanced_wordnet_synonyms(original_word)
                
                if synonyms:
                    # Use context-aware selection if similarity model available
                    best_synonym = self.select_best_synonym(original_word, synonyms, words, idx)
                    if best_synonym:
                        new_words[idx] = best_synonym
                    else:
                        # Fallback to random synonym
                        new_words[idx] = random.choice(synonyms)
        
        return ' '.join(new_words)
    
    def contextual_word_replacement(self, text):
        """Replace words based on context without POS tagging."""
        words = text.split()
        if len(words) < 3:
            return text
        
        new_words = words.copy()
        
        # Select words for replacement based on simple heuristics
        replacement_candidates = []
        for i, word in enumerate(words):
            word_clean = word.lower().strip('.,!?;:"\'()[]{}')
            if (word_clean not in self.preserve_words and 
                word_clean not in self.stop_words and 
                word.isalpha() and 
                len(word) > 3 and
                len(word) < 12):  # Focus on medium-length words
                replacement_candidates.append((i, word))
        
        if not replacement_candidates:
            return text
        
        # Replace 1-2 words maximum
        num_replacements = min(2, len(replacement_candidates))
        selected_candidates = random.sample(replacement_candidates, num_replacements)
        
        for idx, word in selected_candidates:
            synonyms = self.get_enhanced_wordnet_synonyms(word)
            if synonyms:
                # Use context-aware selection
                best_synonym = self.select_best_synonym(word, synonyms, words, idx)
                if best_synonym:
                    new_words[idx] = best_synonym
        
        return ' '.join(new_words)
    
    # ==== PARAPHRASE AUGMENTATION ====
    def apply_paraphrase_augmentation(self, text):
        """Apply Pegasus-based paraphrasing for high-quality semantic preservation."""
        if not Config.USE_PARAPHRASE_AUGMENTATION or not self.paraphrase_model or not self.paraphrase_tokenizer:
            return []
        
        variations = []
        
        try:
            # Prepare input text (Pegasus doesn't need the "paraphrase:" prefix)
            input_text = text
            
            # Tokenize
            encoding = self.paraphrase_tokenizer(
                input_text, 
                truncation=True, 
                padding='longest', 
                return_tensors="pt",
                max_length=60  # Pegasus works well with shorter inputs
            )
            input_ids = encoding["input_ids"].to(self.paraphrase_device)
            attention_mask = encoding["attention_mask"].to(self.paraphrase_device)
            
            # Generate paraphrases (Pegasus-optimized parameters)
            beam_outputs = self.paraphrase_model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                max_length=60,  # Shorter than T5 for better quality
                num_beams=10,   # Beam search for quality
                num_return_sequences=5,  # Generate fewer but higher quality paraphrases
                temperature=1.5,  # Higher temperature for more diversity
                repetition_penalty=3.0,  # Penalize repetition
                length_penalty=1.0,  # Neutral length penalty
                early_stopping=True,
                do_sample=True,
                top_k=120,
                top_p=0.95
            )
            
            # Decode and filter
            final_outputs = []
            for beam_output in beam_outputs:
                paraphrased = self.paraphrase_tokenizer.decode(
                    beam_output, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True
                )
                # Filter duplicates and identical text
                if (paraphrased.lower() != text.lower() and 
                    paraphrased not in final_outputs and
                    len(paraphrased.split()) >= Config.MIN_WORDS_THRESHOLD and
                    self.enhanced_quality_check(paraphrased, text)):
                    final_outputs.append(paraphrased)
            
            # Limit to top variations
            variations = final_outputs[:Config.MAX_AUGMENTATIONS_PER_SAMPLE]
                        
        except Exception as e:
            print(f"⚠️ Paraphrase error: {e}")
        
        return variations
    
    # ==== ENHANCED CHARACTER-LEVEL AUGMENTATION ====
    
    def enhanced_character_level_augmentation(self, text):
        """Enhanced character-level augmentation with better quality control."""
        if not Config.USE_CHARACTER_NOISE:
            return []
        
        variations = []
        
        # More conservative character-level changes
        methods = [
            (Config.KEYBOARD_NOISE_PROB, self.enhanced_keyboard_neighbor_replacement),
            (Config.CHAR_SWITCH_PROB, self.enhanced_random_character_switch),
            (Config.SPELLING_ERROR_PROB, self.enhanced_spelling_error_induction),
            (Config.CHAR_INSERT_PROB * 0.5, self.enhanced_random_character_insertion),  # Reduced
            (Config.CHAR_DELETE_PROB * 0.5, self.enhanced_random_character_deletion),   # Reduced
        ]
        
        for prob, method in methods:
            if random.random() < prob:
                aug_text = method(text)
                if (aug_text != text and 
                    self.enhanced_quality_check(aug_text, text)):
                    variations.append(aug_text)
        
        return variations[:2]  # Limit character-level augmentations
    
    def enhanced_keyboard_neighbor_replacement(self, text):
        """Enhanced keyboard neighbor replacement with word boundary awareness."""
        if len(text) < 5:
            return text
        
        words = text.split()
        new_words = []
        
        for word in words:
            if (len(word) > 3 and 
                word.lower() not in self.preserve_words and
                word.isalpha()):
                
                chars = list(word.lower())
                # Only replace 1 character per word maximum
                if random.random() < 0.3:  # 30% chance per word
                    pos = random.randint(1, len(chars) - 2)  # Avoid first/last char
                    char = chars[pos]
                    
                    if char in self.keyboard_neighbors:
                        replacement = random.choice(self.keyboard_neighbors[char])
                        chars[pos] = replacement
                        new_words.append(''.join(chars))
                    else:
                        new_words.append(word)
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        
        return ' '.join(new_words)
    
    def enhanced_random_character_switch(self, text):
        """Enhanced character switching with word awareness."""
        words = text.split()
        new_words = []
        
        for word in words:
            if (len(word) > 4 and 
                word.lower() not in self.preserve_words and
                word.isalpha()):
                
                chars = list(word)
                # Switch adjacent characters (realistic typo)
                if random.random() < 0.2:  # 20% chance per word
                    pos = random.randint(1, len(chars) - 3)  # Safe position
                    if chars[pos].isalpha() and chars[pos + 1].isalpha():
                        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
                new_words.append(''.join(chars))
            else:
                new_words.append(word)
        
        return ' '.join(new_words)
    
    def enhanced_spelling_error_induction(self, text):
        """Enhanced spelling errors with common patterns."""
        # Common and realistic spelling error patterns
        error_patterns = [
            ('receive', 'recieve'), ('believe', 'beleive'), ('achieve', 'acheive'),
            ('separate', 'seperate'), ('definitely', 'definately'), ('necessary', 'neccessary'),
            ('their', 'thier'), ('friend', 'freind'), ('piece', 'peice'),
            ('weird', 'wierd'), ('maintenance', 'maintainance'), ('occurred', 'occured'),
            ('ie', 'ei'), ('ei', 'ie'),  # ie/ei confusion
            ('tion', 'sion'), ('sion', 'tion'),  # suffix confusion
        ]
        
        result = text
        for wrong, right in error_patterns:
            if right in result.lower() and random.random() < 0.15:  # 15% chance
                # Case-aware replacement
                if right in result:
                    result = result.replace(right, wrong, 1)
                elif right.capitalize() in result:
                    result = result.replace(right.capitalize(), wrong.capitalize(), 1)
                break
        
        return result
    
    def enhanced_random_character_insertion(self, text):
        """Conservative character insertion."""
        words = text.split()
        new_words = []
        
        for word in words:
            if (len(word) > 4 and 
                word.lower() not in self.preserve_words and
                word.isalpha() and
                random.random() < 0.1):  # 10% chance per word
                
                chars = list(word)
                pos = random.randint(1, len(chars) - 1)
                char_to_insert = random.choice('aeiou')  # Vowels are safer
                chars.insert(pos, char_to_insert)
                new_words.append(''.join(chars))
            else:
                new_words.append(word)
        
        return ' '.join(new_words)
    
    def enhanced_random_character_deletion(self, text):
        """Conservative character deletion."""
        words = text.split()
        new_words = []
        
        for word in words:
            if (len(word) > 5 and 
                word.lower() not in self.preserve_words and
                word.isalpha() and
                random.random() < 0.1):  # 10% chance per word
                
                chars = list(word)
                # Only delete consonants in middle positions
                consonants_pos = [i for i, c in enumerate(chars[1:-1], 1) 
                                 if c.lower() not in 'aeiou']
                if consonants_pos:
                    pos = random.choice(consonants_pos)
                    chars.pop(pos)
                    new_words.append(''.join(chars))
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        
        return ' '.join(new_words)
    
    # ==== ENHANCED SENTENCE-LEVEL AUGMENTATION ====
    
    def enhanced_sentence_level_augmentation(self, text):
        """Enhanced sentence-level augmentation with multiple techniques."""
        if not Config.USE_SENTENCE_LEVEL_AUG:
            return []
        
        variations = []
        
        # Enhanced contraction transformation
        if random.random() < Config.CONTRACTION_TRANSFORM_PROB:
            contraction_aug = self.enhanced_contraction_transformation(text)
            if contraction_aug != text:
                variations.append(contraction_aug)
        
        return variations
    
    def enhanced_contraction_transformation(self, text):
        """Enhanced contraction handling with better context awareness."""
        result = text
        
        if random.random() < 0.5:
            # Expand contractions
            for contraction, expansion in self.contractions.items():
                # Case-aware replacement
                pattern = r'\b' + re.escape(contraction) + r'\b'
                result = re.sub(pattern, expansion, result, flags=re.IGNORECASE)
        else:
            # Contract expansions (reverse process)
            for contraction, expansion in self.contractions.items():
                pattern = r'\b' + re.escape(expansion) + r'\b'
                result = re.sub(pattern, contraction, result, flags=re.IGNORECASE)
        
        return result
    
    # ==== ENHANCED QUALITY CONTROL ====
    
    def calculate_semantic_similarity(self, text1, text2):
        """Calculate semantic similarity using transformer models."""
        if self.similarity_model:
            try:
                embeddings = self.similarity_model.encode([text1, text2])
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                return float(similarity)
            except:
                pass
        
        # Enhanced fallback: word overlap with weights
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Weight important words more heavily
        important_words1 = set(w for w in words1 if w in self.preserve_words)
        important_words2 = set(w for w in words2 if w in self.preserve_words)
        
        # Calculate weighted similarity
        important_overlap = len(important_words1 & important_words2)
        total_overlap = len(words1 & words2)
        total_union = len(words1 | words2)
        
        if total_union == 0:
            return 0.0
        
        # Give extra weight to important word preservation
        basic_similarity = total_overlap / total_union
        important_weight = (important_overlap / max(1, len(important_words1 | important_words2))) * 0.3
        
        return min(1.0, basic_similarity + important_weight)
    
    def enhanced_quality_check(self, augmented_text, original_text):
        """Enhanced quality check with stricter criteria for performance."""
        # Basic validation
        aug_words = augmented_text.split()
        orig_words = original_text.split()
        
        if len(aug_words) < max(2, len(orig_words) * 0.7):
            return False
        if len(aug_words) > len(orig_words) * 1.6:
            return False
        
        # Character length validation
        if len(augmented_text) < len(original_text) * 0.5:
            return False
        
        # Enhanced semantic similarity check
        similarity = self.calculate_semantic_similarity(original_text, augmented_text)
        if similarity < Config.MIN_SEMANTIC_SIMILARITY:
            return False
        
        # CRITICAL: Preserve important words with higher threshold
        orig_preserve_words = set(word.lower().strip('.,!?;:"\'()[]{}') 
                                 for word in orig_words 
                                 if word.lower().strip('.,!?;:"\'()[]{}') in self.preserve_words)
        aug_preserve_words = set(word.lower().strip('.,!?;:"\'()[]{}') 
                                for word in aug_words 
                                if word.lower().strip('.,!?;:"\'()[]{}') in self.preserve_words)
        
        if len(orig_preserve_words) > 0:
            preserved_ratio = len(orig_preserve_words & aug_preserve_words) / len(orig_preserve_words)
            if preserved_ratio < 0.9:  # Very high threshold for hate speech
                return False
        
        # CRITICAL: Negation preservation (essential for hate speech detection)
        negations = {'not', 'never', 'no', 'nothing', 'dont', 'doesnt', 'didnt', 
                    'cant', 'wont', 'isnt', 'arent', 'wasnt', 'werent'}
        orig_negations = set(word.lower().strip('.,!?;:"\'') for word in orig_words 
                            if word.lower().strip('.,!?;:"\'') in negations)
        aug_negations = set(word.lower().strip('.,!?;:"\'') for word in aug_words 
                           if word.lower().strip('.,!?;:"\'') in negations)
        
        if len(orig_negations) > 0:
            neg_preserved = len(orig_negations & aug_negations) / len(orig_negations)
            if neg_preserved < 0.95:  # Extremely high threshold for negations
                return False
        
        # Quality checks for readability
        if re.search(r'(.)\1{3,}', augmented_text):  # No repeated chars
            return False
        
        if re.search(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{6,}', augmented_text):
            return False
        
        # Check lexical diversity (ensure some change occurred)
        orig_unique = set(word.lower() for word in orig_words)
        aug_unique = set(word.lower() for word in aug_words)
        
        # Should have some diversity but not too much
        diversity = len(aug_unique - orig_unique) / max(1, len(orig_unique))
        if diversity > 0.5:  # Too much change
            return False
        
        return True
    
    # ==== MAIN ENHANCED AUGMENTATION METHOD ====
    
    def augment_text(self, text):
        """ENHANCED main augmentation method optimized for MAXIMUM PERFORMANCE."""
        all_augmented = []
        
        # Preprocessing and validation
        text = text.strip()
        if not text:
            return []
        
        words = text.split()
        if (len(words) < Config.MIN_WORDS_THRESHOLD or 
            len(words) > Config.MAX_WORDS_THRESHOLD):
            return []
        
        # Apply augmentation techniques in order of effectiveness (based on survey paper)
        
        # 1. ENHANCED EDA (Paper shows best performance - HIGHEST PRIORITY)
        eda_augmentations = self.apply_enhanced_eda(text)
        all_augmented.extend(eda_augmentations)
        
        # 2. Enhanced BERT-based augmentation (Second best performance)
        bert_augmentations = self.apply_enhanced_bert_augmentation(text)
        all_augmented.extend(bert_augmentations)
        
        # 3. Paraphrasing (High semantic fidelity)
        if Config.USE_PARAPHRASE_AUGMENTATION:
            para_augmentations = self.apply_paraphrase_augmentation(text)
            all_augmented.extend(para_augmentations)
        
        # 4. Enhanced sentence-level (Safest transformations)
        sentence_augmentations = self.enhanced_sentence_level_augmentation(text)
        all_augmented.extend(sentence_augmentations)
        
        # 5. Conservative character-level (Lowest priority, most conservative)
        if Config.USE_CHARACTER_NOISE:
            char_augmentations = self.enhanced_character_level_augmentation(text)
            all_augmented.extend(char_augmentations)
        
        # ENHANCED quality filtering with performance optimization
        filtered_augmented = []
        seen = {text.lower()}
        
        # Sort by quality score for better selection
        quality_scored = []
        for aug_text in all_augmented:
            if (aug_text and 
                aug_text.lower() not in seen and 
                len(aug_text.split()) >= Config.MIN_WORDS_THRESHOLD):
                
                if self.enhanced_quality_check(aug_text, text):
                    # Calculate quality score
                    similarity = self.calculate_semantic_similarity(text, aug_text)
                    quality_score = similarity  # Higher similarity = higher quality
                    quality_scored.append((aug_text, quality_score))
        
        # Sort by quality and take best ones
        quality_scored.sort(key=lambda x: x[1], reverse=True)
        
        for aug_text, score in quality_scored[:Config.MAX_AUGMENTATIONS_PER_SAMPLE]:
            if aug_text.lower() not in seen:
                filtered_augmented.append(aug_text)
                seen.add(aug_text.lower())
        
        return filtered_augmented

def save_datasets_to_csv(train_df, val_df, test_df, config):
    """Save train, validation, and test datasets to CSV files for analysis."""
    print("💾 Saving datasets to CSV files...")
    
    # Create datasets directory if it doesn't exist
    datasets_dir = os.path.join(config.MODEL_OUTPUT_DIR, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Define file paths
    train_csv_path = os.path.join(datasets_dir, f"train_dataset_{timestamp}.csv")
    val_csv_path = os.path.join(datasets_dir, f"validation_dataset_{timestamp}.csv")
    test_csv_path = os.path.join(datasets_dir, f"test_dataset_{timestamp}.csv")
    
    # Save datasets
    try:
        # Save training dataset (with augmentation info if available)
        train_df.to_csv(train_csv_path, index=False, encoding='utf-8')
        print(f"✅ Training dataset saved: {train_csv_path}")
        print(f"   📊 Shape: {train_df.shape}")
        
        # Save validation dataset
        val_df.to_csv(val_csv_path, index=False, encoding='utf-8')
        print(f"✅ Validation dataset saved: {val_csv_path}")
        print(f"   📊 Shape: {val_df.shape}")
        
        # Save test dataset
        test_df.to_csv(test_csv_path, index=False, encoding='utf-8')
        print(f"✅ Test dataset saved: {test_csv_path}")
        print(f"   📊 Shape: {test_df.shape}")
        
        # Create dataset summary
        summary_path = os.path.join(datasets_dir, f"dataset_summary_{timestamp}.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("DATASET SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {config.MODEL_NAME}\n")
            f.write(f"Survey Augmentation: {config.USE_SURVEY_AUGMENTATION}\n\n")
            
            # Training dataset analysis
            f.write("TRAINING DATASET:\n")
            f.write(f"  Total samples: {len(train_df)}\n")
            if 'label' in train_df.columns:
                train_counts = train_df['label'].value_counts().sort_index()
                for label, count in train_counts.items():
                    label_name = 'Hate' if label == 1 else 'No Hate'
                    percentage = count / len(train_df) * 100
                    f.write(f"  {label_name}: {count} ({percentage:.1f}%)\n")
            
            # Check for augmentation info
            if 'source' in train_df.columns:
                f.write("\n  Augmentation breakdown:\n")
                source_counts = train_df['source'].value_counts()
                for source, count in source_counts.items():
                    percentage = count / len(train_df) * 100
                    f.write(f"    {source}: {count} ({percentage:.1f}%)\n")
            
            if 'augmentation_method' in train_df.columns:
                f.write("\n  Augmentation methods:\n")
                method_counts = train_df['augmentation_method'].value_counts()
                for method, count in method_counts.items():
                    percentage = count / len(train_df) * 100
                    f.write(f"    {method}: {count} ({percentage:.1f}%)\n")
            
            # Validation dataset analysis
            f.write(f"\nVALIDATION DATASET:\n")
            f.write(f"  Total samples: {len(val_df)}\n")
            if 'label' in val_df.columns:
                val_counts = val_df['label'].value_counts().sort_index()
                for label, count in val_counts.items():
                    label_name = 'Hate' if label == 1 else 'No Hate'
                    percentage = count / len(val_df) * 100
                    f.write(f"  {label_name}: {count} ({percentage:.1f}%)\n")
            
            # Test dataset analysis
            f.write(f"\nTEST DATASET:\n")
            f.write(f"  Total samples: {len(test_df)}\n")
            if 'label' in test_df.columns:
                test_counts = test_df['label'].value_counts().sort_index()
                for label, count in test_counts.items():
                    label_name = 'Hate' if label == 1 else 'No Hate'
                    percentage = count / len(test_df) * 100
                    f.write(f"  {label_name}: {count} ({percentage:.1f}%)\n")
            
            # Overall statistics
            total_samples = len(train_df) + len(val_df) + len(test_df)
            f.write(f"\nOVERALL STATISTICS:\n")
            f.write(f"  Total samples: {total_samples}\n")
            f.write(f"  Train split: {len(train_df)/total_samples*100:.1f}%\n")
            f.write(f"  Validation split: {len(val_df)/total_samples*100:.1f}%\n")
            f.write(f"  Test split: {len(test_df)/total_samples*100:.1f}%\n")
        
        print(f"✅ Dataset summary saved: {summary_path}")
        
        # Create augmentation analysis (if augmentation was used)
        if config.USE_SURVEY_AUGMENTATION and 'source' in train_df.columns:
            aug_analysis_path = os.path.join(datasets_dir, f"augmentation_analysis_{timestamp}.csv")
            
            # Detailed augmentation analysis
            aug_data = train_df[train_df['source'] != 'original'].copy()
            if len(aug_data) > 0:
                # Sample some augmented examples for inspection
                sample_aug = aug_data.sample(min(100, len(aug_data))).copy()
                sample_aug = sample_aug[['text', 'label', 'source', 'augmentation_method', 'round']].copy()
                sample_aug.to_csv(aug_analysis_path, index=False, encoding='utf-8')
                print(f"✅ Augmentation analysis saved: {aug_analysis_path}")
                print(f"   📊 Sample augmented texts: {len(sample_aug)}")
        
        # Return file paths for reference
        return {
            'train': train_csv_path,
            'validation': val_csv_path,
            'test': test_csv_path,
            'summary': summary_path
        }
        
    except Exception as e:
        print(f"❌ Error saving datasets: {e}")
        return None

def create_augmentation_examples_report(train_df, config):
    """Create a detailed report showing original vs augmented examples."""
    if not config.USE_SURVEY_AUGMENTATION or 'source' not in train_df.columns:
        return
    
    print("📋 Creating augmentation examples report...")
    
    datasets_dir = os.path.join(config.MODEL_OUTPUT_DIR, "datasets")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    examples_path = os.path.join(datasets_dir, f"augmentation_examples_{timestamp}.csv")
    
    try:
        # Get original samples that were augmented
        original_samples = train_df[train_df['source'] == 'original'].copy()
        augmented_samples = train_df[train_df['source'] != 'original'].copy()
        
        if len(augmented_samples) > 0:
            examples_data = []
            
            # Sample 20 original texts that have augmentations
            sample_originals = original_samples.sample(min(20, len(original_samples)))
            
            for idx, orig_row in sample_originals.iterrows():
                # Find augmentations for this original sample
                if 'original_idx' in augmented_samples.columns:
                    augs = augmented_samples[augmented_samples['original_idx'] == idx]
                else:
                    # If no original_idx, just get some random augmented samples
                    augs = augmented_samples[augmented_samples['label'] == orig_row['label']].head(3)
                
                # Add original
                examples_data.append({
                    'type': 'ORIGINAL',
                    'text': orig_row['text'],
                    'label': orig_row['label'],
                    'label_name': 'Hate' if orig_row['label'] == 1 else 'No Hate',
                    'source': orig_row['source'],
                    'method': 'none',
                    'group_id': idx
                })
                
                # Add augmentations
                for aug_idx, aug_row in augs.head(3).iterrows():  # Max 3 augmentations per original
                    examples_data.append({
                        'type': 'AUGMENTED',
                        'text': aug_row['text'],
                        'label': aug_row['label'],
                        'label_name': 'Hate' if aug_row['label'] == 1 else 'No Hate',
                        'source': aug_row['source'],
                        'method': aug_row.get('augmentation_method', 'unknown'),
                        'group_id': idx
                    })
                
                # Add separator
                examples_data.append({
                    'type': '---SEPARATOR---',
                    'text': '---',
                    'label': -1,
                    'label_name': '---',
                    'source': '---',
                    'method': '---',
                    'group_id': idx
                })
            
            # Save examples
            examples_df = pd.DataFrame(examples_data)
            examples_df.to_csv(examples_path, index=False, encoding='utf-8')
            print(f"✅ Augmentation examples report saved: {examples_path}")
            print(f"   📊 Examples included: {len(sample_originals)} original texts with their augmentations")
            
    except Exception as e:
        print(f"❌ Error creating augmentation examples report: {e}")

def apply_survey_augmentation(df):
    """Apply survey-based augmentation to create a BALANCED dataset using TDA."""
    if not Config.USE_SURVEY_AUGMENTATION:
        print("🚫 Survey-based augmentation disabled")
        return df
    
    print("📚 Applying TDA Survey-Based Text Data Augmentation for BALANCED Dataset")
    print("📊 Using proven methods from data augmentation survey to achieve class balance")
    
    # LOG ORIGINAL SIZE AND DISTRIBUTION
    original_size = len(df)
    print(f"📊 ORIGINAL training set size: {original_size}")
    
    # Initialize augmenter
    augmenter = SurveyBasedAugmenter()
    
    # Analyze class distribution
    hate_samples = df[df['label'] == 1].copy()
    no_hate_samples = df[df['label'] == 0].copy()
    
    print(f"📊 Original distribution:")
    print(f"   - Hate: {len(hate_samples)} ({len(hate_samples)/original_size*100:.1f}%)")
    print(f"   - No Hate: {len(no_hate_samples)} ({len(no_hate_samples)/original_size*100:.1f}%)")
    
    # Determine which class needs augmentation
    if len(hate_samples) < len(no_hate_samples):
        minority_class = hate_samples
        majority_class = no_hate_samples
        minority_label = 1
        minority_name = "Hate"
        majority_name = "No Hate"
    else:
        minority_class = no_hate_samples
        majority_class = hate_samples
        minority_label = 0
        minority_name = "No Hate"
        majority_name = "Hate"
    
    print(f"🎯 Minority class: {minority_name} ({len(minority_class)} samples)")
    print(f"🎯 Majority class: {majority_name} ({len(majority_class)} samples)")
    
    # Calculate target samples for BALANCED dataset
    if Config.AUGMENTATION_TARGET_RATIO >= 1.0:
        # Perfect balance (1:1 ratio)
        target_minority_samples = len(majority_class)
        print(f"🎯 Target: PERFECT BALANCE (1:1 ratio)")
    else:
        # Specified ratio balance
        target_minority_samples = int(len(majority_class) * Config.AUGMENTATION_TARGET_RATIO)
        print(f"🎯 Target: {Config.AUGMENTATION_TARGET_RATIO*100:.0f}% balance")
    
    samples_needed = max(0, target_minority_samples - len(minority_class))
    
    print(f"📈 Target {minority_name} samples: {target_minority_samples}")
    print(f"📈 Additional samples needed: {samples_needed}")
    print(f"📈 Augmentation multiplier: {samples_needed/len(minority_class):.2f}x")
    
    if samples_needed > 0:
        print(f"🔄 Starting TDA augmentation process...")
        all_augmented_data = []
        
        # Calculate augmentation strategy
        max_augs_per_sample = Config.MAX_AUGMENTATIONS_PER_SAMPLE
        samples_per_round = len(minority_class) * max_augs_per_sample
        rounds_needed = max(1, (samples_needed + samples_per_round - 1) // samples_per_round)
        
        print(f"📈 Augmentation strategy:")
        print(f"   - Max augmentations per original sample: {max_augs_per_sample}")
        print(f"   - Samples per round: {samples_per_round}")
        print(f"   - Rounds needed: {rounds_needed}")
        
        # TDA Augmentation Rounds
        for round_num in range(rounds_needed):
            if len(all_augmented_data) >= samples_needed:
                break
            
            print(f"🔄 TDA Round {round_num + 1}/{rounds_needed}")
            round_samples_added = 0
            
            # Process each minority sample for augmentation
            for idx, (_, row) in enumerate(minority_class.iterrows()):
                if len(all_augmented_data) >= samples_needed:
                    break
                
                text = str(row['text']).strip()
                
                # Apply TDA augmentation techniques
                try:
                    augmented_texts = augmenter.augment_text(text)
                except Exception as e:
                    print(f"⚠️ Error augmenting text at index {idx}: {e}")
                    continue
                
                # Add augmented samples
                for aug_text in augmented_texts:
                    if len(all_augmented_data) >= samples_needed:
                        break
                    
                    all_augmented_data.append({
                        'text': aug_text,
                        'label': minority_label,
                        'source': f'TDA_round_{round_num}',
                        'round': round_num,
                        'original_idx': idx,
                        'augmentation_method': 'survey_based_TDA'
                    })
                    round_samples_added += 1
                
                # Progress tracking
                if (idx + 1) % 100 == 0:
                    print(f"   Processed {idx + 1}/{len(minority_class)} samples, "
                          f"Generated {round_samples_added} augmented samples")
            
            print(f"✅ Round {round_num + 1} completed: +{round_samples_added} samples")
            print(f"📊 Total augmented so far: {len(all_augmented_data)}/{samples_needed}")
        
        print(f"✅ TDA Augmentation completed!")
        print(f"📊 Generated {len(all_augmented_data)} TDA augmented samples")
        
        # Create BALANCED final dataset
        augmented_df = pd.DataFrame(all_augmented_data)
        df['source'] = 'original'
        df['round'] = -1
        df['augmentation_method'] = 'none'
        
        final_df = pd.concat([df, augmented_df], ignore_index=True)
        
        # Verify balance achievement
        final_counts = final_df['label'].value_counts().sort_index()
        print(f"\n📊 FINAL BALANCED DATASET:")
        
        total_final = len(final_df)
        for label, count in final_counts.items():
            label_name = 'Hate' if label == 1 else 'No Hate'
            percentage = count / total_final * 100
            print(f"   - {label_name}: {count} ({percentage:.1f}%)")
        
        # Calculate balance metrics
        hate_final = final_counts.get(1, 0)
        no_hate_final = final_counts.get(0, 0)
        
        if no_hate_final > 0:
            balance_ratio = hate_final / no_hate_final
            print(f"📊 Balance ratio (Hate:No Hate): {balance_ratio:.3f}:1")
            
            if 0.8 <= balance_ratio <= 1.25:
                print("✅ EXCELLENT BALANCE ACHIEVED!")
            elif 0.6 <= balance_ratio <= 1.67:
                print("✅ GOOD BALANCE ACHIEVED!")
            else:
                print("⚠️ Moderate balance - consider adjusting AUGMENTATION_TARGET_RATIO")
        
        # LOG FINAL STATISTICS
        increase = total_final - original_size
        percentage_increase = (increase / original_size) * 100
        
        print(f"\n📈 AUGMENTATION STATISTICS:")
        print(f"   - Original size: {original_size}")
        print(f"   - Final size: {total_final}")
        print(f"   - Increase: +{increase} samples ({percentage_increase:.1f}%)")
        print(f"   - Minority class multiplier: {len(all_augmented_data)/len(minority_class):.2f}x")
        
        # TDA Method breakdown
        print(f"\n🔬 TDA METHODS USED:")
        print(f"   ✅ Character-level: switch, insert, delete, keyboard typos, spelling errors")
        print(f"   ✅ Word-level: synonym replacement, random swap/delete/insert (EDA)")
        print(f"   ✅ Sentence-level: contraction transformation")
        
        return final_df
    else:
        print("✅ Dataset already balanced - no augmentation needed")
        df['source'] = 'original'
        df['round'] = -1
        df['augmentation_method'] = 'none'
        return df

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
        "accuracy": accuracy, "f1": f1, "f05": f05, "f2": f2, "auc": auc,
        "precision": precision, "recall": recall, "f1_macro": f1_macro,
        "precision_macro": precision_macro, "recall_macro": recall_macro,
        "f1_weighted": f1_weighted, "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
    }

def load_and_split_data():
    """Load and split the dataset with survey-based augmentation."""
    print("📂 Loading dataset...")
    df = pd.read_csv(Config.DATASET_PATH)
    print(f"✅ Loaded {len(df)} samples")
    
    if df['label'].dtype == 'object':
        label_map = {'noHate': 0, 'hate': 1}
        df['label'] = df['label'].map(label_map)
        print("✅ Converted string labels to integers")
    
    label_counts = df['label'].value_counts()
    print(f"📊 Original label distribution: {dict(label_counts)}")
    
    from sklearn.model_selection import train_test_split
    
    train_df, temp_df = train_test_split(
        df, test_size=(Config.VAL_SPLIT + Config.TEST_SPLIT),
        random_state=42, stratify=df['label']
    )
    
    val_df, test_df = train_test_split(
        temp_df, test_size=Config.TEST_SPLIT / (Config.VAL_SPLIT + Config.TEST_SPLIT),
        random_state=42, stratify=temp_df['label']
    )
    
    # Apply survey-based augmentation
    train_df = apply_survey_augmentation(train_df)
    
    print(f"📊 Final dataset sizes:")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    for name, data in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        hate_count = len(data[data['label'] == 1])
        no_hate_count = len(data[data['label'] == 0])
        print(f"  {name} - Hate: {hate_count} ({hate_count/len(data)*100:.1f}%), "
              f"No Hate: {no_hate_count} ({no_hate_count/len(data)*100:.1f}%)")
    
    return train_df, val_df, test_df

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
        "survey_augmentation_methods": {
            "character_level": {
                "random_character_switch": Config.CHAR_SWITCH_PROB,
                "character_insertion": Config.CHAR_INSERT_PROB,
                "character_deletion": Config.CHAR_DELETE_PROB,
                "keyboard_noise": Config.KEYBOARD_NOISE_PROB,
                "spelling_errors": Config.SPELLING_ERROR_PROB
            },
            "word_level_eda": {
                "synonym_replacement": Config.EDA_SYNONYM_REPLACE_PROB,
                "random_swap": Config.EDA_RANDOM_SWAP_PROB,
                "random_deletion": Config.EDA_RANDOM_DELETE_PROB,
                "random_insertion": Config.EDA_RANDOM_INSERT_PROB,
                "eda_alpha": Config.EDA_ALPHA
            },
            "bert_augmentation": {
                "mask_probability": Config.BERT_MASK_PROB,
                "replace_probability": Config.BERT_REPLACE_PROB
            },
            "sentence_level": {
                "contraction_transformation": Config.CONTRACTION_TRANSFORM_PROB
            }
        },
        "augmentation_target_ratio": Config.AUGMENTATION_TARGET_RATIO,
        "max_augmentations_per_sample": Config.MAX_AUGMENTATIONS_PER_SAMPLE,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    config_path = os.path.join(Config.MODEL_OUTPUT_DIR, "survey_augmentation_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"💾 Survey augmentation configuration saved to {config_path}")

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

def train_model():
    """Training function with survey-based augmentation and CSV saving."""
    print("📚 Starting DistilBERT Training with Survey-Based Data Augmentation")
    print("📊 Addressing Class Imbalance with Proven Methods")
    print("=" * 80)
    
    device = setup_gpu()
    cleanup_output_directory()
    train_df, val_df, test_df = load_and_split_data()
    
    # Save datasets to CSV BEFORE training
    print("\n💾 Saving datasets for analysis...")
    saved_paths = save_datasets_to_csv(train_df, val_df, test_df, Config)
    
    # Create augmentation examples report
    if Config.USE_SURVEY_AUGMENTATION:
        create_augmentation_examples_report(train_df, Config)
    
    if saved_paths:
        print(f"\n📁 Dataset files saved in: {os.path.dirname(saved_paths['train'])}")
        print("   You can now examine the datasets before training!")
        
        # Ask user if they want to continue with training
        print("   The saved files include:")
        print(f"   - Training dataset: {os.path.basename(saved_paths['train'])}")
        print(f"   - Validation dataset: {os.path.basename(saved_paths['validation'])}")
        print(f"   - Test dataset: {os.path.basename(saved_paths['test'])}")
        print(f"   - Summary report: {os.path.basename(saved_paths['summary'])}")
    
    print(f"\n🤖 Loading model: {Config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_NAME, num_labels=2)
    
    # Set dropout
    if hasattr(model.config, 'dropout'):
        model.config.dropout = Config.DROPOUT
    if hasattr(model.config, 'attention_dropout'):
        model.config.attention_dropout = Config.DROPOUT
    if hasattr(model.config, 'classifier_dropout'):
        model.config.classifier_dropout = Config.DROPOUT
    
    model = model.to(device)
    
    # Create datasets
    train_dataset = HateSpeechDataset(train_df, tokenizer, max_length=Config.MAX_LENGTH)
    val_dataset = HateSpeechDataset(val_df, tokenizer, max_length=Config.MAX_LENGTH)
    test_dataset = HateSpeechDataset(test_df, tokenizer, max_length=Config.MAX_LENGTH)
    
    num_training_steps = (len(train_dataset) // (Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS)) * Config.NUM_EPOCHS
    warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)
    
    training_args = TrainingArguments(
        output_dir=Config.MODEL_OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        fp16=Config.FP16,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        label_smoothing_factor=Config.LABEL_SMOOTHING,
        lr_scheduler_type=Config.SCHEDULER_TYPE,
        logging_dir=os.path.join(Config.MODEL_OUTPUT_DIR, 'logs'),
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=3,
        save_safetensors=False,
        overwrite_output_dir=True,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        push_to_hub=False,
        report_to="none",
        max_grad_norm=1.0,
        seed=42,
        data_seed=42,
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        processing_class=tokenizer
    )
    
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
            raise e
        else:
            raise e
    
    # Final evaluation
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    print("\n📊 Test Set Results:")
    for key, value in test_results.items():
        if key.startswith('test_'):
            metric_name = key.replace('test_', '').replace('_', ' ').title()
            print(f"  {metric_name}: {value:.4f}")
    
    # Generate detailed evaluation
    predictions = trainer.predict(test_dataset)
    y_pred = predictions.predictions.argmax(-1)
    y_true = predictions.label_ids
    
    target_names = ['No Hate', 'Hate']
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print("\n📊 Detailed Classification Report:")
    print(report)
    
    # Save results
    report_path = os.path.join(Config.MODEL_OUTPUT_DIR, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("DistilBERT with Survey-Based Data Augmentation - Hate Speech Detection Results\n")
        f.write("=" * 80 + "\n\n")
        f.write("Survey-Based Augmentation Methods Used for Class Imbalance:\n")
        f.write("- Character-level: Random switching, insertion, deletion, keyboard noise, spelling errors\n")
        f.write("- Word-level: Synonym replacement (WordNet), random swap, random deletion, random insertion (EDA)\n")
        f.write("- Sentence-level: Contraction handling\n\n")
        f.write(f"Target augmentation ratio: {Config.AUGMENTATION_TARGET_RATIO}\n")
        f.write(f"Max augmentations per sample: {Config.MAX_AUGMENTATIONS_PER_SAMPLE}\n\n")
        f.write(report)
        f.write("\n\nTest Set Metrics:\n")
        for key, value in test_results.items():
            if key.startswith('test_'):
                f.write(f"{key}: {value:.4f}\n")
    
    cm_path = os.path.join(Config.MODEL_OUTPUT_DIR, "confusion_matrix.pdf")
    plot_confusion_matrix(y_true, y_pred, cm_path)
    
    # Save model
    trainer.save_model(Config.MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(Config.MODEL_OUTPUT_DIR)
    
    # Save predictions
    test_predictions_path = os.path.join(Config.MODEL_OUTPUT_DIR, "test_predictions.json")
    predictions_data = {
        "predictions": y_pred.tolist(),
        "true_labels": y_true.tolist(),
        "probabilities": torch.softmax(torch.from_numpy(predictions.predictions), dim=1).numpy().tolist()
    }
    
    with open(test_predictions_path, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    
    # Save final results with dataset info
    results_path = os.path.join(Config.MODEL_OUTPUT_DIR, "training_results.json")
    results = {
        "test_results": test_results,
        "dataset_sizes": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df)
        },
        "augmentation_used": Config.USE_SURVEY_AUGMENTATION,
        "saved_datasets": saved_paths,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n" + "="*80)
    print("🎉 SURVEY-BASED AUGMENTATION TRAINING COMPLETED!")
    print("="*80)
    print(f"📁 Model and results saved in: {Config.MODEL_OUTPUT_DIR}")
    print(f"🎯 Test F1 Score: {test_results.get('test_f1', 0):.4f}")
    print(f"🎯 Test Accuracy: {test_results.get('test_accuracy', 0):.4f}")
    print("📚 Survey-based augmentation methods used:")
    print("  ✅ Character-level augmentation (5 techniques)")
    print("  ✅ Word-level augmentation (4 techniques including EDA)")
    print("  ✅ Sentence-level augmentation (contraction handling)")
    print(f"  ✅ Target class balance ratio: {Config.AUGMENTATION_TARGET_RATIO}")
    
    if saved_paths:
        print(f"\n📁 Dataset CSV files saved in: {os.path.dirname(saved_paths['train'])}")
    
    return model, tokenizer, test_results

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        Config.DATASET_PATH = sys.argv[1]
        Config.MODEL_OUTPUT_DIR = sys.argv[2]
    elif len(sys.argv) == 2:
        Config.DATASET_PATH = sys.argv[1]
        Config.MODEL_OUTPUT_DIR = "distilbert_survey_aug_" + os.path.splitext(os.path.basename(Config.DATASET_PATH))[0] + "_model"
    else:
        print("Usage: python train_distilbert_survey_augmentation.py <dataset_path> <output_dir>")
        sys.exit(1)
    
    model, tokenizer, results = train_model()