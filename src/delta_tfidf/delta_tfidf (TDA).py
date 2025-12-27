import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, fbeta_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from collections import Counter
import math
import joblib
import os
import json
import scipy.sparse
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

# Set Delta TF-IDF-specific cache directories under delta_tfidf/ subdirectory
os.environ["HF_HOME"] = "/scratch/ml23/bnge/delta_tfidf/hf_cache"
os.environ["HF_HUB_CACHE"] = "/scratch/ml23/bnge/delta_tfidf/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/scratch/ml23/bnge/delta_tfidf/hf_cache"
os.environ["XDG_CACHE_HOME"] = "/scratch/ml23/bnge/delta_tfidf/xdg_cache"
os.environ["MPLCONFIGDIR"] = "/scratch/ml23/bnge/delta_tfidf/matplotlib_cache"
os.environ["TORCH_HOME"] = "/scratch/ml23/bnge/delta_tfidf/torch_cache"
os.environ["TMPDIR"] = "/scratch/ml23/bnge/delta_tfidf/tmp"

# Set NLTK data path
nltk_data_path = "/scratch/ml23/bnge/distilbert/nltk_data"
os.environ["NLTK_DATA"] = nltk_data_path
nltk.data.path.append(nltk_data_path)

# Create directories
cache_dirs = [
    "/scratch/ml23/bnge/delta_tfidf/hf_cache",
    "/scratch/ml23/bnge/delta_tfidf/xdg_cache",
    "/scratch/ml23/bnge/delta_tfidf/matplotlib_cache",
    "/scratch/ml23/bnge/delta_tfidf/torch_cache",
    "/scratch/ml23/bnge/delta_tfidf/tmp"
]

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

# Import torch for augmentation (if available)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - paraphrasing disabled")

class Config:
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

def tokenize(text):
    # Simple whitespace tokenizer, but you can use nltk or spacy for better results
    return text.lower().split()

def build_ngrams(tokens, ngram_range=(1,2)):
    # Generate unigrams and bigrams
    ngrams = []
    for n in range(ngram_range[0], ngram_range[1]+1):
        ngrams += [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return ngrams

def compute_delta_tfidf_features(texts, labels, ngram_range=(1,2), vocab=None, delta_idf=None):
    # Tokenize and build ngrams
    tokenized = [build_ngrams(tokenize(text), ngram_range) for text in texts]
    if vocab is None:
        vocab = sorted(set(word for doc in tokenized for word in doc))
    word2idx = {w: i for i, w in enumerate(vocab)}
    n_docs = len(texts)

    # If delta_idf is not provided, compute it from labels
    if delta_idf is None:
        pos_docs = [i for i, l in enumerate(labels) if l == 1]
        neg_docs = [i for i, l in enumerate(labels) if l == 0]
        df_pos = Counter()
        df_neg = Counter()
        for i in pos_docs:
            for w in set(tokenized[i]):
                df_pos[w] += 1
        for i in neg_docs:
            for w in set(tokenized[i]):
                df_neg[w] += 1
        P = len(pos_docs)
        N = len(neg_docs)
        delta_idf = {}
        for w in vocab:
            Pt = df_pos[w] + 1
            Nt = df_neg[w] + 1
            delta_idf[w] = math.log2((Pt / P) / (Nt / N))

    # Build feature matrix
    X = scipy.sparse.lil_matrix((n_docs, len(vocab)), dtype=np.float32)
    for i, doc in enumerate(tokenized):
        counts = Counter(doc)
        for w in counts:
            if w in word2idx:
                idx = word2idx[w]
                X[i, idx] = counts[w] * delta_idf.get(w, 0)
    return X.tocsr(), vocab, delta_idf

def main(dataset_path):
    # Load your data (expects columns: 'text', 'label')
    df = pd.read_csv(dataset_path)
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    models_dir = os.path.join("models", dataset_name)
    os.makedirs(models_dir, exist_ok=True)
    label_map = {'noHate': 0, 'hate': 1}
    if df['label'].dtype == 'object':
        df['label'] = df['label'].map(label_map)
    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()

    # Set seed for reproducibility
    SEED = 42

    # Split into train/val/test (e.g., 80/10/10)
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=0.1, random_state=SEED, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1111, random_state=SEED, stratify=y_temp
    )
    # Now X_train, X_val, X_test are fixed for this seed

    # Apply survey-based augmentation to training data only
    train_df = pd.DataFrame({'text': X_train, 'label': y_train})
    train_df = apply_survey_augmentation(train_df)
    X_train_aug = train_df['text'].tolist()
    y_train_aug = train_df['label'].tolist()

    # Cross-validation ONLY on augmented train split
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    all_reports = []
    all_acc = []
    all_f1 = []
    all_f05 = []
    all_f2 = []
    all_auc = []
    fold_stats = []
    all_y_true = []
    all_y_pred = []
    coef_vocab_pairs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_train_aug, y_train_aug), 1):
        print(f"\nFold {fold}/10")
        X_fold_train = [X_train_aug[i] for i in train_idx]
        y_fold_train = [y_train_aug[i] for i in train_idx]
        X_fold_test = [X_train_aug[i] for i in test_idx]
        y_fold_test = [y_train_aug[i] for i in test_idx]

        # Training features and vocab
        X_train_feats, vocab, delta_idf = compute_delta_tfidf_features(X_fold_train, y_fold_train, ngram_range=(1,2))
        # Test features using train vocab and delta_idf
        X_test_feats, _, _ = compute_delta_tfidf_features(X_fold_test, y_fold_test, ngram_range=(1,2), vocab=vocab, delta_idf=delta_idf)

        # Train SVM
        clf = LinearSVC(C=1, max_iter=5000)
        clf.fit(X_train_feats, y_fold_train)

        # Save the model after training in the models directory
        model_path = os.path.join(models_dir, f"svm_delta_tfidf_fold{fold}.joblib")
        joblib.dump(clf, model_path)

        # Save vocab and delta_idf for this fold
        vocab_path = os.path.join(models_dir, f"vocab_fold{fold}.json")
        delta_idf_path = os.path.join(models_dir, f"delta_idf_fold{fold}.json")
        with open(vocab_path, "w") as f:
            json.dump(vocab, f)
        with open(delta_idf_path, "w") as f:
            json.dump(delta_idf, f)

        y_pred = clf.predict(X_test_feats)
        acc = accuracy_score(y_fold_test, y_pred)
        report = classification_report(y_fold_test, y_pred, target_names=['noHate', 'hate'], output_dict=True)

        # Compute additional metrics
        f1 = report['macro avg']['f1-score']
        f05 = fbeta_score(y_fold_test, y_pred, beta=0.5, average='macro')
        f2 = fbeta_score(y_fold_test, y_pred, beta=2, average='macro')
        y_scores = clf.decision_function(X_test_feats)
        auc = roc_auc_score(y_fold_test, y_scores)

        all_acc.append(acc)
        all_f1.append(f1)
        all_f05.append(f05)
        all_f2.append(f2)
        all_auc.append(auc)
        all_reports.append(report)

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 (macro): {f1:.4f}")
        print(f"F0.5 (macro): {f05:.4f}")
        print(f"F2 (macro): {f2:.4f}")
        print(f"AUC: {auc:.4f}")
        print(classification_report(y_fold_test, y_pred, target_names=['noHate', 'hate']))

        # Show top discriminative features for this fold
        coef = clf.coef_[0]
        top_pos = np.argsort(coef)[-10:][::-1]
        top_neg = np.argsort(coef)[:10]
        print("Top positive (hate) features:")
        for idx in top_pos:
            print(f"{vocab[idx]}: {coef[idx]:.3f}")
        print("Top negative (noHate) features:")
        for idx in top_neg:
            print(f"{vocab[idx]}: {coef[idx]:.3f}")

        # Save per-fold statistics
        fold_stats.append({
            "fold": fold,
            "accuracy": acc,
            "f1": f1,
            "f05": f05,
            "f2": f2,
            "auc": auc,
            "f1_hate": report['hate']['f1-score'],
            "f1_noHate": report['noHate']['f1-score'],
            "classification_report": report
        })

        # accumulate for aggregated plots
        all_y_true.append(np.array(y_fold_test))
        all_y_pred.append(np.array(y_pred))
        coef_vocab_pairs.append((coef, vocab))

    # Summary statistics
    mean_acc = float(np.mean(all_acc))
    std_acc = float(np.std(all_acc))
    mean_f1 = float(np.mean(all_f1))
    mean_f05 = float(np.mean(all_f05))
    mean_f2 = float(np.mean(all_f2))
    mean_auc = float(np.mean(all_auc))
    mean_f1_hate = float(np.mean([r['hate']['f1-score'] for r in all_reports]))
    mean_f1_nohate = float(np.mean([r['noHate']['f1-score'] for r in all_reports]))

    summary_stats = {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_f1": mean_f1,
        "mean_f05": mean_f05,
        "mean_f2": mean_f2,
        "mean_auc": mean_auc,
        "mean_f1_hate": mean_f1_hate,
        "mean_f1_noHate": mean_f1_nohate,
        "folds": fold_stats
    }

    # Save statistics as JSON
    stats_path = os.path.join(models_dir, "delta_tfidf_stats.json")
    with open(stats_path, "w") as f:
        json.dump(summary_stats, f, indent=2)

    print("\n==== Cross-validation summary ====")
    print(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Mean F1 (macro): {mean_f1:.4f}")
    print(f"Mean F0.5 (macro): {mean_f05:.4f}")
    print(f"Mean F2 (macro): {mean_f2:.4f}")
    print(f"Mean AUC: {mean_auc:.4f}")
    print(f"Mean F1 (hate): {mean_f1_hate:.4f}")
    print(f"Mean F1 (noHate): {mean_f1_nohate:.4f}")

    # -----------------------
    # AGGREGATED PLOTS (save as PDF)
    # -----------------------
    def plot_aggregated_confusion(y_true_list, y_pred_list, save_path):
        y_true_all = np.concatenate(y_true_list)
        y_pred_all = np.concatenate(y_pred_list)
        cm = confusion_matrix(y_true_all, y_pred_all)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Hate','Hate'], yticklabels=['No Hate','Hate'])
        plt.title('Aggregated Confusion Matrix (all folds)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        print(f"Aggregated confusion matrix saved to {save_path}")

    def plot_aggregated_top_features(coef_vocab_pairs, top_k=20, save_path=None):
        sums = {}
        counts = {}
        for coef, vocab in coef_vocab_pairs:
            for i, token in enumerate(vocab):
                val = float(coef[i])
                sums[token] = sums.get(token, 0.0) + val
                counts[token] = counts.get(token, 0) + 1
        avg = {token: sums[token]/counts[token] for token in sums}
        sorted_items = sorted(avg.items(), key=lambda x: x[1])
        top_neg = sorted_items[:top_k][::-1]
        top_pos = sorted_items[-top_k:][::-1]
        pos_tokens, pos_vals = zip(*top_pos) if top_pos else ([],[])
        neg_tokens, neg_vals = zip(*top_neg) if top_neg else ([],[])
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,8))
        ax1.barh(pos_tokens[::-1], pos_vals[::-1], color='firebrick')
        ax1.set_title(f"Top {top_k} Positive (hate) features")
        ax1.set_xlabel("Avg coefficient")
        ax2.barh(neg_tokens[::-1], [v for v in neg_vals[::-1]], color='steelblue')
        ax2.set_title(f"Top {top_k} Negative (noHate) features")
        ax2.set_xlabel("Avg coefficient")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
            plt.close()
            print(f"Aggregated top-features plot saved to {save_path}")
        else:
            plt.show()

    agg_cm_path = os.path.join(models_dir, "aggregated_confusion_matrix.pdf")
    plot_aggregated_confusion(all_y_true, all_y_pred, agg_cm_path)

    top_feats_path = os.path.join(models_dir, "aggregated_top_features.pdf")
    plot_aggregated_top_features(coef_vocab_pairs, top_k=20, save_path=top_feats_path)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()