import sys
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, fbeta_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import math
import joblib
import os
import json
import scipy.sparse
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import spacy
from scipy.sparse import hstack

# Load spaCy model for POS tagging
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    raise

# Set Delta TF-IDF-specific cache directories under delta_tfidf/ subdirectory
os.environ["HF_HOME"] = "/scratch/ml23/bnge/delta_tfidf/hf_cache"
os.environ["HF_HUB_CACHE"] = "/scratch/ml23/bnge/delta_tfidf/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/scratch/ml23/bnge/delta_tfidf/hf_cache"
os.environ["XDG_CACHE_HOME"] = "/scratch/ml23/bnge/delta_tfidf/xdg_cache"
os.environ["MPLCONFIGDIR"] = "/scratch/ml23/bnge/delta_tfidf/matplotlib_cache"
os.environ["TORCH_HOME"] = "/scratch/ml23/bnge/delta_tfidf/torch_cache"
os.environ["TMPDIR"] = "/scratch/ml23/bnge/delta_tfidf/tmp"

# Create directories
cache_dirs = [
    "/scratch/ml23/bnge/delta_tfidf/hf_cache",
    "/scratch/ml23/bnge/delta_tfidf/xdg_cache",
    "/scratch/ml23/bnge/delta_tfidf/matplotlib_cache",
    "/scratch/ml23/bnge/delta_tfidf/torch_cache",
    "/scratch/ml23/bnge/delta_tfidf/tmp"
]
for cache_dir in cache_dirs:
    os.makedirs(cache_dir, exist_ok=True)
    
# Configuration for imbalance handling and POS features
class Config:
    USE_SMOTE = True  # Enable SMOTE for training data
    USE_CLASS_WEIGHTS = True  # Enable class weights in SVM
    USE_POS_FEATURES = True  # Enable POS tagging features
    SMOTE_K_NEIGHBORS = 3  # Number of neighbors for SMOTE
    # IMBALANCE_THRESHOLD = 2.0  # Only apply SMOTE if imbalance ratio > this

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

def extract_pos_features_batch(texts):
    """Extract POS features for a batch of texts."""
    print("🔄 Extracting POS features...")
    pos_features_list = []
    
    for idx, text in enumerate(texts):
        pos_feat = extract_pos_features(str(text))
        # Convert to list in consistent order
        feat_vector = [
            pos_feat['noun_ratio'], pos_feat['verb_ratio'], pos_feat['adj_ratio'],
            pos_feat['adv_ratio'], pos_feat['pronoun_ratio'], pos_feat['intj_ratio'],
            pos_feat['profanity_ratio'], pos_feat['caps_ratio']
        ]
        pos_features_list.append(feat_vector)
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(texts)} samples...")
    
    print("✅ POS feature extraction completed!")
    return np.array(pos_features_list)

def tokenize(text):
    # Simple whitespace tokenizer, but you can use nltk or spacy for better results
    return text.lower().split()

def build_ngrams(tokens, ngram_range=(1,2)):
    # Generate unigrams and bigrams
    ngrams = []
    for n in range(ngram_range[0], ngram_range[1]+1):
        ngrams += [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return ngrams

def apply_smote_resampling(X_features, y_labels, random_state=42):
    """Apply SMOTE to balance the dataset using existing features"""
    print("\n🔄 Applying SMOTE resampling...")
    print(f"Original distribution: {Counter(y_labels)}")
    
    # Apply SMOTE directly to the feature matrix (both TF-IDF and POS features combined)
    smote = SMOTE(random_state=random_state, k_neighbors=Config.SMOTE_K_NEIGHBORS)
    X_resampled, y_resampled = smote.fit_resample(X_features, y_labels)
    
    print(f"✅ SMOTE completed!")
    print(f"Resampled distribution: {Counter(y_resampled)}")
    
    return X_resampled, y_resampled

def compute_delta_tfidf_features_with_pos(texts, labels, ngram_range=(1,2), vocab=None, delta_idf=None, pos_scaler=None):
    """Compute Delta TF-IDF features combined with POS features"""
    
    # Tokenize and build ngrams for Delta TF-IDF
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

    # Build Delta TF-IDF feature matrix
    X_tfidf = scipy.sparse.lil_matrix((n_docs, len(vocab)), dtype=np.float32)
    for i, doc in enumerate(tokenized):
        counts = Counter(doc)
        for w in counts:
            if w in word2idx:
                idx = word2idx[w]
                X_tfidf[i, idx] = counts[w] * delta_idf.get(w, 0)
    X_tfidf = X_tfidf.tocsr()
    
    # Extract POS features if enabled
    if Config.USE_POS_FEATURES:
        print("🏷️ Computing POS features...")
        pos_features = extract_pos_features_batch(texts)
        
        # Normalize POS features if scaler is provided (for test data)
        if pos_scaler is not None:
            pos_features = pos_scaler.transform(pos_features)
        else:
            # Fit scaler on training data
            from sklearn.preprocessing import StandardScaler
            pos_scaler = StandardScaler()
            pos_features = pos_scaler.fit_transform(pos_features)
        
        # Convert to sparse matrix for concatenation
        pos_features_sparse = scipy.sparse.csr_matrix(pos_features)
        
        # Combine Delta TF-IDF and POS features
        X_combined = hstack([X_tfidf, pos_features_sparse])
        print(f"✅ Combined features: {X_tfidf.shape[1]} TF-IDF + {pos_features.shape[1]} POS = {X_combined.shape[1]} total")
        
        return X_combined, vocab, delta_idf, pos_scaler
    else:
        return X_tfidf, vocab, delta_idf, None

def main(dataset_path):
    # Load your data (expects columns: 'text', 'label')
    df = pd.read_csv(dataset_path)
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    
    # Create output/possmote directory structure
    output_dir = "output"
    possmote_dir = os.path.join(output_dir, "possmote")
    models_dir = os.path.join(possmote_dir, dataset_name)
    os.makedirs(models_dir, exist_ok=True)
    
    label_map = {'noHate': 0, 'hate': 1}
    if df['label'].dtype == 'object':
        df['label'] = df['label'].map(label_map)
    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()

    # Check for class imbalance
    label_counts = Counter(labels)
    print(f"📊 Original label distribution: {dict(label_counts)}")
    minority_class_count = min(label_counts.values())
    majority_class_count = max(label_counts.values())
    imbalance_ratio = majority_class_count / minority_class_count
    print(f"⚖️ Imbalance ratio: {imbalance_ratio:.2f}:1")

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

    # Cross-validation ONLY on train split
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

    # Save configuration
    config_info = {
        "dataset": dataset_name,
        "use_smote": Config.USE_SMOTE,
        "use_class_weights": Config.USE_CLASS_WEIGHTS,
        "use_pos_features": Config.USE_POS_FEATURES,
        "imbalance_ratio": imbalance_ratio,
        "smote_applied": Config.USE_SMOTE,
        "original_distribution": dict(label_counts),
        "seed": SEED
    }

    print(f"\n🚀 Starting Enhanced Delta TF-IDF Training with POS Features")
    print(f"🏷️ POS Features Enabled: {Config.USE_POS_FEATURES}")
    print(f"⚖️ SMOTE Enabled: {Config.USE_SMOTE}")
    print(f"⚖️ Class Weights Enabled: {Config.USE_CLASS_WEIGHTS}")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\n📁 Fold {fold}/10")
        X_fold_train = [X_train[i] for i in train_idx]
        y_fold_train = [y_train[i] for i in train_idx]
        X_fold_test = [X_train[i] for i in test_idx]
        y_fold_test = [y_train[i] for i in test_idx]

        # Training features and vocab with POS features
        if Config.USE_POS_FEATURES:
            X_train_feats, vocab, delta_idf, pos_scaler = compute_delta_tfidf_features_with_pos(
                X_fold_train, y_fold_train, ngram_range=(1,2)
            )
        else:
            X_train_feats, vocab, delta_idf, pos_scaler = compute_delta_tfidf_features_with_pos(
                X_fold_train, y_fold_train, ngram_range=(1,2)
            )
        
        # Apply SMOTE if enabled and dataset is imbalanced
        if Config.USE_SMOTE:
            print(f"  ⚖️ Applying SMOTE (imbalance ratio: {imbalance_ratio:.2f})")
            X_train_feats, y_fold_train = apply_smote_resampling(
                X_train_feats, y_fold_train, random_state=SEED + fold
            )
        
        # Test features using train vocab, delta_idf, and pos_scaler
        if Config.USE_POS_FEATURES:
            X_test_feats, _, _, _ = compute_delta_tfidf_features_with_pos(
                X_fold_test, y_fold_test, ngram_range=(1,2), 
                vocab=vocab, delta_idf=delta_idf, pos_scaler=pos_scaler
            )
        else:
            X_test_feats, _, _, _ = compute_delta_tfidf_features_with_pos(
                X_fold_test, y_fold_test, ngram_range=(1,2), 
                vocab=vocab, delta_idf=delta_idf, pos_scaler=pos_scaler
            )

        # Configure SVM with class weights if enabled
        svm_params = {'C': 1, 'max_iter': 5000, 'random_state': SEED}
        
        if Config.USE_CLASS_WEIGHTS:
            # Compute balanced class weights
            unique_classes = np.unique(y_fold_train)
            class_weights_array = compute_class_weight(
                'balanced', 
                classes=unique_classes, 
                y=y_fold_train
            )
            class_weights_dict = dict(zip(unique_classes, class_weights_array))
            svm_params['class_weight'] = class_weights_dict
            print(f"  📊 Using class weights: {class_weights_dict}")
        
        # Train SVM
        clf = LinearSVC(**svm_params)
        clf.fit(X_train_feats, y_fold_train)

        # Save the model after training in the models directory
        model_path = os.path.join(models_dir, f"svm_delta_tfidf_fold{fold}.joblib")
        joblib.dump(clf, model_path)

        # Save vocab, delta_idf, and pos_scaler for this fold
        vocab_path = os.path.join(models_dir, f"vocab_fold{fold}.json")
        delta_idf_path = os.path.join(models_dir, f"delta_idf_fold{fold}.json")
        with open(vocab_path, "w") as f:
            json.dump(vocab, f)
        with open(delta_idf_path, "w") as f:
            json.dump(delta_idf, f)
        
        # Save POS scaler if POS features are used
        if Config.USE_POS_FEATURES and pos_scaler is not None:
            scaler_path = os.path.join(models_dir, f"pos_scaler_fold{fold}.joblib")
            joblib.dump(pos_scaler, scaler_path)

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

        print(f"  📊 Accuracy: {acc:.4f}")
        print(f"  📊 F1 (macro): {f1:.4f}")
        print(f"  📊 F0.5 (macro): {f05:.4f}")
        print(f"  📊 F2 (macro): {f2:.4f}")
        print(f"  📊 AUC: {auc:.4f}")
        print(classification_report(y_fold_test, y_pred, target_names=['noHate', 'hate']))

        # Show top discriminative features for this fold
        coef = clf.coef_[0]
        
        # Separate TF-IDF and POS feature coefficients
        if Config.USE_POS_FEATURES:
            tfidf_coef = coef[:len(vocab)]
            pos_coef = coef[len(vocab):]
            pos_feature_names = ['noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio', 
                               'pronoun_ratio', 'intj_ratio', 'profanity_ratio', 'caps_ratio']
            
            # Top TF-IDF features
            top_pos_tfidf = np.argsort(tfidf_coef)[-10:][::-1]
            top_neg_tfidf = np.argsort(tfidf_coef)[:10]
            
            print("  🔤 Top positive (hate) TF-IDF features:")
            for idx in top_pos_tfidf:
                print(f"    {vocab[idx]}: {tfidf_coef[idx]:.3f}")
            print("  🔤 Top negative (noHate) TF-IDF features:")
            for idx in top_neg_tfidf:
                print(f"    {vocab[idx]}: {tfidf_coef[idx]:.3f}")
            
            # POS feature importance
            print("  🏷️ POS feature coefficients:")
            for i, name in enumerate(pos_feature_names):
                print(f"    {name}: {pos_coef[i]:.3f}")
        else:
            # Original TF-IDF only display
            top_pos = np.argsort(coef)[-10:][::-1]
            top_neg = np.argsort(coef)[:10]
            print("  🔤 Top positive (hate) features:")
            for idx in top_pos:
                print(f"    {vocab[idx]}: {coef[idx]:.3f}")
            print("  🔤 Top negative (noHate) features:")
            for idx in top_neg:
                print(f"    {vocab[idx]}: {coef[idx]:.3f}")

        # Weighted metrics
        weighted_f1 = f1_score(y_fold_test, y_pred, average='weighted')
        weighted_precision = precision_score(y_fold_test, y_pred, average='weighted')
        weighted_recall = recall_score(y_fold_test, y_pred, average='weighted')

        # Print comprehensive metrics (match baseline format)
        print(f"  📊 Accuracy: {acc:.4f}")
        print(f"  📊 Macro F1: {f1:.4f}")
        print(f"  📊 F0.5 Score: {f05:.4f}")
        print(f"  📊 F2 Score: {f2:.4f}")
        print(f"  📊 AUC: {auc:.4f}")
        print(f"  📊 Weighted F1: {weighted_f1:.4f}")
        print(f"  📊 Weighted Precision: {weighted_precision:.4f}")
        print(f"  📊 Weighted Recall: {weighted_recall:.4f}")

        # Save per-fold statistics with all metrics (match baseline format)
        fold_stats.append({
            "fold": fold,
            "accuracy": acc,
            "macro_f1": f1,
            "f05": f05,
            "f2": f2,
            "auc": auc,
            "weighted_f1": weighted_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "f1_hate": report['hate']['f1-score'],
            "f1_noHate": report['noHate']['f1-score'],
            "classification_report": report
        })

        # accumulate for aggregated plots
        all_y_true.append(np.array(y_fold_test))
        all_y_pred.append(np.array(y_pred))
        coef_vocab_pairs.append((coef, vocab))

    # Calculate means for all metrics (match baseline format)
    all_weighted_f1 = [stat['weighted_f1'] for stat in fold_stats]
    all_weighted_precision = [stat['weighted_precision'] for stat in fold_stats]
    all_weighted_recall = [stat['weighted_recall'] for stat in fold_stats]

    # Summary statistics (match baseline format)
    mean_acc = float(np.mean(all_acc))
    std_acc = float(np.std(all_acc))
    mean_macro_f1 = float(np.mean(all_f1))
    mean_f05 = float(np.mean(all_f05))
    mean_f2 = float(np.mean(all_f2))
    mean_auc = float(np.mean(all_auc))
    mean_weighted_f1 = float(np.mean(all_weighted_f1))
    mean_weighted_precision = float(np.mean(all_weighted_precision))
    mean_weighted_recall = float(np.mean(all_weighted_recall))
    mean_f1_hate = float(np.mean([r['hate']['f1-score'] for r in all_reports]))
    mean_f1_nohate = float(np.mean([r['noHate']['f1-score'] for r in all_reports]))

    summary_stats = {
        "dataset_info": config_info,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_macro_f1": mean_macro_f1,
        "mean_f05": mean_f05,
        "mean_f2": mean_f2,
        "mean_auc": mean_auc,
        "mean_weighted_f1": mean_weighted_f1,
        "mean_weighted_precision": mean_weighted_precision,
        "mean_weighted_recall": mean_weighted_recall,
        "mean_f1_hate": mean_f1_hate,
        "mean_f1_noHate": mean_f1_nohate,
        "folds": fold_stats
    }

    # Save statistics as JSON
    stats_path = os.path.join(models_dir, "delta_tfidf_stats.json")
    with open(stats_path, "w") as f:
        json.dump(summary_stats, f, indent=2)

    print("\n==== Cross-validation summary ====")
    print(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Macro F1: {mean_macro_f1:.4f}")
    print(f"F0.5 Score: {mean_f05:.4f}")
    print(f"F2 Score: {mean_f2:.4f}")
    print(f"AUC: {mean_auc:.4f}")
    print(f"Weighted F1: {mean_weighted_f1:.4f}")
    print(f"Weighted Precision: {mean_weighted_precision:.4f}")
    print(f"Weighted Recall: {mean_weighted_recall:.4f}")
    print(f"F1 (hate): {mean_f1_hate:.4f}")
    print(f"F1 (noHate): {mean_f1_nohate:.4f}")

    # ==== FINAL TEST SET EVALUATION ====
    print("\n" + "="*50)
    print("FINAL TEST SET EVALUATION")
    print("="*50)
    
    print(f"Training final model on full training set ({len(X_train)} samples)...")
    print(f"Evaluating on held-out test set ({len(X_test)} samples)...")
    
    # Train final model on full training set
    if Config.USE_POS_FEATURES:
        X_final_train, vocab_final, delta_idf_final, pos_scaler_final = compute_delta_tfidf_features_with_pos(
            X_train, y_train, ngram_range=(1,2)
        )
    else:
        X_final_train, vocab_final, delta_idf_final, pos_scaler_final = compute_delta_tfidf_features_with_pos(
            X_train, y_train, ngram_range=(1,2)
        )
    
    # Apply SMOTE to final training set if enabled
    if Config.USE_SMOTE:
        print(f"⚖️ Applying SMOTE to final training set...")
        X_final_train, y_final_train = apply_smote_resampling(X_final_train, y_train, random_state=SEED)
    else:
        y_final_train = y_train
    
    # Transform test set using training vocabulary, delta_idf, and pos_scaler
    if Config.USE_POS_FEATURES:
        X_final_test, _, _, _ = compute_delta_tfidf_features_with_pos(
            X_test, y_test, ngram_range=(1,2), vocab=vocab_final, 
            delta_idf=delta_idf_final, pos_scaler=pos_scaler_final
        )
    else:
        X_final_test, _, _, _ = compute_delta_tfidf_features_with_pos(
            X_test, y_test, ngram_range=(1,2), vocab=vocab_final, 
            delta_idf=delta_idf_final, pos_scaler=pos_scaler_final
        )
    
    # Configure final SVM with class weights if enabled
    svm_params = {'C': 1, 'max_iter': 5000, 'random_state': SEED}
    
    if Config.USE_CLASS_WEIGHTS:
        unique_classes = np.unique(y_final_train)
        class_weights_array = compute_class_weight('balanced', classes=unique_classes, y=y_final_train)
        class_weights_dict = dict(zip(unique_classes, class_weights_array))
        svm_params['class_weight'] = class_weights_dict
    
    # Train final SVM model
    final_clf = LinearSVC(**svm_params)
    final_clf.fit(X_final_train, y_final_train)
    
    # Save final model and artifacts
    final_model_path = os.path.join(models_dir, "svm_delta_tfidf_final.joblib")
    joblib.dump(final_clf, final_model_path)
    
    final_vocab_path = os.path.join(models_dir, "vocab_final.json")
    final_delta_idf_path = os.path.join(models_dir, "delta_idf_final.json")
    with open(final_vocab_path, "w") as f:
        json.dump(vocab_final, f)
    with open(final_delta_idf_path, "w") as f:
        json.dump(delta_idf_final, f)
    
    if Config.USE_POS_FEATURES and pos_scaler_final is not None:
        final_scaler_path = os.path.join(models_dir, "pos_scaler_final.joblib")
        joblib.dump(pos_scaler_final, final_scaler_path)
    
    # Predict on test set
    y_test_pred = final_clf.predict(X_final_test)
    y_test_scores = final_clf.decision_function(X_final_test)
    
    # Calculate comprehensive test metrics
    test_acc = accuracy_score(y_test, y_test_pred)
    test_report = classification_report(y_test, y_test_pred, target_names=['noHate', 'hate'], output_dict=True)
    test_macro_f1 = test_report['macro avg']['f1-score']
    test_f05 = fbeta_score(y_test, y_test_pred, beta=0.5, average='macro')
    test_f2 = fbeta_score(y_test, y_test_pred, beta=2, average='macro')
    test_auc = roc_auc_score(y_test, y_test_scores)
    test_weighted_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_weighted_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_weighted_recall = recall_score(y_test, y_test_pred, average='weighted')
    
    # Print comprehensive test results
    print(f"\nTEST SET RESULTS:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Macro F1: {test_macro_f1:.4f}")
    print(f"F0.5 Score: {test_f05:.4f}")
    print(f"F2 Score: {test_f2:.4f}")
    print(f"AUC: {test_auc:.4f}")
    print(f"Weighted F1: {test_weighted_f1:.4f}")
    print(f"Weighted Precision: {test_weighted_precision:.4f}")
    print(f"Weighted Recall: {test_weighted_recall:.4f}")
    print(f"F1 (hate): {test_report['hate']['f1-score']:.4f}")
    print(f"F1 (noHate): {test_report['noHate']['f1-score']:.4f}")
    print("\nDetailed Test Set Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['noHate', 'hate']))
    
    # Test set confusion matrix
    test_cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nTest Set Confusion Matrix:")
    print(f"              Predicted")
    print(f"              No Hate  Hate")
    print(f"Actual No Hate   {test_cm[0,0]:4d}   {test_cm[0,1]:4d}")
    print(f"       Hate      {test_cm[1,0]:4d}   {test_cm[1,1]:4d}")
    
    # Show top discriminative features from final model
    final_coef = final_clf.coef_[0]
    
    if Config.USE_POS_FEATURES:
        tfidf_coef_final = final_coef[:len(vocab_final)]
        pos_coef_final = final_coef[len(vocab_final):]
        pos_feature_names = ['noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio', 
                           'pronoun_ratio', 'intj_ratio', 'profanity_ratio', 'caps_ratio']
        
        top_pos_final = np.argsort(tfidf_coef_final)[-15:][::-1]
        top_neg_final = np.argsort(tfidf_coef_final)[:15]
        
        print("\nTop discriminative features (final model):")
        print("Top positive (hate) features:")
        for idx in top_pos_final:
            print(f"  {vocab_final[idx]}: {tfidf_coef_final[idx]:.3f}")
        print("Top negative (noHate) features:")
        for idx in top_neg_final:
            print(f"  {vocab_final[idx]}: {tfidf_coef_final[idx]:.3f}")
        
        print("\nPOS feature coefficients (final model):")
        for i, name in enumerate(pos_feature_names):
            print(f"  {name}: {pos_coef_final[i]:.3f}")
    else:
        top_pos_final = np.argsort(final_coef)[-15:][::-1]
        top_neg_final = np.argsort(final_coef)[:15]
        
        print("\nTop discriminative features (final model):")
        print("Top positive (hate) features:")
        for idx in top_pos_final:
            print(f"  {vocab_final[idx]}: {final_coef[idx]:.3f}")
        print("Top negative (noHate) features:")
        for idx in top_neg_final:
            print(f"  {vocab_final[idx]}: {final_coef[idx]:.3f}")
    
    # Add comprehensive test results to summary stats
    summary_stats.update({
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_macro_f1),
        "test_f05": float(test_f05),
        "test_f2": float(test_f2),
        "test_auc": float(test_auc),
        "test_weighted_f1": float(test_weighted_f1),
        "test_weighted_precision": float(test_weighted_precision),
        "test_weighted_recall": float(test_weighted_recall),
        "test_f1_hate": float(test_report['hate']['f1-score']),
        "test_f1_noHate": float(test_report['noHate']['f1-score']),
        "test_classification_report": test_report,
        "test_confusion_matrix": test_cm.tolist()
    })
    
    # Save updated statistics
    with open(stats_path, "w") as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"\n✅ Final model saved to: {final_model_path}")
    print(f"✅ Final vocabulary saved to: {final_vocab_path}")
    print(f"✅ Final delta-idf saved to: {final_delta_idf_path}")
    print(f"✅ Complete statistics saved to: {stats_path}")

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
        
        # Add enhancement info to plot
        plt.figtext(0.02, 0.02, 
                   f"POS: {Config.USE_POS_FEATURES}, SMOTE: {config_info['smote_applied']}, Class Weights: {Config.USE_CLASS_WEIGHTS}", 
                   fontsize=8, style='italic')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        print(f"📊 Aggregated confusion matrix saved to {save_path}")

    def plot_aggregated_top_features(coef_vocab_pairs, top_k=20, save_path=None):
        sums = {}
        counts = {}
        for coef, vocab in coef_vocab_pairs:
            # Only consider TF-IDF features for top features plot
            tfidf_coef = coef[:len(vocab)] if Config.USE_POS_FEATURES else coef
            for i, token in enumerate(vocab):
                val = float(tfidf_coef[i])
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
        ax1.set_title(f"Top {top_k} Positive (hate) TF-IDF features")
        ax1.set_xlabel("Avg coefficient")
        ax2.barh(neg_tokens[::-1], [v for v in neg_vals[::-1]], color='steelblue')
        ax2.set_title(f"Top {top_k} Negative (noHate) TF-IDF features")
        ax2.set_xlabel("Avg coefficient")
        
        # Add enhancement info to plot
        plt.figtext(0.02, 0.02, 
                   f"POS: {Config.USE_POS_FEATURES}, SMOTE: {config_info['smote_applied']}, Class Weights: {Config.USE_CLASS_WEIGHTS}", 
                   fontsize=10, style='italic')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
            plt.close()
            print(f"📊 Aggregated top-features plot saved to {save_path}")
        else:
            plt.show()

    def plot_pos_feature_importance(coef_vocab_pairs, save_path=None):
        """Plot POS feature importance across all folds."""
        if not Config.USE_POS_FEATURES:
            return
            
        pos_feature_names = ['noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio', 
                           'pronoun_ratio', 'intj_ratio', 'profanity_ratio', 'caps_ratio']
        
        # Collect POS coefficients from all folds
        pos_coefs = []
        for coef, vocab in coef_vocab_pairs:
            pos_coef = coef[len(vocab):]  # POS features are after TF-IDF features
            pos_coefs.append(pos_coef)
        
        # Calculate mean and std for each POS feature
        pos_coefs = np.array(pos_coefs)
        mean_coefs = np.mean(pos_coefs, axis=0)
        std_coefs = np.std(pos_coefs, axis=0)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(pos_feature_names))
        colors = ['firebrick' if c > 0 else 'steelblue' for c in mean_coefs]
        
        bars = plt.bar(x_pos, mean_coefs, yerr=std_coefs, capsize=5, color=colors, alpha=0.7)
        plt.xlabel('POS Features')
        plt.ylabel('Average Coefficient')
        plt.title('POS Feature Importance (Average across all folds)')
        plt.xticks(x_pos, pos_feature_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, coef, std in zip(bars, mean_coefs, std_coefs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01, 
                    f'{coef:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
            plt.close()
            print(f"📊 POS feature importance plot saved to {save_path}")
        else:
            plt.show()

    # Generate all plots
    agg_cm_path = os.path.join(models_dir, "aggregated_confusion_matrix.pdf")
    plot_aggregated_confusion(all_y_true, all_y_pred, agg_cm_path)

    top_feats_path = os.path.join(models_dir, "aggregated_top_tfidf_features.pdf")
    plot_aggregated_top_features(coef_vocab_pairs, top_k=20, save_path=top_feats_path)

    if Config.USE_POS_FEATURES:
        pos_feats_path = os.path.join(models_dir, "pos_feature_importance.pdf")
        plot_pos_feature_importance(coef_vocab_pairs, save_path=pos_feats_path)

    # Save a comprehensive summary report
    report_path = os.path.join(models_dir, "enhanced_delta_tfidf_report.txt")
    with open(report_path, 'w') as f:
        f.write("Enhanced Delta TF-IDF SVM with POS Features - Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Original imbalance ratio: {imbalance_ratio:.2f}:1\n")
        f.write(f"Original distribution: {config_info['original_distribution']}\n\n")
        f.write("Enhancement Techniques:\n")
        f.write(f"- POS features used: {Config.USE_POS_FEATURES}\n")
        f.write(f"- SMOTE applied: {config_info['smote_applied']}\n")
        f.write(f"- Class weights used: {Config.USE_CLASS_WEIGHTS}\n\n")
        f.write("Cross-validation Results (10-fold):\n")
        f.write(f"- Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n")
        f.write(f"- Mean F1 (macro): {mean_f1:.4f}\n")
        f.write(f"- Mean F0.5 (macro): {mean_f05:.4f}\n")
        f.write(f"- Mean F2 (macro): {mean_f2:.4f}\n")
        f.write(f"- Mean AUC: {mean_auc:.4f}\n")
        f.write(f"- Mean F1 (hate): {mean_f1_hate:.4f}\n")
        f.write(f"- Mean F1 (noHate): {mean_f1_nohate:.4f}\n")

    print(f"\n✅ Enhanced Delta TF-IDF with POS features training completed!")
    print(f"📁 Results saved in: {models_dir}")
    print(f"🏷️ POS features used: {Config.USE_POS_FEATURES}")
    print(f"⚖️ SMOTE applied: {config_info['smote_applied']}")
    print(f"⚖️ Class weights used: {Config.USE_CLASS_WEIGHTS}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python delta_tfidf_pos.py <dataset_path>")
        sys.exit(1)