import sys
import pandas as pd
import numpy as np
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

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\nFold {fold}/10")
        X_fold_train = [X_train[i] for i in train_idx]
        y_fold_train = [y_train[i] for i in train_idx]
        X_fold_test = [X_train[i] for i in test_idx]
        y_fold_test = [y_train[i] for i in test_idx]

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