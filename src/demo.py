import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go
import json
import os
import numpy as np
import joblib
import scipy.sparse
from collections import Counter
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Hate Speech Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .model-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .hate-result {
        background-color: #ffebee;
        border-left-color: #f44336 !important;
    }
    .no-hate-result {
        background-color: #e8f5e9;
        border-left-color: #4caf50 !important;
    }
    .metric-box {
        text-align: center;
        padding: 0.5rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def tokenize(text):
    return text.lower().split()

def build_ngrams(tokens, ngram_range=(1,2)):
    ngrams = []
    for n in range(ngram_range[0], ngram_range[1]+1):
        ngrams += [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return ngrams

def compute_delta_tfidf_features_single(text, vocab, delta_idf, ngram_range=(1,2)):
    tokens = build_ngrams(tokenize(text), ngram_range)
    word2idx = {w: i for i, w in enumerate(vocab)}
    X = scipy.sparse.lil_matrix((1, len(vocab)), dtype=np.float32)
    counts = Counter(tokens)
    for w in counts:
        if w in word2idx:
            idx = word2idx[w]
            X[0, idx] = counts[w] * delta_idf.get(w, 0)
    return X.tocsr()

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model(model_key):
    """Load a single model on demand."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_key == 'roberta':
        try:
            tokenizer = AutoTokenizer.from_pretrained("models/roberta_hate_speech_model")
            model = AutoModelForSequenceClassification.from_pretrained("models/roberta_hate_speech_model")
            model.to(device)
            model.eval()
            return {
                'model': model, 
                'tokenizer': tokenizer, 
                'device': device,
                'name': 'RoBERTa-SMOTE',
                'max_length': 256
            }
        except Exception as e:
            st.error(f"RoBERTa model loading failed: {e}")
            return None
    
    elif model_key == 'distilbert':
        try:
            tokenizer = AutoTokenizer.from_pretrained("models/distilbert_hate_speech_model")
            model = AutoModelForSequenceClassification.from_pretrained("models/distilbert_hate_speech_model")
            model.to(device)
            model.eval()
            return {
                'model': model, 
                'tokenizer': tokenizer, 
                'device': device,
                'name': 'DistilBERT-SMOTE',
                'max_length': 256
            }
        except Exception as e:
            st.error(f"DistilBERT model loading failed: {e}")
            return None
    
    # elif model_key == 'gemma':
    #     try:
    #         from peft import PeftModel
    #         tokenizer = AutoTokenizer.from_pretrained("models/gemma_hate_speech_model")
    #         model = AutoModelForSequenceClassification.from_pretrained(
    #             "models/gemma_hate_speech_model",
    #             dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    #             device_map="auto" if torch.cuda.is_available() else None
    #         )
    #         if torch.cuda.is_available():
    #             model.eval()
    #         return {
    #             'model': model,
    #             'tokenizer': tokenizer,
    #             'device': device,
    #             'name': 'Gemma-7B-POS',
    #             'max_length': 256
    #         }
    #     except Exception as e:
    #         st.error(f"Gemma model loading failed: {e}")
    #         return None
    
    elif model_key == 'delta_tfidf':
        try:
            svm_models, vocabs, delta_idfs = [], [], []
            for fold in range(1, 11):
                model_file = f"models/delta_tfidf_hate_speech_model/svm_delta_tfidf_fold{fold}.joblib"
                vocab_file = f"models/delta_tfidf_hate_speech_model/vocab_fold{fold}.json"
                delta_idf_file = f"models/delta_tfidf_hate_speech_model/delta_idf_fold{fold}.json"
                
                if os.path.exists(model_file):
                    svm_models.append(joblib.load(model_file))
                    with open(vocab_file, 'r') as f:
                        vocabs.append(json.load(f))
                    with open(delta_idf_file, 'r') as f:
                        delta_idfs.append(json.load(f))
            
            if svm_models:
                return {
                    'models': svm_models, 
                    'vocabs': vocabs, 
                    'delta_idfs': delta_idfs,
                    'name': 'Delta TF-IDF TDA',
                    'num_folds': len(svm_models)
                }
        except Exception as e:
            st.error(f"Delta TF-IDF model loading failed: {e}")
            return None
    
    return None

# ============================================================================
# PREDICTION
# ============================================================================

def predict_transformer(text, model_info, threshold=0.5):
    """Predict using RoBERTa or DistilBERT."""
    try:
        # DistilBERT doesn't use token_type_ids, RoBERTa also doesn't use them
        tokenizer_kwargs = {
            'text': text,
            'return_tensors': "pt",
            'truncation': True,
            'max_length': model_info['max_length'],
            'padding': "max_length",
            'return_token_type_ids': False  # Neither RoBERTa nor DistilBERT use this
        }
        
        inputs = model_info['tokenizer'](**tokenizer_kwargs)
        
        # Remove token_type_ids if present (some tokenizers might still return it)
        if 'token_type_ids' in inputs:
            inputs.pop('token_type_ids')
        
        inputs = {k: v.to(model_info['device']) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = model_info['model'](**inputs).logits
            probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
        
        if isinstance(probs, float):
            probs = [1 - probs, probs]
        
        return {
            'hate_prob': probs[1],
            'prediction': 'Hate' if probs[1] >= threshold else 'No Hate',
            'confidence': max(probs)
        }
    except Exception as e:
        st.error(f"{model_info['name']} prediction error: {e}")
        return None

def predict_delta_tfidf(text, model_info, threshold=0.5):
    """Predict using Delta TF-IDF SVM ensemble."""
    try:
        scores = []
        for model, vocab, delta_idf in zip(
            model_info['models'], 
            model_info['vocabs'], 
            model_info['delta_idfs']
        ):
            X = compute_delta_tfidf_features_single(text, vocab, delta_idf)
            scores.append(model.decision_function(X)[0])
        
        avg_score = np.mean(scores)
        hate_prob = 1 / (1 + np.exp(-avg_score))
        
        return {
            'hate_prob': float(hate_prob),
            'prediction': 'Hate' if hate_prob >= threshold else 'No Hate',
            'confidence': float(max(hate_prob, 1 - hate_prob)),
            'agreement': float(np.mean([1 if s > 0 else 0 for s in scores]))
        }
    except Exception as e:
        st.error(f"Delta TF-IDF prediction error: {e}")
        return None

def predict_all(text, models, threshold=0.5):
    """Get predictions from all loaded models."""
    results = {}
    
    # Transformer models (RoBERTa, DistilBERT)
    for key in ['roberta', 'distilbert']:  # 'gemma' removed
        if key in models and models[key] is not None:
            result = predict_transformer(text, models[key], threshold)
            if result:
                results[models[key]['name']] = result
    
    # Delta TF-IDF
    if 'delta_tfidf' in models and models['delta_tfidf'] is not None:
        result = predict_delta_tfidf(text, models['delta_tfidf'], threshold)
        if result:
            results[models['delta_tfidf']['name']] = result
    
    return results

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header with logo
    logo_path = "models/ailecs_logo.png"
    if os.path.exists(logo_path):
        col_logo, col_title = st.columns([1, 5])
        with col_logo:
            st.image(Image.open(logo_path), width='stretch')
        with col_title:
            st.markdown("""
                <div style='display: flex; align-items: center; height: 150px;'>
                    <h1 style='margin: 0;'>AiLECS Hate Speech Detection</h1>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.title("AiLECS Hate Speech Detection")
    
    # Model selection
    st.markdown("### Select Models")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_roberta = st.checkbox("RoBERTa-SMOTE", value=True)
    with col2:
        use_distilbert = st.checkbox("DistilBERT-SMOTE", value=True)
    with col3:
        use_delta_tfidf = st.checkbox("Delta TF-IDF TDA", value=True)
    # with col4:
    #     use_gemma = st.checkbox("Gemma-7B-POS", value=False)
    
    # Load selected models
    models = {}
    with st.spinner("Loading selected models..."):
        if use_roberta:
            model = load_model('roberta')
            if model:
                models['roberta'] = model
        if use_distilbert:
            model = load_model('distilbert')
            if model:
                models['distilbert'] = model
        # if use_gemma:
        #     model = load_model('gemma')
        #     if model:
        #         models['gemma'] = model
        if use_delta_tfidf:
            model = load_model('delta_tfidf')
            if model:
                models['delta_tfidf'] = model
    
    if not models:
        st.error("No models loaded. Please select at least one model and check model paths.")
        return
    
    model_names = [info.get('name', key) for key, info in models.items()]
    st.success(f"Loaded {len(models)} models: {', '.join(model_names)}")
    
    # Check if GPU is available
    if torch.cuda.is_available():
        st.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Single input area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text = st.text_area(
            "Enter text to analyze:", 
            height=100, 
            placeholder="Type or paste text here..."
        )
    
    with col2:
        st.write("")
        st.write("")
        threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.05)
        analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    if analyze and text.strip():
        with st.spinner("Analyzing with all models..."):
            results = predict_all(text, models, threshold)
        
        if not results:
            st.error("No predictions generated. Please check model status.")
            return
        
        # Display results side by side
        st.markdown("### 📊 Results")
        
        cols = st.columns(len(results))
        
        for idx, (model_name, result) in enumerate(results.items()):
            with cols[idx]:
                result_class = "hate-result" if result['prediction'] == 'Hate' else "no-hate-result"
                
                st.markdown(f"""
                    <div class="model-card {result_class}">
                        <h4 style="margin: 0 0 1rem 0;">{model_name}</h4>
                        <div class="metric-box">
                            <div class="metric-value">{result['prediction']}</div>
                            <div class="metric-label">Prediction</div>
                        </div>
                        <hr style="margin: 1rem 0;">
                        <div class="metric-box">
                            <div class="metric-value">{result['hate_prob']*100:.1f}%</div>
                            <div class="metric-label">Hate Probability</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{result['confidence']*100:.1f}%</div>
                            <div class="metric-label">Confidence</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show agreement for ensemble
                if 'agreement' in result:
                    st.caption(f"Fold Agreement: {result['agreement']*100:.0f}%")
        
        # Comparison chart
        st.markdown("### 📈 Visual Comparison")
        
        fig = go.Figure()
        
        model_names_list = list(results.keys())
        hate_probs = [r['hate_prob'] * 100 for r in results.values()]
        
        fig.add_trace(go.Bar(
            x=model_names_list,
            y=hate_probs,
            marker_color=['#f44336' if p > 50 else '#4caf50' for p in hate_probs],
            text=[f"{p:.1f}%" for p in hate_probs],
            textposition='auto',
        ))
        
        fig.add_hline(
            y=threshold * 100, 
            line_dash="dash", 
            line_color="gray", 
            annotation_text=f"Threshold ({threshold*100:.0f}%)"
        )
        
        fig.update_layout(
            title="Hate Speech Probability by Model",
            yaxis_title="Probability (%)",
            yaxis_range=[0, 100],
            height=300,
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model agreement analysis
        if len(results) > 1:
            st.markdown("### 🤝 Model Agreement")
            
            predictions = [r['prediction'] for r in results.values()]
            hate_votes = predictions.count('Hate')
            no_hate_votes = predictions.count('No Hate')
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Hate Votes", hate_votes)
            with col2:
                st.metric("No Hate Votes", no_hate_votes)
            with col3:
                agreement = (max(hate_votes, no_hate_votes) / len(predictions)) * 100
                st.metric("Agreement", f"{agreement:.0f}%")
            
            if hate_votes == no_hate_votes:
                st.warning("⚠️ Models are split - consider reviewing manually")
            elif agreement == 100:
                st.success("✅ All models agree")
            else:
                st.info(f"ℹ️ Majority prediction: {predictions[0] if hate_votes > no_hate_votes else 'No Hate'}")

if __name__ == "__main__":
    main()