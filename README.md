# Hate Speech Detection Models

This repository contains multiple hate speech detection models trained on various datasets, along with their training scripts and evaluation metrics.

## Project Structure

```
src/
├── demo.py                          # Streamlit demo application
├── delta_tfidf/                     # Delta TF-IDF training scripts
├── distilbert/                      # DistilBERT training scripts
├── gemma/                           # Gemma training scripts
├── roberta/                         # RoBERTa training scripts
├── delta_tfidf_hate_speech_model/   # Trained Delta TF-IDF model
├── distilbert_hate_speech_model/    # Trained DistilBERT model
└── roberta_hate_speech_model/       # Trained RoBERTa model
```

## Training Scripts

Each model has dedicated training scripts located in their respective folders:

- **Delta TF-IDF**: `src/delta_tfidf/`
  - `delta_tfidf (baseline).py` - Baseline model
  - `delta_tfidf (POS).py` - With POS features
  - `delta_tfidf (POSwithSMOTE).py` - POS + SMOTE
  - `delta_tfidf (SMOTE).py` - With SMOTE
  - `delta_tfidf (TDA).py` - Text Data Augmentation
  - `train_delta_tfidf.slurm` - SLURM job script

- **DistilBERT**: `src/distilbert/`
  - `train_distilbert (baseline).py` - Baseline model
  - `train_distilbert (POS).py` - With POS features
  - `train_distilbert (POSwithClass).py` - POS + class weights
  - `train_distilbert (POSwithSMOTE).py` - POS + SMOTE
  - `train_distilbert (SMOTE).py` - With SMOTE
  - `train_distilbert (TDA).py` - Text Data Augmentation
  - `train_distilbert.slurm` - SLURM job script

- **Gemma**: `src/gemma/`
  - Contains Gemma-7B training scripts with LoRA and POS features

- **RoBERTa**: `src/roberta/`
  - Contains RoBERTa training scripts with various enhancement techniques

## Demo Application

Run the Streamlit demo to compare model predictions:

```bash
streamlit run src/demo.py
```

The demo provides:
- Interactive model selection (toggle models on/off)
- Real-time hate speech predictions
- Confidence scores and probability visualizations
- Model agreement analysis

## Model Metrics

All models have been evaluated across four datasets using comprehensive metrics:

- **Datasets**: Hate Corpus, Gab & Reddit, Stormfront, and Merged Dataset
- **Models Evaluated**: Delta TF-IDF, DistilBERT, RoBERTa, Gemma-7B, DeBERTa, GPT-OSS 20B
- **Evaluation Metrics**:
  - **Accuracy**: Overall classification accuracy
  - **Macro F1**: Unweighted average F1 score across classes
  - **F0.5 Score**: Emphasizes precision over recall (β=0.5)
  - **F2 Score**: Emphasizes recall over precision (β=2)
  - **AUC**: Area under the ROC curve
  - **Weighted F1**: F1 score weighted by class support
  - **Weighted Precision**: Precision weighted by class support
  - **Weighted Recall**: Recall weighted by class support

*Format: Each metric shows the initial base score. The score after applying the technique is in parentheses.*

## Key Features

- **Multiple Architectures**: Delta TF-IDF, DistilBERT, RoBERTa, Gemma-7B, DeBERTa, GPT-OSS 20B
- **Enhancement Techniques**: 
  - SMOTE & Weighted Loss for handling class imbalance
  - POS (Part-of-Speech) tagging features
  - SMOTE & Weighted Loss & POS Integration
  - Text Data Augmentation (TDA)
- **Cross-Validation**: 10-fold CV for Delta TF-IDF
- **Multiple Datasets**: Hate Corpus, Gab & Reddit, Stormfront, and merged datasets

## Requirements

Install all dependencies using the requirements file:

```bash
pip install -r requirements.txt
```

Or install core packages individually:

```bash
pip install transformers torch pandas streamlit plotly scikit-learn imbalanced-learn joblib scipy peft
```

For training scripts with text augmentation, additional packages are needed:

```bash
pip install nltk spacy sentence-transformers
python -m spacy download en_core_web_sm
```

## Usage

### Training a Model

Models are trained using **Monash M3 High Performance Computing** cluster with GPU resources.

#### Local Training

Navigate to the appropriate folder and run the training script:

```bash
# Example: Train DistilBERT with SMOTE
python src/distilbert/train_distilbert\ \(SMOTE\).py

# Example: Train Delta TF-IDF with TDA
python src/delta_tfidf/delta_tfidf\ \(TDA\).py
```

#### HPC Training with SLURM

Each model folder contains a `.slurm` file for batch job submission on the M3 cluster. The SLURM script specifies resource requirements and execution commands. The models are trained using Monash M3 High Performance Computing cluster with the following specifications:

**HPC Resources Used**:
- **GPU**: NVIDIA A40 (1x)
- **Memory**: 128GB RAM
- **CPUs**: 8 cores
- **Time**: 3 hours per job
- **Storage**: Scratch area for caching and intermediate files

### Using the Demo

```bash
streamlit run src/demo.py
```

Select models using checkboxes, enter text, adjust the threshold, and analyze results across multiple models.

## Model Descriptions

- **Delta TF-IDF**: SVM-based classifier using delta TF-IDF features with 10-fold cross-validation
- **DistilBERT**: Distilled BERT model fine-tuned on hate speech with SMOTE augmentation
- **RoBERTa**: Robustly optimized BERT approach with domain-specific fine-tuning
- **Gemma-7B**: Google's 7B parameter model with LoRA and POS features
- **DeBERTa**: Decoding-enhanced BERT with disentangled attention
- **GPT-OSS 20B**: Open-source 20B parameter GPT model
