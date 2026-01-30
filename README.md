## Trained Model Weights
The trained DeBERTa-base model weights are provided via GitHub Releases.
Download `deberta_best.pt` from the latest release, place it under `models/`,
and run inference using:
python scripts/run_inference.py --checkpoint models/deberta_best.pt


Emotion Classification in Text using Fine-Tuned Transformer Models

This project implements an end-to-end emotion classification system for short text (tweets) using fine-tuned Transformer-based language models.
The goal is to accurately classify text into one of six emotional categories while comparing multiple pretrained architectures and applying model compression techniques to improve efficiency.

The project was developed as part of an advanced Natural Language Processing (NLP) course and follows reproducible research and clean ML engineering practices.

Task Description
Given a short text input (e.g., a tweet), the system predicts one of the following emotions:
| Label | Emotion  |
| ----: | -------- |
|     0 | Sadness  |
|     1 | Joy      |
|     2 | Love     |
|     3 | Anger    |
|     4 | Fear     |
|     5 | Surprise |

The dataset is imbalanced, so special care is taken in evaluation and training strategies.

Models Used

The project fine-tunes and evaluates multiple Transformer architectures:

BERT (bert-base-uncased)

ELECTRA

DeBERTa

Each model is:

Fine-tuned on the same preprocessing pipeline

Evaluated using consistent metrics

Compared in terms of performance, size, and inference time



Evaluation Metrics

Due to class imbalance, the evaluation goes beyond accuracy:
Accuracy
Precision
Recall
F1-Score (macro & weighted)
Confusion Matrix
Inference Time
Model Size (parameters & disk footprint)

Model Compression

After selecting the best-performing model, model compression techniques are applied and analyzed:
Knowledge Distillation (KD)
Quantization
Pruning (optional / experimental)
Compressed models are compared against the original model in terms of:
Accuracy drop
Speedup
Memory reduction


Preprocessing Pipeline

A dedicated preprocessing pipeline is used for all models to ensure fairness:

Text normalization
Tokenization using pretrained tokenizers
Padding & truncation based on dataset statistics
Train / Validation split (test set kept unseen)
Tokenizer-aware preprocessing is applied (no manual token manipulation).

