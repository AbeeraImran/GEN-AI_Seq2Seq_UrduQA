


# GEN-AI Seq2Seq UrduQA

A sequence-to-sequence (Seq2Seq) neural pipeline designed to generate context-relevant Urdu questions from context paragraphs and answer spans using the `uqa/UQA` dataset.



---

## Project Overview

* **Task:** Given an Urdu context sentence containing an answer span (wrapped in `...` tags), generate the corresponding question.
* **Domain:** Natural Language Processing / Generative AI
* **Framework:** PyTorch & SentencePiece

---

## Architecture Pipeline

* **Tokenizer:** SentencePiece Unigram model with an 8,000-token vocabulary and user-defined symbols (`, `).
* **Encoder:** 2-layer Bidirectional LSTM (Embedding Dim: 256, Hidden Dim: 512, Dropout: 0.3) with packed sequence handling.
* **Decoder (Base):** 2-layer Unidirectional LSTM with step-wise decoding and Teacher Forcing support (50% ratio).
* **Optimization:** Adam Optimizer (lr=0.001), Cross-Entropy Loss ignoring padding (ignore_index=0).

---

## Repository Structure

```
├── data/                  # Cleaned TSV splits (train.tsv, valid.tsv)
├── tokenizer/             # ur_sp.model and ur_sp.vocab
├── notebooks/             # End-to-end training and inference notebooks
├── checkpoints/           # Saved model state dicts (best_seq2seq_model.pt)
└── README.md

```

---

## Getting Started

### 1. Requirements

Install the core dependencies:

```
pip install torch datasets sentencepiece sacrebleu rouge-score

```

### 2. Pipeline Execution

1. **Data Prep & Tokenizer:** Run the preprocessing cells to filter question-answer pairs and train the SentencePiece model.
2. **Model Training:** Initialize the Encoder and DecoderBase modules and run train_one_epoch on GPU.
3. **Checkpointing:** Model weights are automatically saved to best_seq2seq_model.pt whenever validation loss improves.

---

## Roadmap & Next Steps

* [x] Data extraction, preprocessing, and SentencePiece tokenization
* [x] PyTorch custom QGDataset, collate function, and batched DataLoader
* [x] Bi-LSTM Encoder, Base Decoder, and training loop validation
* [ ] Implement Bahdanau / Luong Attention in the Decoder
* [ ] Add Beam Search decoding (beam size 3–5)
* [ ] Quantitative evaluation (BLEU-4, ROUGE-L, Perplexity) on validation and test sets
* [ ] Streamlit / Gradio web demo interface

