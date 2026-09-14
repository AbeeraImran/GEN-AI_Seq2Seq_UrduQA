# Teaching AI to Ask Questions in Urdu: A Seq2Seq Approach

## Overview
This repository contains the codebase for an Answer-Aware Question Generation (QG) system built specifically for Urdu, a morphologically rich and low-resource language. Instead of fine-tuning a pretrained transformer, this model is built and trained entirely from the ground up to explore the foundational challenges of subword tokenization and sequence-to-sequence modeling in South Asian NLP.

## Architecture & Tech Stack
* **Frameworks:** PyTorch, NumPy, Pandas, Matplotlib
* **Tokenizer:** SentencePiece (Unigram language model, 8,000 vocabulary size)
* **Encoder:** 2-layer Bidirectional LSTM
* **Decoder:** 2-layer Unidirectional LSTM with Bahdanau (Additive) Attention
* **Inference Strategy:** Greedy Decoding with UNK-token Logit Suppression

## Dataset & Preprocessing
The model is trained using the [uqa/UQA](https://huggingface.co/datasets/uqa/UQA) dataset from Hugging Face. 
To achieve "answer-aware" generation, context paragraphs are parsed and the specific answer spans are wrapped in custom ` ... ` tags. These tags are registered as user-defined symbols in SentencePiece to prevent them from being fragmented during subword tokenization.

## Evaluation & Metrics
Because standard automated metrics (like BLEU) heavily penalize morphologically rich languages for valid paraphrasing, our evaluation relies on a combination of automated scoring and independent human review. 

**Automated Metrics (Validation Subset):**
* **BLEU-4 Score:** 3.04
* **ROUGE-L (F-Measure):** 0.0063
* **UNK Token Rate:** 0.00%

*Note on Inference:* Early greedy decoding passes exhibited mode collapse, where the decoder would loop the Unknown (``) token. This was corrected by implementing **Logit Suppression** during the forward pass (setting the `UNK_ID` probability to `-inf` prior to the `argmax` selection). This forced the model to utilize its learned vocabulary, successfully dropping the UNK rate to 0.00%.

**Qualitative Evaluation:**
Automated metrics are paired with a 50-sample human evaluation. Generated questions are scored on a 1–5 scale across three dimensions:
1. **Fluency:** Is the Urdu grammatically correct?
2. **Relevance:** Does the question relate to the context paragraph?
3. **Answerability:** Can the question be answered by the target `` span?
To ensure evaluation integrity, we calculate Cohen’s Kappa across our grading team to measure inter-rater reliability.

## Known Limitations
* **Matplotlib RTL Rendering:** Matplotlib does not natively support Right-to-Left (RTL) text layout or complex glyph rendering for Nastaliq scripts. As a result, the Urdu axis labels on our Attention Alignment Heatmaps render as disconnected or reversed characters. This is a visualization library limitation, not a tokenization error.
* **Vocabulary Ceiling:** The restricted 8,000-token vocabulary limits the model's ability to perfectly capture Urdu's morphological complexity.

## Future Improvements
While the current architecture successfully proves the viability of answer-aware QG in Urdu, future scaling could benefit from:
* Implementing Beam Search decoding to improve sentence fluency and structure over greedy decoding.
* Expanding the SentencePiece vocabulary size and experimenting with BPE vs. Unigram segmentation to better capture Urdu's morphological complexity.
