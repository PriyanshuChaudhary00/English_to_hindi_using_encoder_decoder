# English to Hindi Translator (Seq2Seq)

A sequence-to-sequence (Seq2Seq) neural network built from scratch in PyTorch to translate English sentences to Hindi. This project demonstrates core NLP concepts, dataset preprocessing, custom vocabulary mapping, and PyTorch training workflows.

## 🚀 Features
- **Custom Encoder-Decoder Architecture:** Implemented with PyTorch GRU cells.
- **Teacher Forcing:** Applied during training to stabilize and speed up learning.
- **Dynamic Padding:** Uses PyTorch's `pad_sequence` within a custom `collate_fn` for efficient batching.
- **Vocabulary Mapping:** Automatically handles tokenization, builds vocabulary, and maps tokens to indices with `<PAD>`, `<UNK>`, `<start>`, and `<end>` special tokens.
- **Device Agnostic:** Supports training on Apple Silicon GPUs (`mps`) and standard `cpu`.
- **Training Metrics:** Automatically saves the training loss plot and model checkpoints.

---

## 📁 Repository Structure
- `model.py` — Defines the `Encoder`, `Decoder`, and `Seq2Seq` classes.
- `dataset.py` — Preprocesses text data, builds vocabularies, and loads the PyTorch Dataset.
- `train.py` — Training loop with optimization, loss calculation, and checkpoint saving.
- `predict.py` — Interactive command-line script to test translation on new inputs.
- `Dataset_English_Hindi.csv` — Parallel corpus containing English-Hindi sentence pairs.

---

## 🛠️ Setup & Usage

### 1. Installation
Install the required dependencies:
```bash
pip install torch pandas matplotlib
```

### 2. Training
To preprocess the corpus and train the Seq2Seq model, run:
```bash
python train.py
```
This will:
1. Create a `checkpoint/` directory.
2. Save the trained model (`model_epoch10.pth`) and vocabularies (`vocab.json`).
3. Save the training loss plot as `loss_plot.png`.

### 3. Inference / Prediction
Run the interactive translation script:
```bash
python predict.py
```

---

## 🧠 What I Learned & Engineering Challenges

### 1. Punctuation & Tokenization Bottlenecks
- **Problem:** Simple split-based tokenizers treat words with trailing punctuation (like `"ok."`) as entirely new tokens. This leads to high out-of-vocabulary (`<UNK>`) rates.
- **Lesson:** Proper text normalization (lowercasing, cleaning punctuation, and tokenizing with regex/advanced NLP libraries) is critical to performance.

### 2. Seq2Seq Information Bottleneck
- **Problem:** Standard Encoder-Decoder models compress the entire input sequence into a single context vector. For longer sentences, this causes the decoder to "forget" details, often predicting the `<end>` token too early.
- **Lesson:** Introducing Attention Mechanisms is key to scaling translation models.

### 3. The Power of Teacher Forcing
- **Problem:** Without teacher forcing, early training steps are extremely slow because the decoder feeds its own incorrect predictions back into itself.
- **Lesson:** Tuning the `teacher_forcing_ratio` is essential to balance fast training convergence with robust inference.

