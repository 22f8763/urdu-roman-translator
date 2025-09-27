# Neural Machine Translation – Urdu ➜ Roman Urdu

A sequence-to-sequence **Neural Machine Translation (NMT)** system that transliterates
poetic Urdu text into **Roman Urdu** using a BiLSTM encoder and LSTM decoder.

---

## 📖 Objective
Build and train a **bidirectional LSTM (BiLSTM) encoder–decoder** to translate
Urdu text into its Roman Urdu transliteration.  
The project explores NMT for **low-resource poetic text** using the
[`urdu_ghazals_rekhta`](https://github.com/amir9ume/urdu_ghazals_rekhta) dataset.

---

## 🗂️ Dataset
- **Source**: [urdu_ghazals_rekhta](https://github.com/amir9ume/urdu_ghazals_rekhta)  
- Includes Urdu script, English transliteration, and Hindi script.
- Extracted pairs: **Urdu (source)** → **Roman Urdu (target)**.

---

## ⚙️ Preprocessing
- Urdu text normalization & punctuation cleaning.
- Roman Urdu conversion rules if direct transliteration is missing.
- Tokenization with support for subword methods (e.g., BPE / WordPiece).

---

## 🏗️ Model Architecture
| Component | Details |
|-----------|--------|
| Encoder | **BiLSTM**, 2 layers |
| Decoder | **LSTM**, 4 layers |
| Framework | [PyTorch](https://pytorch.org/) |
| Loss | Cross-Entropy |
| Optimizer | Adam |

---

## 🔬 Experiments
At least three controlled experiments were conducted by varying:
- **Embedding Dimension**: 128 / 256 / 512
- **Hidden Size**: 256 / 512
- **BiLSTM Encoder Layers**: 1 / 2 / 3 / 4
- **Decoder Layers**: 2 / 3 / 4
- **Dropout**: 0.1 / 0.3 / 0.5
- **Learning Rate**: 1e-3 / 5e-4 / 1e-4
- **Batch Size**: 32 / 64 / 128

---

## 🏋️ Training
- Train / Validation / Test split: **50% / 25% / 25%**
- Implemented in PyTorch with GPU support (e.g., Kaggle, Colab).

---

## 📊 Evaluation
- **BLEU** and **Perplexity** scores as primary metrics.
- **Character Error Rate (CER)** / Edit distance for fine-grained analysis.
- Qualitative examples comparing model outputs with ground truth.

---

## 🚀 Deployment
- Final trained model wrapped in a **Streamlit** web app for live inference.

---

## 🔮 Future Work / Bonus
- Data augmentation (back-transliteration, noise injection).
- Experiment with **xLSTM** in place of BiLSTM + LSTM.

---

## 💻 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Streamlit
- Git LFS (for large model files)

### Setup
```bash
# Clone the repository
git clone https://github.com/<your-username>/urdu-roman-translator.git
cd urdu-roman-translator

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
