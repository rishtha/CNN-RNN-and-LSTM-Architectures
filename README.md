# CNN, RNN and LSTM Architectures

Deep learning assignment implementing and comparing CNN, RNN, and LSTM architectures 
from scratch using PyTorch.  
**Course:** CSE 676-B — Deep Learning, University at Buffalo (Spring 2025)

## Overview
This project covers five core deep learning implementations:
- **Part I:** VGG-16 (Version C) and ResNet-18 for image classification
- **Part II:** Vanishing gradient analysis in deep CNNs vs. ResNet
- **Part III:** Time-series forecasting using RNN/LSTM
- **Part IV:** Sentiment analysis using LSTM
- **Part V:** Theoretical derivations — CNN parameter calculations and LSTM backpropagation

## Tech Stack
- **Language:** Python
- **Framework:** PyTorch
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn,
  torchinfo, torchtext, nltk, wordcloud, TensorBoard
- **Tools:** Jupyter Notebook

## Project Structure
├── a1_part_1.ipynb      # VGG-16 and ResNet-18 image classification
├── a1_part_2.ipynb      # Vanishing gradient experiments
├── a1_part_3.ipynb      # Time-series forecasting with RNN/LSTM
├── a1_part_4.ipynb      # Sentiment analysis with LSTM
├── a1_part_5.pdf        # Theoretical derivations (CNN + LSTM)
└── a1_weights.txt       # Link to saved model weights

## Key Results
| Part | Task | Model | Target |
|------|------|-------|--------|
| I | Image Classification (Dogs/Cars/Food) | VGG-16 / ResNet-18 | >80% accuracy |
| II | Vanishing Gradient Analysis | VGG-Deep | Demonstrated |
| III | Time-Series Forecasting | RNN / LSTM | >75% accuracy |
| IV | Sentiment Analysis | Baseline + Improved LSTM | >75% accuracy |

## Key Concepts Covered
- VGG-16 architecture with dropout, LR scheduling, weight initialization
- ResNet-18 with residual blocks and skip connections
- Vanishing gradient problem — gradient norm tracking with PyTorch hooks
- Stacked and bidirectional LSTMs for sequence modeling
- Pre-trained word embeddings (GloVe/Word2Vec) for NLP tasks
- Regularization: dropout, early stopping, L1/L2, image augmentation
- Evaluation: confusion matrix, precision, recall, F1, ROC curve, TensorBoard

## How to Run
```bash
git clone https://github.com/rishtha/CNN-RNN-and-LSTM-Architectures.git
cd CNN-RNN-and-LSTM-Architectures
pip install -r requirements.txt
jupyter notebook
```

## Author
Rishitha Saravanan Priya  
[LinkedIn](https://linkedin.com/in/rishithasp) | [Portfolio](https://rishitha.dev)
