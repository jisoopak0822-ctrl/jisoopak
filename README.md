
# 🐦 Twitter Sentiment Analysis using CNN & NLP
> **A deep learning project exploring the trade-offs between accuracy and computational efficiency in sentiment classification.**

## 📌 Overview
This project investigates sentiment classification on the **Sentiment140 dataset** (1.6M tweets). The core objective was to build an optimized **Convolutional Neural Network (CNN)** for text and compare its performance against a transformer-based model (**DistilBERT**), focusing on convergence behavior and computational costs.

## 📊 Key Results
- **Best Model Accuracy**: **80.08%** (Experiment 6)
- **Primary Insight**: Small 3-gram features (kernel size 3) were more effective for short-form tweets than larger kernels or deeper architectures.
- **Optimization**: Optimized the model to achieve high performance with only ~1.04M parameters, ensuring efficiency.

## 🛠 Tech Stack
- **Language**: Python
- **Deep Learning**: PyTorch
- **Environment**: Google Colab
- **NLP Techniques**: Tokenization, Padding, Word Embeddings, Pre-trained Transformers (DistilBERT)

## 🧪 Experiments & Insights
I conducted 8 controlled experiments to find the optimal balance between accuracy and cost:

| Config | Changes | Test Accuracy | Note |
|:---:|:---|:---:|:---|
| Exp 1 | Baseline CNN | 73.59% | Starting point |
| **Exp 6** | **Kernel Size (5 → 3)** | **80.08%** | **Best Performance** |
| Exp 7 | Embed Dim (100 → 500) | 68.44% | Overfitting observed |
| Exp 8 | Seq Length (50 → 250) | 73.65% | High cost, low gain |

### 🔍 Major Takeaways
1. **Kernel Size Matters**: Using a smaller kernel (size 3) captured short-range linguistic patterns in tweets better than size 5.
2. **Efficiency vs. Performance**: While DistilBERT showed strong potential, the custom CNN (Exp 6) provided a highly competitive accuracy with significantly lower inference latency.
3. **Embedding Overfitting**: Increasing embedding dimensions to 500 actually decreased accuracy, suggesting that a simpler 100-dim vector is sufficient for the 10,000-word vocabulary used.

## 📂 Project Structure
- `9891_final_project.ipynb`: Full training pipeline and experimental logs.
- `Final Project Report.pdf`: Detailed analysis of hyperparameters and convergence.

## 🚀 How to Run
1. Clone the repository.
2. Download the Sentiment140 dataset from Kaggle.
3. Open the notebook in Google Colab and run all cells.
