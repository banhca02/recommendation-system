# Multi-Modal Two-Tower Recommendation System with SASRec

This project implements a personalized recommendation system based on a **Multi-Modal Two-Tower** architecture. The system combines a Transformer network (**SASRec**) to model sequential user behavior history and a **Fusion mechanism** to integrate diverse item features (Text, Images, Tabular data). The user-item matching process is optimized using **Contrastive Learning** via **InfoNCE Loss**.

![Architecture Diagram](images/two_tower_model.png)

---

## Key Features

| Feature | Description |
|---|---|
| **Sequential Modeling** | Utilizes Self-Attention Sequential Recommendation (SASRec) to capture dynamic user interests over time. |
| **Multi-Modal Fusion** | Effectively handles and fuses heterogeneous item data (Metadata, Reviews, Visuals) without relying on a monolithic architecture. |
| **Highly Scalable** | Two-Tower separation allows for offline item embedding precomputation, enabling lightning-fast online inference via approximate nearest neighbor search (e.g., FAISS). |
| **Automated Pipeline** | The entire workflow (Item feature extraction, User Tower training, and Full-Ranking Evaluation) is streamlined into a single automated execution script. |

---

## Installation & Usage

### 1. Prerequisites

Ensure you have **Python >= 3.8** installed. Clone this repository and install the required dependencies:

```bash
git https://github.com/banhca02/recommendation-system.git
cd cd recommendation-system
pip install -r requirements.txt
```

### 2. Execution

The execution pipeline has been fully optimized. You no longer need to run separate phases manually. Trigger the entire process — from data loading to training and evaluation — using a single command:

```bash
cd recommendation-system
python main.py \
  --item_data dataset/item_metadata.json \
  --user_data dataset/user_history.json \
  --output_dir ./outputs
```

### 3. Arguments Breakdown

| Argument | Description |
|---|---|
| `--item_data` | Path to the raw item metadata file (contains descriptions, categories, etc.). |
| `--user_data` | Path to the sequential user interaction history. |
| `--output_dir` | Directory where the pipeline saves precomputed embeddings (`.pt`), model checkpoints (`.pth`), and evaluation logs. |

---

## Evaluation Metrics

During the evaluation phase, the system performs a **Full-Ranking** sweep across the entire item catalog to ensure realistic metrics. Performance is measured using standard industry metrics:

- **Hit Rate @K (HR@K)** — Measures whether the ground-truth item appears in the top K recommendations.
- **NDCG @K** — Normalized Discounted Cumulative Gain; penalizes the score if the correct item is ranked lower in the top K list.
- **MRR** — Mean Reciprocal Rank; evaluates the position of the first relevant recommendation.

