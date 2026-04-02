# Enzyme Graph Classification with GCN

**Course**: Deep Learning — Homework 4  
**Instructor**: Professor Jun Bai  
**Author**: Iman Jamshidi  

## Overview

This project applies Graph Convolutional Networks (GCN) to classify enzymes into 6 categories using the ENZYMES benchmark dataset. Three GCN architectures with increasing depth (1, 2, and 3 layers) are compared using Accuracy, F1, and AUC metrics.

## Dataset

- **ENZYMES**: 600 molecular graphs representing enzyme structures
- **Task**: 6-class graph classification
- **Node features**: 19 (18 continuous attributes + 1 categorical node label)
- **Split**: 480 train (80%) / 60 validation (10%) / 60 test (10%)
- **Labels**: 6 enzyme classes (0–5, zero-indexed)

## Models & Results

| Model | Accuracy | F1 Score | AUC | Parameters |
|---|---|---|---|---|
| GCN-1 layer | 25.0% | 28.5% | 0.65 | 3,590 |
| GCN-2 layers | 38.3% | 38.1% | 0.78 | 20,358 |
| **GCN-3 layers** | **46.7%** ✅ | **48.3%** | **0.80** | 37,126 |

Best model: **GCN-3** — Accuracy: 46.7%, F1: 0.483, AUC: 0.801

## Key Findings

- Deeper GCN models consistently outperformed shallower ones — each extra layer allowed more message passing across molecular graph structures
- GCN-1 severely underfitted with only 25% accuracy — one layer of message passing was not enough
- GCN-3 reached 80% AUC despite modest accuracy, showing it learns good probability estimates
- Class 2 and Class 3 were easiest to classify (AUC ≈ 0.89); Class 6 was hardest (AUC = 0.66)
- Global add pooling outperformed mean pooling for graph-level aggregation
- Cross-entropy loss converged smoothly from ~10–12 down to ~1.5–2.0

## Files

| File | Description |
|---|---|
| `models.py` | GCN architectures (GCN1, GCN2, GCN3) |
| `train.py` | Training loop with early stopping |
| `dataset.py` | ENZYMES dataset loading and preprocessing |
| `metrics.py` | Evaluation metrics and visualization |
| `test_model.py` | Test function for saved models |
| `utils.py` | Helper functions |
| `HW4_Jamshidi.ipynb` | Main training notebook (Google Colab) |
| `Step2_Test_All_Models.ipynb` | Testing notebook for all three models |
| `Iman_jamshidi_HW4.pdf` | Full written report |

## How to Run (Google Colab — Recommended)

1. Upload `HW4_Jamshidi.ipynb` to [Google Colab](https://colab.research.google.com)
2. Upload all `.py` files when prompted
3. Upload the 5 ENZYMES dataset `.txt` files
4. Run all cells sequentially

## How to Test Saved Models
```python
from test_model import test_model
from models import GraphClassifierGCN3

results = test_model(
    model_path='HW4_Results_Jamshidi/gcn3_best.pth',
    test_loader=test_loader,
    model_class=GraphClassifierGCN3,
    input_dim=19,
    hidden_dim=128,
    num_classes=6,
    device=device,
    dropout_rate=0.5
)

print(f"Test Accuracy: {results['accuracy']:.4f}")
```

Or run `Step2_Test_All_Models.ipynb` to evaluate all three models at once.

## Requirements
```bash
pip install torch==2.5.0 torchvision
pip install torch-geometric torch-scatter torch-sparse
pip install scikit-learn matplotlib pandas
```
