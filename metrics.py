import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize


# Compute evaluation metrics
def compute_metrics(y_true, logits):
    num_classes = logits.shape[1]
    y_pred = logits.argmax(dim=1).cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    
    acc = accuracy_score(y_true_np, y_pred)
    f1 = f1_score(y_true_np, y_pred, average='weighted')
    
    # Compute AUC for multi-class
    probs = F.softmax(logits, dim=1).cpu().numpy()
    try:
        auc_score = roc_auc_score(y_true_np, probs, multi_class='ovr', average='weighted')
    except:
        auc_score = 0.0
    
    return {
        'accuracy': acc,
        'f1_score': f1,
        'auc': auc_score
    }


# Compute AUC for multi-class classification
def compute_auc_multiclass(y_true, logits):
    y_true_np = y_true.cpu().numpy()
    probs = F.softmax(logits, dim=1).cpu().numpy()
    
    try:
        auc_score = roc_auc_score(y_true_np, probs, multi_class='ovr', average='weighted')
    except:
        auc_score = 0.0
    
    return auc_score


# Plot ROC curve for multi-class
def plot_roc_curve(y_true, logits, save_path=None):
    num_classes = logits.shape[1]
    y_true_np = y_true.cpu().numpy()
    probs = F.softmax(logits, dim=1).cpu().numpy()
    
    # Binarize labels
    y_true_bin = label_binarize(y_true_np, classes=range(num_classes))
    
    plt.figure(figsize=(10, 8))
    
    # Compute ROC curve for each class
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {i+1} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Multi-class Classification')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


# Plot training curves
def plot_training_curves(history, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


# Plot confusion matrix
def plot_confusion_matrix(y_true, logits, class_names=None, save_path=None):
    y_pred = logits.argmax(dim=1).cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    
    cm = confusion_matrix(y_true_np, y_pred)
    
    if class_names is None:
        class_names = [f'Class {i+1}' for i in range(cm.shape[0])]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


# Print classification report
def print_classification_report(y_true, logits, class_names=None):
    y_pred = logits.argmax(dim=1).cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    
    if class_names is None:
        num_classes = logits.shape[1]
        class_names = [f'Class {i+1}' for i in range(num_classes)]

    
    report = classification_report(y_true_np, y_pred, target_names=class_names)
    print(report)


# Create empty history dictionary
def create_history():
    return {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
