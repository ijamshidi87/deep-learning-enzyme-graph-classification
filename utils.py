import os
import random
import numpy as np
import torch
import json
from datetime import datetime


def set_random_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    
    # Ensure deterministic behavior in CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_model_parameters(model, only_trainable=True):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        

def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': float(accuracy),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss'],
        'accuracy': checkpoint['accuracy'],
        'timestamp': checkpoint.get('timestamp', 'Unknown')
    }


def save_metrics(metrics_dict, filepath):
    with open(filepath, 'w') as f:
        json.dump(metrics_dict, f, indent=4)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_model_summary(model, model_name="Model"):
    total_params = count_model_parameters(model, only_trainable=False)
    trainable_params = count_model_parameters(model, only_trainable=True)
    
    print(f"\n{model_name} Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
