import torch
import torch.nn as nn


# Test saved model on test dataset
def test_model(model_path, test_loader, model_class, input_dim, hidden_dim, num_classes, device, dropout_rate=0.5):
    # Initialize model
    model = model_class(input_dim, hidden_dim, num_classes, dropout_rate).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    all_logits = []
    all_labels = []
    all_preds = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.num_graphs
            
            all_logits.append(out)
            all_labels.append(data.y)
            all_preds.append(pred)
    
    accuracy = correct / total
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    
    return {
        'accuracy': accuracy,
        'logits': all_logits,
        'labels': all_labels,
        'predictions': all_preds
    }
