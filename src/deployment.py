# Model Export and Deployment Package
print("\nCreating Deployment Package")

# Create deployment metadata
deployment_info = {
    'model_type': 'HGNN_Intrusion_Detection',
    'version': '1.0',
    'created_date': datetime.now().isoformat(),
    'performance': {
        'test_accuracy': float(test_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc)
    },
    'model_config': config,
    'dataset': 'NSL-KDD',
    'features': processor.columns[:-2]  # Exclude label and difficulty
}

# Save processor
with open('models/processor.pkl', 'wb') as f:
    pickle.dump(processor, f)
print("Processor saved")

# Save complete model
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': deployment_info['model_config'],
    'performance': deployment_info['performance']
}, 'models/hgnn_model_complete.pth')
print("Model saved")

# Save deployment info
with open('models/deployment_info.json', 'w') as f:
    json.dump(deployment_info, f, indent=2)
print("Deployment info saved")

# Create model documentation
readme_content = f"""# HGNN Intrusion Detection System

## Performance Results
- **Test Accuracy**: {test_acc:.4f} ({test_acc*100:.2f}%)
- **Precision**: {precision:.4f}
- **Recall**: {recall:.4f}
- **F1-Score**: {f1:.4f}
- **ROC AUC**: {roc_auc:.4f}
- **PR AUC**: {pr_auc:.4f}

## Model Architecture
- **Input Dimension**: {config['input_dim']} features
- **Hidden Layers**: {config['hidden_dims']} neurons
- **Output Dimension**: {config['output_dim']} classes
- **Total Parameters**: {sum(p.numel() for p in model.parameters()):,}
- **Dropout Rate**: {config['dropout']}

## Usage Instructions

### Loading the Model
```python
import torch
import pickle

# Load processor
with open('processor.pkl', 'rb') as f:
    processor = pickle.load(f)

# Load model
checkpoint = torch.load('hgnn_model_complete.pth')
config = checkpoint['model_config']

model = HGNN(
    input_dim=config['input_dim'],
    hidden_dims=config['hidden_dims'],
    output_dim=config['output_dim']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Initialize detector
detector = RealTimeDetector(model, processor, device)
```

### Making Predictions
```python
# Process network traffic data
predictions, probabilities, scores = detector.detect_batch(network_data)

# Generate alerts
for i, (pred, prob, score) in enumerate(zip(predictions, probabilities, scores)):
    alert = detector.generate_alert(pred, prob, score, f"sample_{i}")
    if alert['action_required']:
        print(f"ALERT: {alert}")
```

## File Structure
- `complete_hgnn_model.pth`: Complete trained model
- `processor.pkl`: Data preprocessor
- `deployment_info.json`: Model metadata
- `README.md`: This documentation

## Training Configuration
- **Epochs**: {len(train_losses)}
- **Learning Rate**: {config['learning_rate']}
- **Weight Decay**: {config['weight_decay']}
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss

## Dataset Information
- **Source**: NSL-KDD
- **Training Samples**: {len(X_train)}
- **Test Samples**: {len(X_test)}
- **Features**: {len(processor.columns)-2} network traffic features
- **Classes**: Binary (Normal/Attack)

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open('models/README.md', 'w') as f:
    f.write(readme_content)
print("Documentation created")

print("\nDeployment package created!")
print("Files in models/ directory:")
for file in os.listdir('models/'):
    file_path = os.path.join('models', file)
    size_kb = os.path.getsize(file_path) / 1024
    print(f" {file} ({size_kb:.1f} KB)")

print("Deployment package ready!")
