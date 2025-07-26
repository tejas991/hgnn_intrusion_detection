# HGNN Intrusion Detection System

## Performance Results
- **Test Accuracy**: 0.7829 (78.29%)
- **Precision**: 0.9675
- **Recall**: 0.6408
- **F1-Score**: 0.7710
- **ROC AUC**: 0.9019
- **PR AUC**: 0.9354
- **Throughput**: 92.7 samples/second

## Model Architecture
- **Input Dimension**: 41 features
- **Hidden Layers**: [32, 16] neurons
- **Output Dimension**: 2 classes (Normal/Attack)
- **Total Parameters**: 2,002
- **Dropout Rate**: 0.3
- **Memory Optimized**: Yes (Google Colab compatible)

## Package Contents
- `hgnn_model_complete.pth`: Complete trained model
- `processor.pkl`: Data preprocessor with encoders and scaler
- `deployment_info.json`: Model metadata and performance metrics
- `README.md`: This documentation
- `Project_Summary.md`: Executive summary
- `metrics.json`: Detailed performance metrics

## Quick Start

### Loading the Model
```python
import torch
import pickle
import numpy as np

# Load processor
with open('processor.pkl', 'rb') as f:
    processor = pickle.load(f)

# Load model
checkpoint = torch.load('complete_hgnn_model.pth', map_location='cpu')
config = checkpoint['model_config']

# Initialize model (you'll need the HGNN class definition)
model = HGNN(
    input_dim=config['input_dim'],
    hidden_dims=config['hidden_dims'],
    output_dim=config['output_dim'],
    dropout=config['dropout']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Making Predictions
```python
# Preprocess your network traffic data
X_processed = processor.scaler.transform(your_network_data)

# Create hypergraph (you'll need the hypergraph construction function)
H = construct_memory_efficient_hypergraph(X_processed)

# Convert to tensors
X_tensor = torch.FloatTensor(X_processed)
H_tensor = H

# Predict
with torch.no_grad():
    outputs = model(X_tensor, H_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    predictions = torch.argmax(outputs, dim=1)

    # predictions: 0=Normal, 1=Attack
    # probabilities: [normal_prob, attack_prob]
```

### Using the Real-time Detector
```python
# Initialize detector (requires RealTimeDetector class)
detector = RealTimeDetector(model, processor, device)

# Process batch of network data
predictions, probabilities, scores = detector.detect_batch(network_data)

# Generate alerts
for i, (pred, prob, score) in enumerate(zip(predictions, probabilities, scores)):
    alert = detector.generate_alert(pred, prob, score, f"sample_{i}")
    if alert['action_required']:
        print(f"ALERT: {alert}")
```

## Performance Metrics Details
```json
{
  "test_accuracy": 0.7829,
  "precision": 0.9675,
  "recall": 0.6408,
  "f1_score": 0.7710,
  "roc_auc": 0.9019,
  "pr_auc": 0.9354,
  "throughput": 92.7
}
```

## System Requirements
- Python 3.7+
- PyTorch 1.8+
- NumPy, Pandas, Scikit-learn
- Memory: Optimized for 12GB RAM environments

## File Structure
```
Project/
├── complete_hgnn_model.pth    # Trained model
├── processor.pkl                      # Data preprocessor
├── deployment_info.json              # Model metadata
├── README.md                         # This documentation
├── Project_Summary.md               # Executive summary
└── metrics.json                     # Performance metrics
```

## Training Configuration
- **Epochs**: 40
- **Learning Rate**: 0.01
- **Weight Decay**: 0.0001
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss

## Dataset Information
- **Source**: NSL-KDD Network Intrusion Detection Dataset
- **Training Samples**: 20000
- **Test Samples**: 15000
- **Features**: 41 network traffic features
- **Classes**: Binary classification (Normal/Attack)

## Deployment Notes
This model is production-ready and can be deployed in:
- Real-time network monitoring systems
- Security information and event management (SIEM) platforms
- Intrusion detection systems (IDS)
- Network security appliances

## License
This implementation is for research and educational purposes.

---
Generated on: 2025-07-26 22:51:20
Package created by HGNN Intrusion Detection System v1.0
