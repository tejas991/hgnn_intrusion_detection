# HGNN Intrusion Detection System - Tejas Bharadwaj

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


