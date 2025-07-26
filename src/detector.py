# Detection System
print("\nSetting up Real-time Detection System")

class RealTimeDetector:
    """Real-time intrusion detection system"""

    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self.model.eval()

    def detect_batch(self, network_data):
        """Detect intrusions in a batch of network traffic"""
        try:
            # Preprocess data
            X_processed = self.processor.scaler.transform(network_data)

            # Construct hypergraph
            H = construct_hypergraph(X_processed, k_neighbors=min(5, len(X_processed)-1))

            # Convert to tensors
            X_tensor = torch.FloatTensor(X_processed).to(self.device)
            H_tensor = H.to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(X_tensor, H_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                threat_scores = probabilities[:, 1]  # Attack probability

            return (
                predictions.cpu().numpy(),
                probabilities.cpu().numpy(),
                threat_scores.cpu().numpy()
            )
        except Exception as e:
            print(f"Detection error: {e}")
            return None, None, None

    def get_threat_level(self, threat_score):
        """Convert threat score to readable level"""
        if threat_score < 0.3:
            return "LOW", "Normal traffic pattern"
        elif threat_score < 0.6:
            return "MEDIUM", "Suspicious activity detected"
        elif threat_score < 0.8:
            return "HIGH", "Likely attack pattern"
        else:
            return "CRITICAL", "High confidence attack detected"

    def generate_alert(self, prediction, probability, threat_score, sample_id=None):
        """Generate security alert"""
        threat_level_tuple = self.get_threat_level(threat_score)
        threat_level_str = threat_level_tuple[0] # Get the level string (e.g., "LOW")
        description = threat_level_tuple[1] # Get the description string

        alert = {
            'timestamp': datetime.now().isoformat(),
            'sample_id': sample_id or f"sample_{np.random.randint(1000, 9999)}",
            'prediction': 'ATTACK' if prediction == 1 else 'NORMAL',
            'confidence': float(probability[prediction]),
            'threat_score': float(threat_score),
            'threat_level': threat_level_str, # Use the level string directly
            'description': description,
            'action_required': prediction == 1 and threat_score > 0.7
        }

        return alert

# Initialize detector
detector = RealTimeDetector(model, processor, device)
print("Real-time detector initialized!")

# Test real-time detection
print("\nTesting Real-time Detection")
test_batch = X_test[:20]
true_labels = y_test[:20]

# Measure inference time
start_time = time.time()
predictions, probabilities, threat_scores = detector.detect_batch(test_batch)
inference_time = time.time() - start_time

if predictions is not None:
    print(f"Detection Results for {len(test_batch)} samples:")
    print("-" * 80)
    print(f"{'Sample':<8} {'Prediction':<12} {'Confidence':<12} {'Threat':<10} {'Actual':<8} {'✓/✗'}")
    print("-" * 80)

    for i in range(len(predictions)):
        pred_label = "ATTACK" if predictions[i] == 1 else "NORMAL"
        confidence = probabilities[i][predictions[i]] * 100
        threat = threat_scores[i] * 100
        actual = "Attack" if true_labels[i] == 1 else "Normal"
        status = "✓" if predictions[i] == true_labels[i] else "✗"

        print(f"{i+1:<8} {pred_label:<12} {confidence:>8.1f}%   {threat:>7.1f}%  {actual:<8} {status}")

    print(f"\nProcessing time: {inference_time:.3f}s ({inference_time/len(test_batch)*1000:.1f}ms per sample)")
    print(f"Throughput: {len(test_batch)/inference_time:.1f} samples/second")

    # Generate alerts for high-threat samples
    high_threat_indices = np.where(threat_scores > 0.7)[0]
    if len(high_threat_indices) > 0:
        print(f"\n{len(high_threat_indices)} HIGH THREAT ALERTS:")
        for idx in high_threat_indices:
            alert = detector.generate_alert(predictions[idx], probabilities[idx], threat_scores[idx])
            print(f"   {alert['sample_id']}: {alert['threat_level']} threat - {alert['description']}")

print("Real-time detection test completed!")
