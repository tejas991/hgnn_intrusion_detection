# Evaluation
print("\nMemory-Efficient Model Evaluation")

# Load best model
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()

# Evaluate on test set with memory management
print("Evaluating on test set")

with torch.no_grad():
    # Process test data in batches if needed
    if len(X_test) > 8000:
        print("Large test set (processed in batches)")

        all_preds = []
        all_probs = []
        batch_size = 4000

        for i in range(0, len(X_test), batch_size):
            end_idx = min(i + batch_size, len(X_test))

            X_test_batch = torch.FloatTensor(X_test[i:end_idx]).to(device)
            H_test_batch = H_test[i:end_idx, :].to(device)

            outputs = model(X_test_batch, H_test_batch)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            # Clear batch tensors
            del X_test_batch, H_test_batch, outputs, probs, preds

        # Combine results
        test_preds_np = np.hstack(all_preds)
        test_probs_np = np.vstack(all_probs)

    else:
        # Small test set - process all at once
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        H_test_tensor = H_test.to(device)

        test_outputs = model(X_test_tensor, H_test_tensor)
        test_probs = torch.softmax(test_outputs, dim=1)
        test_preds = torch.argmax(test_outputs, dim=1)

        test_preds_np = test_preds.cpu().numpy()
        test_probs_np = test_probs.cpu().numpy()

        # Clear tensors
        del X_test_tensor, H_test_tensor, test_outputs, test_probs, test_preds

# Calculate metrics
attack_probs = test_probs_np[:, 1]

test_acc = accuracy_score(y_test, test_preds_np)
precision = precision_score(y_test, test_preds_np)
recall = recall_score(y_test, test_preds_np)
f1 = f1_score(y_test, test_preds_np)

# ROC and PR curves
fpr, tpr, _ = roc_curve(y_test, attack_probs)
roc_auc = auc(fpr, tpr)

precision_curve, recall_curve, _ = precision_recall_curve(y_test, attack_probs)
pr_auc = auc(recall_curve, precision_curve)

# Display results
print("Performance Metrics:")
print("-" * 40)
print(f"Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Precision:      {precision:.4f}")
print(f"Recall:         {recall:.4f}")
print(f"F1-Score:       {f1:.4f}")
print(f"ROC AUC:        {roc_auc:.4f}")
print(f"PR AUC:         {pr_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, test_preds_np, target_names=['Normal', 'Attack']))

# Clear evaluation variables
del all_preds, all_probs
gc.collect()

print("Evaluation completed")

# Visualization
print("\nCreating Performance Visualizations")

# Create memory-efficient plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Training History
axes[0,0].plot(train_losses, label='Training Loss', color='blue')
axes[0,0].set_title('Training Loss')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss')
axes[0,0].legend()
axes[0,0].grid(True)

# 2. Validation Accuracy
val_epochs = list(range(0, len(train_losses), 2))[:len(val_accuracies)]
axes[0,1].plot(val_epochs, val_accuracies, 'g-', marker='o', label='Validation Accuracy')
axes[0,1].axhline(y=best_val_acc, color='red', linestyle='--', label=f'Best: {best_val_acc:.3f}')
axes[0,1].set_title('Validation Accuracy')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Accuracy')
axes[0,1].legend()
axes[0,1].grid(True)

# 3. ROC Curve
axes[1,0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
axes[1,0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1,0].set_xlim([0.0, 1.0])
axes[1,0].set_ylim([0.0, 1.05])
axes[1,0].set_xlabel('False Positive Rate')
axes[1,0].set_ylabel('True Positive Rate')
axes[1,0].set_title('ROC Curve')
axes[1,0].legend(loc="lower right")
axes[1,0].grid(True)

# 4. Confusion Matrix
cm = confusion_matrix(y_test, test_preds_np)
im = axes[1,1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
axes[1,1].set_title('Confusion Matrix')
axes[1,1].set_xlabel('Predicted Label')
axes[1,1].set_ylabel('True Label')

# Add text annotations
for i in range(2):
    for j in range(2):
        axes[1,1].text(j, i, str(cm[i, j]), ha='center', va='center',
                      fontsize=14, color='white' if cm[i, j] > cm.max()/2 else 'black')

axes[1,1].set_xticks([0, 1])
axes[1,1].set_yticks([0, 1])
axes[1,1].set_xticklabels(['Normal', 'Attack'])
axes[1,1].set_yticklabels(['Normal', 'Attack'])

plt.tight_layout()
plt.savefig('results/performance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Performance visualizations created!")
