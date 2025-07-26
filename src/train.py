# Training

print("\nMemory-Efficient Training Pipeline")

# model configuration
config = {
    'input_dim': X_train.shape[1],
    'hidden_dims': [32, 16],  # Smaller architecture
    'output_dim': 2,
    'dropout': 0.3,
    'learning_rate': 0.01,
    'weight_decay': 1e-4,
    'epochs': 40,  # Fewer epochs for efficiency
    'patience': 8
}

print(f"Model configuration: {config}")

# Initialize model
model = HGNN(
    input_dim=config['input_dim'],
    hidden_dims=config['hidden_dims'],
    output_dim=config['output_dim'],
    dropout=config['dropout']
).to(device)

# Training setup
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config['learning_rate'],
    weight_decay=config['weight_decay']
)

criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=4, factor=0.5, verbose=True
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Convert data to tensors in batches to save memory
def create_tensor_batch(X, y, H, batch_size=None):
    """Create tensor batches for memory efficiency"""
    if batch_size is None:
        batch_size = len(X)

    X_tensor = torch.FloatTensor(X[:batch_size]).to(device)
    y_tensor = torch.LongTensor(y[:batch_size]).to(device)
    H_tensor = H[:batch_size, :].to(device)

    return X_tensor, y_tensor, H_tensor

# Training loop
train_losses = []
val_accuracies = []
best_val_acc = 0
patience_counter = 0

print("Starting training")

for epoch in tqdm(range(config['epochs']), desc="Training"):
    # Training phase
    model.train()

    # Process training data in batches if needed
    if len(X_train) > 10000:
        # Large dataset - use gradient accumulation
        optimizer.zero_grad()
        total_loss = 0
        batch_size = 5000

        for i in range(0, len(X_train), batch_size):
            end_idx = min(i + batch_size, len(X_train))

            X_batch, y_batch, H_batch = create_tensor_batch(
                X_train[i:end_idx], y_train[i:end_idx],
                H_train[i:end_idx, :], end_idx - i
            )

            outputs = model(X_batch, H_batch)
            loss = criterion(outputs, y_batch) / ((len(X_train) - 1) // batch_size + 1)
            loss.backward()
            total_loss += loss.item()

            # Clear batch from memory
            del X_batch, y_batch, H_batch, outputs

        optimizer.step()
        epoch_loss = total_loss

    else:
        # Small dataset - normal training
        optimizer.zero_grad()

        X_train_tensor, y_train_tensor, H_train_tensor = create_tensor_batch(
            X_train, y_train, H_train
        )

        outputs = model(X_train_tensor, H_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        epoch_loss = loss.item()

        # Clear tensors
        del X_train_tensor, y_train_tensor, H_train_tensor, outputs

    train_losses.append(epoch_loss)

    # Validation
    if epoch % 2 == 0:  # Validate every 2 epochs to save time
        model.eval()
        with torch.no_grad():
            X_val_tensor, y_val_tensor, H_val_tensor = create_tensor_batch(
                X_val, y_val, H_val
            )

            val_outputs = model(X_val_tensor, H_val_tensor)
            val_preds = torch.argmax(val_outputs, dim=1)
            val_acc = (val_preds == y_val_tensor).float().mean().item()
            val_accuracies.append(val_acc)

            # Clear validation tensors
            del X_val_tensor, y_val_tensor, H_val_tensor, val_outputs, val_preds

        scheduler.step(val_acc)

        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_model.pth')
        else:
            patience_counter += 1

        # Progress logging
        if epoch % 10 == 0 or epoch == config['epochs'] - 1:
            print(f"Epoch {epoch:3d}: Loss={epoch_loss:.4f}, Val Acc={val_acc:.4f}, Best={best_val_acc:.4f}")

        # Early stopping
        if patience_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch}")
            break

    # Periodic garbage collection
    if epoch % 5 == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}")
