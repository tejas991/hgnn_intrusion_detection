# HGNN Model

print("\nImplementing HGNN Model")

class HypergraphConv(nn.Module):
      #convolution layer

    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Single linear layer --> efficient
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, X, H):
        # hypergraph convolution
        # Dimensions
        N, E = H.shape

        # Normalization factors
        D_v = torch.sum(H, dim=1, keepdim=True) + 1e-8  # Node degrees [N, 1]
        D_e = torch.sum(H, dim=0, keepdim=True) + 1e-8  # Edge degrees [1, E]

        # Normalize H for message passing
        H_norm = H / torch.sqrt(D_v)  # Row normalization
        H_norm = H_norm / torch.sqrt(D_e)  # Column normalization

        # Apply transformation
        X_transformed = self.linear(X)  # [N, out_features]

        # Hypergraph message passing
        # X -> H -> H^T -> X
        messages = torch.mm(H_norm.t(), X_transformed)  # [E, out_features]
        X_updated = torch.mm(H_norm, messages)  # [N, out_features]

        return self.dropout(X_updated)

class HGNN(nn.Module):
      # HGNN for memory-constrained environments

    def __init__(self, input_dim, hidden_dims=[32, 16], output_dim=2, dropout=0.3):
        super().__init__()

        # Build architecture
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Input layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(HypergraphConv(prev_dim, hidden_dim, dropout))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim

        # Output layer
        self.layers.append(HypergraphConv(prev_dim, output_dim, dropout=0))

        self.activation = nn.ReLU(inplace=True)

    def forward(self, X, H):
        # Hidden layers
        for i, (layer, norm) in enumerate(zip(self.layers[:-1], self.norms)):
            X = layer(X, H)
            X = norm(X)
            X = self.activation(X)

        # Output layer
        X = self.layers[-1](X, H)
        return X

print("HGNN model ready")
