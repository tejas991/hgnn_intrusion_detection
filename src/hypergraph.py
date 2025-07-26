# HyperGraph Construction
# ================================================
print("\nSetting up Hypergraph Constructor")

def construct_hypergraph(X, k_neighbors=5, max_hyperedges=None):
    #Memory-efficient hypergraph construction
    n_samples = len(X)
    k = min(k_neighbors, n_samples - 1, 6)  # Limiting k --> memory efficiency

    print(f"Building {k}-HGNN hypergraph for {n_samples} samples")

    # Sampling Strategy
    if n_samples > 15000:
        print("Large dataset detected - using sampled hypergraph")
        # Sample a subset for hypergraph construction
        sample_size = min(10000, n_samples)
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        X_sample = X[sample_indices]

        # Build k-NN on sample
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine', n_jobs=1)
        nbrs.fit(X_sample)

        # Create hypergraph matrix
        hyperedges = []
        distances, indices = nbrs.kneighbors(X_sample)

        for i, neighbors in enumerate(indices):
            original_neighbors = [sample_indices[neighbor] for neighbor in neighbors]
            hyperedges.append(original_neighbors)

        # Create incidence matrix
        n_hyperedges = len(hyperedges)
        H = torch.zeros(n_samples, n_hyperedges, dtype=torch.float32)

        for j, hyperedge in enumerate(hyperedges):
            for node in hyperedge:
                H[node, j] = 1.0

    else:
        # Standard construction for smaller datasets
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine', n_jobs=1)
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)

        # Create hyperedges
        hyperedges = []
        for i, neighbors in enumerate(indices):
            hyperedges.append(neighbors.tolist())

        # Create incidence matrix
        n_hyperedges = len(hyperedges)
        H = torch.zeros(n_samples, n_hyperedges, dtype=torch.float32)

        for j, hyperedge in enumerate(hyperedges):
            for node in hyperedge:
                H[node, j] = 1.0

    print(f"Created hypergraph: {H.shape}")

    del hyperedges
    if 'indices' in locals():
        del indices
    gc.collect()

    return H

print("HyperGraph Constructor Ready")
