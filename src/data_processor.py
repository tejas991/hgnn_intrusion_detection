# Basic set-up for HGNN Intrusion Detection project
print("HGNN Intrusion Detection Set-up")

#Imports
import os
import gc
import warnings

# Kill warnings
warnings.filterwarnings('ignore')

# Clean up memory
gc.collect()

#Libraries
import time
import json
import pickle
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve
)
from sklearn.neighbors import NearestNeighbors

# For some reason float32 behaves better with limited RAM
torch.set_default_dtype(torch.float32)

# Clear CUDA memory, if we're lucky enough to have a GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()

#Device Config
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Just some reassurance prints
print(f"Device in use: {device}")
print(f"Is CUDA available? {torch.cuda.is_available()}")

#Folder Structure
dir_list = ['data', 'data/raw', 'models', 'results']
for folder in dir_list:
    try:
        os.makedirs(folder, exist_ok=True)
    except Exception as e:
        print(f"Couldn't make folder {folder}: {e}")

print("Setup complete. Environment is ready.")

# Data Processing

print("\nSetting up Data Processor")

class MemoryEfficientNSLKDD:

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
        ]

    def get_file_size(self, file_path):
        #Get file size and estimate memory requirements
        file_size = os.path.getsize(file_path) / (1024**2)  # MB
        print(f"File size: {file_size:.1f} MB")
        return file_size

    def process_data_smart(self, file_path, is_training=True, max_samples=None):
        #Smart data processing with automatic memory management"""
        print(f"Processing {file_path}")

        # Check file size and set processing strategy
        file_size = self.get_file_size(file_path)

        if file_size > 50:  # Large file==(>50MB)
            return self._process_chunked(file_path, is_training, max_samples)
        else:
            return self._process_direct(file_path, is_training, max_samples)

    def _process_direct(self, file_path, is_training, max_samples):
        #Direct processing for smaller files"""
        print("Using direct processing")

        # Read with row limit
        df = pd.read_csv(file_path, names=self.columns, header=None, nrows=max_samples)
        print(f"Loaded {len(df)} samples")

        return self._process_dataframe(df, is_training)

    def _process_chunked(self, file_path, is_training, max_samples):
        #Chunked processing for large files"""
        print("Using chunked processing")

        # Determine optimal chunk size based on available memory
        chunk_size = 5000
        total_processed = 0

        all_X = []
        all_y = []

        # Read in chunks
        chunk_reader = pd.read_csv(file_path, names=self.columns, header=None,
                                 chunksize=chunk_size, nrows=max_samples)

        for i, chunk in enumerate(chunk_reader):
            if max_samples and total_processed >= max_samples:
                break

            print(f"  Processing chunk {i+1} ({len(chunk)} samples)")

            # Process chunk
            X_chunk, y_chunk = self._process_dataframe(chunk, is_training and i == 0)

            all_X.append(X_chunk)
            all_y.append(y_chunk)

            total_processed += len(chunk)

            # Clear chunk from memory
            del chunk, X_chunk, y_chunk
            gc.collect()

            if i % 5 == 0:  # Periodic garbage collection
                gc.collect()

        # Combine chunks efficiently
        print("Combining processed chunks")
        X_combined = np.vstack(all_X)
        y_combined = np.hstack(all_y)

        # Clear chunk lists
        del all_X, all_y
        gc.collect()

        return X_combined, y_combined

    def _process_dataframe(self, df, fit_encoders):
        #Process a single dataframe
        # Remove difficulty column
        df = df.drop('difficulty', axis=1, errors='ignore')

        # Handle categorical features efficiently
        categorical_cols = ['protocol_type', 'service', 'flag']

        for col in categorical_cols:
            if fit_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.label_encoders:
                    df[col] = df[col].astype(str)
                    # Handle unknown categories
                    unknown_mask = ~df[col].isin(self.label_encoders[col].classes_)
                    if unknown_mask.any():
                        df.loc[unknown_mask, col] = self.label_encoders[col].classes_[0]
                    df[col] = self.label_encoders[col].transform(df[col])
                else:
                    # Simple numeric encoding as fallback
                    df[col] = pd.factorize(df[col])[0]

        # Separate features and labels
        X = df.drop('label', axis=1)
        y = (df['label'] != 'normal').astype(np.int32)

        # Convert to memory-efficient format
        X_array = X.astype(np.float32).values
        y_array = y.values

        # Scale features
        if fit_encoders:
            X_scaled = self.scaler.fit_transform(X_array)
        else:
            X_scaled = self.scaler.transform(X_array)

        # Ensure memory-efficient output
        X_scaled = X_scaled.astype(np.float32)

        print(f"  Processed to shape: {X_scaled.shape}")
        return X_scaled, y_array

processor = MemoryEfficientNSLKDD()
print("Memory-efficient processor ready")
