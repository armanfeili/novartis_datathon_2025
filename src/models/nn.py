"""
Neural network model for Novartis Datathon 2025.

Simple MLP with optional sample weight support.
"""

from typing import Optional
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd

from .base import BaseModel

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# Default NN configuration for generic erosion forecasting
DEFAULT_CONFIG = {
    'architecture': {
        'mlp': {
            'hidden_layers': [256, 128, 64],
            'dropout': 0.3,
            'activation': 'relu',
            'batch_norm': True
        }
    },
    'training': {
        'batch_size': 256,
        'epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'early_stopping_patience': 10,
        'scheduler': 'reduce_on_plateau',
        'sample_weights': True
    }
}


class SimpleMLP(nn.Module):
    """Simple MLP for regression."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list,
        dropout: float = 0.3,
        output_dim: int = 1,
        batch_norm: bool = True
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class NNModel(BaseModel):
    """Neural network model with sample weight support."""
    
    def __init__(self, config: dict):
        """
        Initialize neural network model.
        
        Args:
            config: Configuration with 'architecture' and 'training' sections
        """
        super().__init__(config)
        
        # Merge with defaults
        self.arch_config = {
            **DEFAULT_CONFIG['architecture'].get('mlp', {}),
            **config.get('architecture', {}).get('mlp', {})
        }
        self.train_config = {
            **DEFAULT_CONFIG['training'],
            **config.get('training', {})
        }
        
        self.device = get_device()
        self.model = None
        self.input_dim = None
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None
    ) -> 'NNModel':
        """
        Train neural network with optional sample weights.
        
        Sample weights are used via WeightedRandomSampler for balanced batch sampling.
        """
        self.feature_names = list(X_train.columns)
        
        # Convert to numpy arrays (handle DataFrames)
        X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        
        # Handle missing values
        X_train_np = np.nan_to_num(X_train_np, nan=0.0)
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train_np)
        y_train_t = torch.FloatTensor(y_train_np).reshape(-1, 1)
        
        train_ds = TensorDataset(X_train_t, y_train_t)
        
        # Create sampler with sample weights if provided
        sampler = None
        shuffle = True
        if sample_weight is not None and self.train_config.get('sample_weights', True):
            weights = sample_weight.values if hasattr(sample_weight, 'values') else sample_weight
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True
            )
            shuffle = False  # Can't use shuffle with sampler
        
        train_dl = DataLoader(
            train_ds,
            batch_size=self.train_config.get('batch_size', 256),
            shuffle=shuffle if sampler is None else False,
            sampler=sampler
        )
        
        val_dl = None
        if X_val is not None and y_val is not None:
            X_val_np = X_val.values if hasattr(X_val, 'values') else X_val
            y_val_np = y_val.values if hasattr(y_val, 'values') else y_val
            X_val_np = np.nan_to_num(X_val_np, nan=0.0)
            
            X_val_t = torch.FloatTensor(X_val_np)
            y_val_t = torch.FloatTensor(y_val_np).reshape(-1, 1)
            val_ds = TensorDataset(X_val_t, y_val_t)
            val_dl = DataLoader(val_ds, batch_size=self.train_config.get('batch_size', 256))
        
        # Initialize model
        self.input_dim = X_train_np.shape[1]
        self.model = SimpleMLP(
            input_dim=self.input_dim,
            hidden_layers=self.arch_config.get('hidden_layers', [256, 128, 64]),
            dropout=self.arch_config.get('dropout', 0.3),
            batch_norm=self.arch_config.get('batch_norm', True)
        ).to(self.device)
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.get('learning_rate', 1e-3),
            weight_decay=self.train_config.get('weight_decay', 1e-5)
        )
        criterion = nn.MSELoss()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training loop with early stopping
        epochs = self.train_config.get('epochs', 100)
        patience = self.train_config.get('early_stopping_patience', 10)
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for X_b, y_b in train_dl:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                pred = self.model(X_b)
                loss = criterion(pred, y_b)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_dl)
            
            # Validation
            if val_dl:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_b, y_b in val_dl:
                        X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                        pred = self.model(X_b)
                        loss = criterion(pred, y_b)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_dl)
                scheduler.step(avg_val_loss)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}")
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            else:
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f}")
        
        # Load best model if we did validation
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using trained model."""
        self.model.eval()
        
        X_np = X.values if hasattr(X, 'values') else X
        X_np = np.nan_to_num(X_np, nan=0.0)
        
        X_t = torch.FloatTensor(X_np).to(self.device)
        
        with torch.no_grad():
            preds = self.model(X_t)
        
        return preds.cpu().numpy().flatten()
    
    def save(self, path: str) -> None:
        """Save model checkpoint to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'arch_config': self.arch_config,
            'feature_names': self.feature_names
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'NNModel':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        
        config = {
            'architecture': {'mlp': checkpoint.get('arch_config', {})}
        }
        instance = cls(config)
        instance.input_dim = checkpoint['input_dim']
        instance.feature_names = checkpoint.get('feature_names', [])
        
        # Rebuild model architecture
        instance.model = SimpleMLP(
            input_dim=instance.input_dim,
            hidden_layers=instance.arch_config.get('hidden_layers', [256, 128, 64]),
            dropout=instance.arch_config.get('dropout', 0.3),
            batch_norm=instance.arch_config.get('batch_norm', True)
        ).to(instance.device)
        
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        return instance
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Compute feature importance via gradient-based attribution.
        
        Note: For neural networks, feature importance is approximated
        using the absolute mean of first layer weights.
        """
        if self.model is None or len(self.feature_names) == 0:
            return pd.DataFrame(columns=['feature', 'importance'])
        
        # Get first layer weights
        first_layer = self.model.net[0]
        if isinstance(first_layer, nn.Linear):
            weights = first_layer.weight.detach().cpu().numpy()
            importance = np.abs(weights).mean(axis=0)
            
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        
        return pd.DataFrame(columns=['feature', 'importance'])
