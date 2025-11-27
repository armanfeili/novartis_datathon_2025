import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from .base import BaseModel
from ..utils import get_device

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout=0.2, output_dim=1):
        super().__init__()
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class NNModel(BaseModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.arch_config = config.get('architecture', {}).get('mlp', {})
        self.train_config = config.get('training', {})
        self.device = get_device()
        self.model = None # Initialized in fit when input dim is known

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
        
        train_ds = TensorDataset(X_train_t, y_train_t)
        train_dl = DataLoader(
            train_ds, 
            batch_size=self.train_config.get('batch_size', 256),
            shuffle=True
        )
        
        val_dl = None
        if X_val is not None and y_val is not None:
            X_val_t = torch.FloatTensor(X_val)
            y_val_t = torch.FloatTensor(y_val).reshape(-1, 1)
            val_ds = TensorDataset(X_val_t, y_val_t)
            val_dl = DataLoader(val_ds, batch_size=self.train_config.get('batch_size', 256))

        # Initialize model
        input_dim = X_train.shape[1]
        self.model = SimpleMLP(
            input_dim=input_dim,
            hidden_layers=self.arch_config.get('hidden_layers', [256, 128]),
            dropout=self.arch_config.get('dropout', 0.2)
        ).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.train_config.get('learning_rate', 1e-3))
        criterion = nn.MSELoss()
        
        # Training loop
        epochs = self.train_config.get('epochs', 10)
        for epoch in range(epochs):
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
                # print(f"Epoch {epoch}: Train Loss {train_loss/len(train_dl):.4f}, Val Loss {val_loss/len(val_dl):.4f}")

    def predict(self, X) -> np.ndarray:
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            preds = self.model(X_t)
        return preds.cpu().numpy().flatten()

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        # Note: This requires re-initializing the model structure first
        # which might be tricky if we don't know input_dim. 
        # For now, assuming model is already initialized or we save full model.
        # Better to save checkpoint dict with config.
        self.model.load_state_dict(torch.load(path))
