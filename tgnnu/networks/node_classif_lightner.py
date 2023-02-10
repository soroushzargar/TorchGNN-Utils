import torch
import torch.nn.functional as F
import math

class NodeLevelGNN(object):
    def __init__(self, model, loss_fn=None):
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.optimizer = None
        self.loss_fn = loss_fn or F.cross_entropy

    def set_optimizer(self, optimizer_lambda=None, optimizer_keyargs=None):
        if optimizer_keyargs is None:
            optimizer_keyargs = {"weight_decay": 1e-2}

        if optimizer_lambda is None:
            optimizer_lambda = lambda model_params: torch.optim.Adam(model_params, **optimizer_keyargs)
            
        self.optimizer = optimizer_lambda(self.model.parameters())

    def forward(self, X):
        return self.model(X)
    
    def loss(self, X, y, mask=None):
        if mask is None:
            mask = torch.ones_like(y).bool()
        if mask == "train":
            mask = X.train_mask
        elif mask == "val":
            mask = X.val_mask
        elif mask == "test":
            mask = X.test_mask
        
        y_hat = self.forward(X)
        return self.loss_fn(y_hat[mask], y[mask])

    def train_iteration(self, X, y, mask=None):
        self.optimizer.zero_grad()
        loss = self.loss(X, y, mask)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(self, X, y=None, epochs=100, train_mask=None, val_mask=None, early_stopping=True, patience=10, warm_steps=0, verbose=False, ignore_validation=False):
        X = X.to(self.device)
        if train_mask is None:
            train_mask = X.train_mask
        if val_mask is None:
            val_mask = X.val_mask

        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        
        if y == None:
            y = X.y

        best_val_loss = math.inf
        best_val_epoch = 0
        best_model = self.model.state_dict()
        for epoch in range(epochs):
            loss = self.train_iteration(X, y, train_mask)
            if ignore_validation:
                val_loss = 0
                best_val_epoch = epoch
                best_model = self.model.state_dict()
            else:
                val_loss = self.loss(X, y, val_mask).item()
                if val_loss < best_val_loss and epoch > warm_steps:
                    best_val_loss = val_loss
                    best_val_epoch = epoch
                    best_model = self.model.state_dict()
                if early_stopping and epoch - best_val_epoch > patience:
                    break
            if verbose:
                print(f"Epoch {epoch:03d}: Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
        self.model.load_state_dict(best_model)

    def predict(self, X, mask=None):
        self.model.eval()
        with torch.no_grad():
            y_hat = self.forward(X)
        if mask is None:
            mask = torch.ones_like(y_hat[:, 0]).bool()
        return y_hat[mask]

    def evaluate(self, X, y, mask=None):
        if mask is None:
            mask = torch.ones_like(y).bool()
        y_hat = self.predict(X)
        return ((y_hat.argmax(dim=1)[mask] == y[mask]).sum() / mask.sum()).item()
    