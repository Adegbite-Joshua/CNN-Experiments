import torch
import torch.nn as nn
import lightning as L



class SimpleCNNModel(L.LightningModule):
    """ Building simple CNN """
    def __init__(self, input_size, hidden_units, num_classes):
        super().__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_classes),            
        )
        
    def forward(self, x):
        return self.layers(x)
        
        
    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        return loss
        
    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        return loss
    
    def _common_step(self, batch, batch_idx):
        x, y = batch
        
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        
        return loss, scores, y
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-3
        )
        return optimizer