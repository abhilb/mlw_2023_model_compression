import lightning as L
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import RichProgressBar, EarlyStopping, ModelCheckpoint
from lightning.pytorch import seed_everything
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T
from torchvision.datasets import OxfordIIITPet
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
import torchmetrics
from PIL import Image

class PetClassifier(L.LightningModule):
    def __init__(self, num_classes: int, lr: float, wd: float):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.wd = wd
        self.loss_module = torch.nn.CrossEntropyLoss()
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        ftrs = self.model.classifier[-1].in_features
        self.model.classifier[-1] = torch.nn.Linear(ftrs, num_classes)
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, 0)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.model(imgs)       
        loss = self.loss_module(logits, labels)
        prob = torch.softmax(logits, dim=1)
        preds = torch.argmax(prob, dim=1)
        self.train_accuracy(preds, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.model(imgs)
        loss = self.loss_module(logits, labels)                
        prob = torch.softmax(logits, dim=1)
        preds = torch.argmax(prob, dim=1)
        self.val_accuracy(preds, labels)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.val_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.model(imgs)        
        prob = torch.softmax(logits, dim=1)
        preds = torch.argmax(prob, dim=1)
        self.test_accuracy(preds, labels)
        self.log("test_acc", self.test_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)