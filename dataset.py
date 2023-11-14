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

class PetDataModule(L.LightningDataModule):
  def __init__(self, batch_size: int, num_workers: int):
    super().__init__()
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.transform = MobileNet_V2_Weights.IMAGENET1K_V2.transforms()

  def prepare_data(self) -> None:
     self.train_val_dataset = OxfordIIITPet("data/train", 
                                        split="trainval", 
                                        target_types="category",
                                        transform=self.transform, 
                                        download=True)

  def setup(self, stage: str) -> None:
    #print(f"Setup is called for {stage}")
    if stage == "fit":
      func = torch.Generator().manual_seed(42)
      self.train_dataset, self.val_dataset = random_split(self.train_val_dataset, [.7, .3], generator=func)
    else:
      self.test_dataset = OxfordIIITPet("data/test", split="test", target_types="category", transform=self.transform, download=True)
  
  def train_dataloader(self):
    return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True)

  def test_dataloader(self):
    return DataLoader(self.test_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True)