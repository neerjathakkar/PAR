from typing import Any, Dict, Optional
import os
from omegaconf import DictConfig
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from trajdata import AgentBatch, UnifiedDataset, AgentType


class TrajDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        cfg: DictConfig,
        train: bool = True,
    ):
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val:   Optional[Dataset] = None
        self.cfg = cfg
        
    @property
    def num_classes(self):
        return 80

    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val:
            dname = "nusc_trainval"
            split = "train"
            self.data_train = UnifiedDataset(
                desired_data=[f"{dname}-{split}"],
                data_dirs={  
                    dname: "/datasets/nuscenes_2024-09-05_1958"
                },
                incl_raster_map=True,
                raster_map_params={
                        "px_per_m": 2,
                        "map_size_px": 224,
                        "offset_frac_xy": (-0.5, 0.0),
                    },
                history_sec=(self.cfg.hist_sec,self.cfg.hist_sec),
                future_sec=(self.cfg.fut_sec,self.cfg.fut_sec),
                only_types=[AgentType.VEHICLE],
                max_neighbor_num=self.cfg.num_agents - 1
            )
            split = "val"
            
            self.data_val = UnifiedDataset( 
                desired_data=[f"{dname}-{split}"],
                data_dirs={  
                    dname: "/datasets/nuscenes_2024-09-05_1958"
                },
                incl_raster_map=True,
                raster_map_params={
                        "px_per_m": 2,
                        "map_size_px": 224,
                        "offset_frac_xy": (-0.5, 0.0),
                    },
                history_sec=(self.cfg.hist_sec,self.cfg.hist_sec),
                future_sec=(self.cfg.fut_sec,self.cfg.fut_sec),
                only_types=[AgentType.VEHICLE],
                max_neighbor_num=self.cfg.num_agents - 1
            )
            
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.cfg.train_batch_size,
            num_workers=self.hparams.cfg.train_num_workers,
            pin_memory=self.hparams.cfg.pin_memory,
            shuffle=True,
            collate_fn=self.data_train.get_collate_fn(),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.cfg.test_batch_size if not(self.hparams.cfg.full_seq_render) else 1,
            num_workers=self.hparams.cfg.test_num_workers,
            pin_memory=self.hparams.cfg.pin_memory,
            shuffle=False,
            collate_fn=self.data_val.get_collate_fn(),
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    pass