import logging
import pyrootutils

root = str(pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "README.md"],
    pythonpath=True,
    dotenv=True))

from src.datamodules.base import BaseDataModule
from src.datasets.GDUT import GDUT

class GDUTDataModle(BaseDataModule):
    def __init__(self, data_dir='', pre_transform=None, train_transform=None, val_transform=None, test_transform=None, on_device_train_transform=None, on_device_val_transform=None, on_device_test_transform=None, dataloader=None, mini=False, trainval=False, val_on_test=False, tta_runs=None, tta_val=False, submit=False, **kwargs):
        super().__init__(data_dir, pre_transform, train_transform, val_transform, test_transform, on_device_train_transform, on_device_val_transform, on_device_test_transform, dataloader, mini, trainval, val_on_test, tta_runs, tta_val, submit, **kwargs)
    
    _DATASET_CLASS = GDUT
    _MINIDATASET_CLASS = None

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    # root = str(pyrootutils.setup_root(__file__, pythonpath=True))
    cfg = omegaconf.OmegaConf.load(root + "/configs/datamodule/s3dis.yaml")
    cfg.data_dir = root + "/data"
    _ = hydra.utils.instantiate(cfg)
    print("???")