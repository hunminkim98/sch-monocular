import pytorch_lightning as pl
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from hydra.utils import instantiate
from torch.utils.data import DataLoader, ConcatDataset, Subset
from omegaconf import ListConfig, DictConfig, OmegaConf
from hmr4d.utils.pylogger import Log
from numpy.random import choice
from torch.utils.data import default_collate


import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def collate_fn(batch):
    """meta 정보를 처리하고 반환 dictionary에 batch size를 추가합니다.

    인자:
        batch: 각 data point dictionary로 구성된 list
    """
    # batch 안의 모든 항목이 같은 key를 가진다고 가정합니다.
    return_dict = {}
    for k in batch[0].keys():
        if k.startswith("meta"):  # data 정보는 batch로 묶지 않습니다.
            return_dict[k] = [d[k] for d in batch]
        else:
            return_dict[k] = default_collate([d[k] for d in batch])
    return_dict["B"] = len(batch)
    return return_dict


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_opts: DictConfig, loader_opts: DictConfig,
                 limit_each_trainset=None):
        """여러 dataset에 사용할 수 있는 범용 data module입니다.

        학습에는 ``ConcatDataset``을 사용합니다. Validation과 test에는
        sequential 방식의 ``CombinedLoader``를 사용하며, 각 iterable을 순서대로
        모두 소비하고 ``(data, idx, iterable_idx)``를 반환합니다.

        인자:
            dataset_opts: dataset target 설정.
                예: ``dataset_opts.train = {_target_: ..., limit_size: None}``
            loader_opts: dataset loader 옵션
            limit_each_trainset: 각 dataset의 최대 크기. ``None``이면 제한하지
                않으며 디버깅할 때 유용합니다.
        """
        super().__init__()
        self.loader_opts = loader_opts
        self.limit_each_trainset = limit_each_trainset

        # 학습 dataset은 하나의 ConcatDataset으로 합칩니다.
        if "train" in dataset_opts:
            assert "train" in self.loader_opts, "train not in loader_opts"
            split_opts = dataset_opts.get("train")
            assert isinstance(split_opts, DictConfig), "split_opts should be a dict for each dataset"
            dataset = []
            dataset_num = len(split_opts)
            for idx, (k, v) in enumerate(split_opts.items()):
                dataset_i = instantiate(v)
                if self.limit_each_trainset:
                    dataset_i = Subset(dataset_i, choice(len(dataset_i), self.limit_each_trainset))
                dataset.append(dataset_i)
                Log.info(f"[Train Dataset][{idx+1}/{dataset_num}]: name={k}, size={len(dataset[-1])}, {v._target_}")
            dataset = ConcatDataset(dataset)
            self.trainset = dataset
            Log.info(f"[Train Dataset][All]: ConcatDataset size={len(dataset)}")
            Log.info(f"")

        # Validation과 test dataset은 순서대로 읽습니다.
        for split in ("val", "test"):
            if split not in dataset_opts:
                continue
            assert split in self.loader_opts, f"split={split} not in loader_opts"
            split_opts = dataset_opts.get(split)
            assert isinstance(split_opts, DictConfig), "split_opts should be a dict for each dataset"
            dataset = []
            dataset_num = len(split_opts)
            for idx, (k, v) in enumerate(split_opts.items()):
                dataset.append(instantiate(v))
                dataset_type = "Val Dataset" if split == "val" else "Test Dataset"
                Log.info(f"[{dataset_type}][{idx+1}/{dataset_num}]: name={k}, size={len(dataset[-1])}, {v._target_}")
            setattr(self, f"{split}sets", dataset)
            Log.info(f"")

    def train_dataloader(self):
        if hasattr(self, "trainset"):
            return DataLoader(
                self.trainset,
                shuffle=True,
                num_workers=self.loader_opts.train.num_workers,
                persistent_workers=True and self.loader_opts.train.num_workers > 0,
                batch_size=self.loader_opts.train.batch_size,
                drop_last=True,
                collate_fn=collate_fn,
            )
        else:
            return super().train_dataloader()

    def val_dataloader(self):
        if hasattr(self, "valsets"):
            loaders = []
            for valset in self.valsets:
                loaders.append(
                    DataLoader(
                        valset,
                        shuffle=False,
                        num_workers=self.loader_opts.val.num_workers,
                        persistent_workers=True and self.loader_opts.val.num_workers > 0,
                        batch_size=self.loader_opts.val.batch_size,
                        collate_fn=collate_fn,
                    )
                )
            return CombinedLoader(loaders, mode="sequential")
        else:
            return None

    def test_dataloader(self):
        if hasattr(self, "testsets"):
            loaders = []
            for testset in self.testsets:
                loaders.append(
                    DataLoader(
                        testset,
                        shuffle=False,
                        num_workers=self.loader_opts.test.num_workers,
                        persistent_workers=False,
                        batch_size=self.loader_opts.test.batch_size,
                        collate_fn=collate_fn,
                    )
                )
            return CombinedLoader(loaders, mode="sequential")
        else:
            return super().test_dataloader()
