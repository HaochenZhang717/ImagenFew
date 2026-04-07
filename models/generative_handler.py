import torch
from abc import ABC, abstractmethod
import importlib.util as _ilu
import os as _os
_spec = _ilu.spec_from_file_location(
    "local_ddp",
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'distributed', 'DDP.py')
)
_ddp_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ddp_mod)
eDDP = _ddp_mod.eDDP

class generativeHandler(ABC):
    def __init__(self, args, rank=None):
        self.args = args
        self.device = rank if (rank is not None) else ("cuda" if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self._model = self.build_model().to(self.device)
        if self.args.ddp and (rank is not None):
            self.model = eDDP(self._model, device_ids=[self.device], output_device=self.device, find_unused_parameters=False)
        else:
            self.model = self._model

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train_iter(self, train_dataloader):
        pass

    @abstractmethod
    def sample(self, n_samples, class_label, class_metadata, test_data=None):
        pass

    def save_model(self, ckpt_dir):
        torch.save(self._model.state_dict(), ckpt_dir)