from .ImagenTime import ImagenTime
from .sampler import DiffusionProcess
from ..generative_handler import generativeHandler

import os
import torch
import logging
from torch.utils.data import TensorDataset


class Handler(generativeHandler):
    def __init__(self, args, rank=None):
        super().__init__(args, rank)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def build_model(self):
        self.model = ImagenTime(self.args, self.args.device)
        if self.args.model_ckpt is not None:
            self._load_model(self.args.model_ckpt, self.args.device)
        return self.model

    def _split_train_batch(self, data):
        sample = data[0]
        if isinstance(sample, (tuple, list)):
            x_ts = sample[0]
            if len(sample) <= 1:
                condition_vectors = None
            elif len(sample) == 2:
                condition_vectors = sample[1]
            else:
                condition_vectors = tuple(sample[1:])
        else:
            x_ts = sample
            condition_vectors = None
        return x_ts, condition_vectors

    def _split_condition_data(self, condition_data):
        if hasattr(condition_data, "primary_tensor") and hasattr(condition_data, "texts"):
            return condition_data.primary_tensor, list(condition_data.texts)
        if isinstance(condition_data, TensorDataset):
            tensors = condition_data.tensors
            if len(tensors) <= 1:
                return tensors[0], None
            if len(tensors) == 2:
                return tensors[0], tensors[1]
            return tensors[0], tuple(tensors[1:])
        if isinstance(condition_data, (tuple, list)):
            x_ts = condition_data[0]
            if len(condition_data) <= 1:
                condition_vectors = None
            elif len(condition_data) == 2:
                condition_vectors = condition_data[1]
            else:
                condition_vectors = tuple(condition_data[1:])
            return x_ts, condition_vectors
        if torch.is_tensor(condition_data):
            return condition_data, None
        raise TypeError(
            "Expected condition data to be a TensorDataset, tensor, or tuple/list of "
            "(time_series, condition_vectors)."
        )

    def _prepare_condition_vectors(self, condition_vectors, n_samples=None):
        if condition_vectors is None:
            return None
        if isinstance(condition_vectors, (tuple, list)):
            if isinstance(condition_vectors, list) and (
                len(condition_vectors) == 0 or isinstance(condition_vectors[0], str)
            ):
                if n_samples is not None and len(condition_vectors) == 1 and n_samples > 1:
                    return condition_vectors * n_samples
                return condition_vectors
            if len(condition_vectors) != 2:
                raise ValueError("Expected token conditions as (input_ids, attention_mask).")
            input_ids, attention_mask = condition_vectors
            if not torch.is_tensor(input_ids) or not torch.is_tensor(attention_mask):
                raise TypeError("Token conditions must be tensors.")
            return (
                input_ids.to(self.args.device, dtype=torch.long),
                attention_mask.to(self.args.device, dtype=torch.long),
            )
        if not torch.is_tensor(condition_vectors):
            raise TypeError(f"Expected condition vectors to be a tensor, got {type(condition_vectors)}")
        cond = condition_vectors.to(self.args.device, dtype=torch.float32)
        if cond.ndim == 1:
            cond = cond.unsqueeze(0)
        if cond.ndim not in (2, 3):
            raise ValueError(
                f"Expected condition vectors with shape (B, D), (D,), or (B, N_ctx, D), got {tuple(cond.shape)}"
            )
        expected_dim = int(getattr(self.args, "condition_dim", getattr(self.args, "context_dim", cond.shape[-1])))
        if cond.shape[-1] != expected_dim:
            raise ValueError(
                f"Condition vector dim mismatch: expected {expected_dim}, got {cond.shape[-1]}"
            )
        if n_samples is not None and cond.shape[0] == 1 and n_samples > 1:
            cond = cond.expand(n_samples, -1)
        return cond

    def _slice_condition_batch(self, condition_bank, start, end):
        if condition_bank is None:
            return None
        if isinstance(condition_bank, list) and (
            len(condition_bank) == 0 or isinstance(condition_bank[0], str)
        ):
            return condition_bank[start:end]
        if isinstance(condition_bank, (tuple, list)):
            return tuple(t[start:end] for t in condition_bank)
        return condition_bank[start:end]

    def _condition_batch_size(self, condition_batch):
        if condition_batch is None:
            return 0
        if isinstance(condition_batch, list) and (
            len(condition_batch) == 0 or isinstance(condition_batch[0], str)
        ):
            return len(condition_batch)
        if isinstance(condition_batch, (tuple, list)):
            return int(condition_batch[0].shape[0])
        return int(condition_batch.shape[0])

    def train_iter(self, train_dataloader, logger):
        epoch = getattr(self, "epoch", None)
        train_loss = 0.0
        num_batches = 0
        for _, data in enumerate(train_dataloader, 1):
            self.optimizer.zero_grad()

            x_ts, condition_vectors = self._split_train_batch(data)
            x_ts = x_ts.to(self.args.device, dtype=torch.float32)
            condition_vectors = self._prepare_condition_vectors(condition_vectors, n_samples=x_ts.shape[0])

            x_img = self.model.ts_to_img(x_ts)
            output, weight = self.model(x_img, text_condition=condition_vectors)
            time_loss = (output - x_img).square()
            loss = (weight * time_loss).mean()
            train_loss += loss.item()
            num_batches += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.model.on_train_batch_end()

        logger.log("train/karras loss", train_loss / num_batches, step=epoch)

    def sample(self, n_samples, class_label, class_metadata, test_data=None):
        generated_set = []
        sample_batch_size = min(getattr(self.args, "batch_size", n_samples), n_samples)
        _, condition_bank = self._split_condition_data(test_data) if test_data is not None else (None, None)
        condition_bank = self._prepare_condition_vectors(condition_bank)
        with self._model.ema_scope():
            self.process = DiffusionProcess(
                self.args,
                self._model.net,
                (self.args.input_channels, self.args.img_resolution, self.args.img_resolution),
            )
            for start in range(0, n_samples, sample_batch_size):
                sample_size = min(sample_batch_size, n_samples - start)
                batch_conditions = None
                if condition_bank is not None:
                    batch_conditions = self._slice_condition_batch(condition_bank, start, start + sample_size)
                    if self._condition_batch_size(batch_conditions) < sample_size:
                        raise ValueError(
                            f"Condition bank only contains {self._condition_batch_size(condition_bank)} items, but {n_samples} samples were requested."
                        )
                x_img_sampled = self.process.sampling(sample_size, text_condition=batch_conditions)
                x_ts = self._model.img_to_ts(x_img_sampled)[:, :, : class_metadata["channels"]]
                generated_set.append(x_ts)
        return torch.concat(generated_set, dim=0)

    def save_model(self, ckpt_dir):
        state = {"model": self._model.state_dict()}
        if self.args.ema is not None:
            state["ema_model"] = self._model.model_ema.state_dict()
        torch.save(state, ckpt_dir)

    def _load_model(self, ckpt_dir, device):
        if not os.path.exists(ckpt_dir):
            os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
            logging.warning(f"No checkpoint found at {ckpt_dir}. Returned the same state as input")
        else:
            loaded_state = torch.load(ckpt_dir, map_location=device)
            self.model.load_state_dict(loaded_state["model"], strict=True)
            if "ema_model" in loaded_state and self.args.ema is not None:
                self.model.model_ema.load_state_dict(loaded_state["ema_model"], strict=True)
            logging.info("Successfully loaded previous state")
