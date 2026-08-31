"""End-to-end Multi30k training orchestration."""

from __future__ import annotations

import time

import torch
from torch.optim.lr_scheduler import LambdaLR

from transformer.config import TrainingConfig
from transformer.dashboard.dashboard import TrainingDashboard
from transformer.dashboard.gpu import get_gpu_temperature
from transformer.dashboard.metrics import EpochMetrics, MetricsTracker
from transformer.data.multi30k import Multi30kDataModule
from transformer.evaluation.evaluator import Evaluator
from transformer.model.transformer import TransformerFactory
from transformer.training.batch import Batch, TrainState
from transformer.training.loss import DummyOptimizer, DummyScheduler, LabelSmoothing, LossComputer
from transformer.training.scheduler import learning_rate


class Trainer:
    """Train the Annotated Transformer on Multi30k with metrics tracking."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._resolve_device(config.device)
        self.data_module = Multi30kDataModule(config)
        self.metrics = MetricsTracker()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _build_model(self) -> torch.nn.Module:
        vocab_src, vocab_tgt = self.data_module.build_vocabularies()
        model = TransformerFactory.build(len(vocab_src), len(vocab_tgt), self.config)
        return model.to(self.device)

    def _build_criterion(self) -> LabelSmoothing:
        pad_idx = self.data_module.pad_idx
        return LabelSmoothing(
            size=len(self.data_module.vocab_tgt),
            padding_idx=pad_idx,
            smoothing=self.config.label_smoothing,
        ).to(self.device)

    def run_epoch(
        self,
        data_iter,
        model: torch.nn.Module,
        loss_computer: LossComputer,
        optimizer,
        scheduler,
        mode: str = "train",
        accum_iter: int = 1,
        train_state: TrainState | None = None,
    ) -> tuple[float, float, float, TrainState]:
        if train_state is None:
            train_state = TrainState()

        total_loss = 0.0
        total_kl = 0.0
        total_accuracy = 0.0
        total_tokens = 0
        n_batches = 0
        start = time.time()
        step_tokens = 0

        for i, batch in enumerate(data_iter):
            out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            loss_val, loss_node, accuracy = loss_computer(
                out, batch.tgt_y, batch.ntokens
            )

            if mode.startswith("train"):
                loss_node.backward()
                train_state.step += 1
                train_state.samples += batch.src.shape[0]
                train_state.tokens += batch.ntokens
                if i % accum_iter == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    train_state.accum_step += 1
                scheduler.step()
                self.metrics.record_batch(train_state.step, loss_val.item() / batch.ntokens)

            total_loss += loss_val.item()
            total_kl += loss_val.item()
            total_accuracy += accuracy
            total_tokens += batch.ntokens
            step_tokens += batch.ntokens
            n_batches += 1

            if i % 40 == 1 and mode.startswith("train"):
                elapsed = time.time() - start
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Step {i:6d} | Loss: {loss_val.item() / batch.ntokens:6.2f} "
                    f"| Tok/s: {step_tokens / max(elapsed, 1e-6):7.1f} | LR: {lr:.2e}"
                )
                start = time.time()
                step_tokens = 0

            del loss_node

        avg_loss = total_loss / max(total_tokens, 1)
        avg_kl = total_kl / max(total_tokens, 1)
        avg_accuracy = total_accuracy / max(n_batches, 1)
        return avg_loss, avg_kl, avg_accuracy, train_state

    def train(self) -> MetricsTracker:
        torch.manual_seed(self.config.seed)
        model = self._build_model()
        criterion = self._build_criterion()
        loss_computer = LossComputer(model.generator, criterion)

        train_loader, valid_loader = self.data_module.create_dataloaders(
            self.device, batch_size=self.config.batch_size
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.base_lr,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: learning_rate(
                step,
                self.config.d_model,
                factor=1,
                warmup=self.config.warmup,
            ),
        )

        evaluator = Evaluator(
            model,
            self.data_module,
            loss_computer,
            self.device,
            max_decode_len=self.config.max_decode_len,
        )

        pad_idx = self.data_module.pad_idx
        train_state = TrainState()

        print(f"Training on {self.device} for {self.config.num_epochs} epochs")
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs} — Training")
            model.train()
            train_loss, train_kl, train_acc, train_state = self.run_epoch(
                (Batch(b[0], b[1], pad_idx) for b in train_loader),
                model,
                loss_computer,
                optimizer,
                scheduler,
                mode="train",
                accum_iter=self.config.accum_iter,
                train_state=train_state,
            )

            print(f"Epoch {epoch + 1}/{self.config.num_epochs} — Validation")
            model.eval()
            val_loss, val_kl, val_acc, _ = self.run_epoch(
                (Batch(b[0], b[1], pad_idx) for b in valid_loader),
                model,
                loss_computer,
                DummyOptimizer(),
                DummyScheduler(),
                mode="eval",
            )

            eval_metrics = {}
            if self.config.eval_every_epoch:
                print(f"Epoch {epoch + 1}/{self.config.num_epochs} — Evaluation")
                eval_metrics = evaluator.evaluate(
                    valid_loader, max_bleu_samples=self.config.bleu_max_samples
                )

            gpu_temp = get_gpu_temperature()
            lr = optimizer.param_groups[0]["lr"]

            epoch_metrics = EpochMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                train_accuracy=train_acc,
                val_accuracy=val_acc,
                train_perplexity=MetricsTracker.loss_to_perplexity(train_loss),
                val_perplexity=MetricsTracker.loss_to_perplexity(val_loss),
                train_kl=train_kl,
                val_kl=val_kl,
                corpus_bleu=eval_metrics.get("corpus_bleu"),
                corpus_perplexity=eval_metrics.get("corpus_perplexity"),
                gpu_temp=gpu_temp,
                learning_rate=lr,
            )
            self.metrics.record_epoch(epoch_metrics)

            print(
                f"  train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_ppl={epoch_metrics.train_perplexity:.2f} "
                f"val_ppl={epoch_metrics.val_perplexity:.2f} "
                f"bleu={epoch_metrics.corpus_bleu} gpu_temp={gpu_temp}"
            )

            ckpt_path = self.config.checkpoint_path(epoch)
            torch.save(model.state_dict(), ckpt_path)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        torch.save(model.state_dict(), self.config.checkpoint_path())
        self.metrics.save(self.config.metrics_path)
        dashboard = TrainingDashboard(self.metrics)
        dashboard_path = dashboard.render(self.config.dashboard_path)
        print(f"\nDashboard saved to {dashboard_path}")
        return self.metrics

    def evaluate_only(self, checkpoint: str | None = None) -> dict:
        """Run corpus_perplexity and corpus_bleu on the validation set."""
        model = self._build_model()
        ckpt = checkpoint or str(self.config.checkpoint_path())
        model.load_state_dict(
            torch.load(ckpt, map_location=self.device, weights_only=True)
        )
        criterion = self._build_criterion()
        loss_computer = LossComputer(model.generator, criterion)
        _, valid_loader = self.data_module.create_dataloaders(self.device)

        evaluator = Evaluator(
            model,
            self.data_module,
            loss_computer,
            self.device,
            max_decode_len=self.config.max_decode_len,
        )
        return evaluator.evaluate(valid_loader)
