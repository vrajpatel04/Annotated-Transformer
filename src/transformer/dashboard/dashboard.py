"""Interactive Plotly training dashboard."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from transformer.dashboard.metrics import MetricsTracker


class TrainingDashboard:
    """Render an HTML dashboard of training and evaluation metrics."""

    def __init__(self, tracker: MetricsTracker):
        self.tracker = tracker

    def render(self, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Loss (Train vs Val)",
                "Token Accuracy",
                "Perplexity",
                "KL Divergence (Label Smoothing)",
                "Corpus BLEU",
                "GPU Temperature (°C)",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        epochs = [m.epoch for m in self.tracker.history]
        if not epochs:
            fig.add_annotation(
                text="No training metrics recorded yet.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"size": 16},
            )
        else:
            self._add_epoch_series(fig, epochs)

        if self.tracker.batch_losses:
            fig.add_trace(
                go.Scatter(
                    x=self.tracker.batch_steps,
                    y=self.tracker.batch_losses,
                    mode="lines",
                    name="Batch Loss",
                    line={"color": "#636efa", "width": 1},
                    opacity=0.35,
                ),
                row=1,
                col=1,
            )

        fig.update_layout(
            title={
                "text": "Annotated Transformer — Multi30k Training Dashboard",
                "x": 0.5,
                "xanchor": "center",
            },
            height=1100,
            width=1200,
            template="plotly_white",
            legend={"orientation": "h", "y": -0.05},
            showlegend=True,
        )
        fig.update_xaxes(title_text="Epoch", row=3, col=1)
        fig.update_xaxes(title_text="Epoch", row=3, col=2)
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        return output_path

    def _add_epoch_series(self, fig: go.Figure, epochs: list[int]) -> None:
        history = self.tracker.history

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[m.train_loss for m in history],
                mode="lines+markers",
                name="Train Loss",
                line={"color": "#636efa"},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[m.val_loss for m in history],
                mode="lines+markers",
                name="Val Loss",
                line={"color": "#ef553b"},
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[m.train_accuracy for m in history],
                mode="lines+markers",
                name="Train Acc",
                line={"color": "#00cc96"},
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[m.val_accuracy for m in history],
                mode="lines+markers",
                name="Val Acc",
                line={"color": "#ab63fa"},
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[m.train_perplexity for m in history],
                mode="lines+markers",
                name="Train PPL",
                line={"color": "#636efa"},
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[m.val_perplexity for m in history],
                mode="lines+markers",
                name="Val PPL",
                line={"color": "#ef553b"},
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[m.corpus_perplexity or 0 for m in history],
                mode="lines+markers",
                name="Corpus PPL",
                line={"color": "#ffa15a", "dash": "dash"},
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[m.train_kl for m in history],
                mode="lines+markers",
                name="Train KL",
                line={"color": "#636efa"},
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[m.val_kl for m in history],
                mode="lines+markers",
                name="Val KL",
                line={"color": "#ef553b"},
            ),
            row=2,
            col=2,
        )

        bleu_epochs = [m.epoch for m in history if m.corpus_bleu is not None]
        bleu_scores = [m.corpus_bleu for m in history if m.corpus_bleu is not None]
        if bleu_scores:
            fig.add_trace(
                go.Scatter(
                    x=bleu_epochs,
                    y=bleu_scores,
                    mode="lines+markers",
                    name="Corpus BLEU",
                    line={"color": "#19d3f3"},
                ),
                row=3,
                col=1,
            )

        gpu_epochs = [m.epoch for m in history if m.gpu_temp is not None]
        gpu_temps = [m.gpu_temp for m in history if m.gpu_temp is not None]
        if gpu_temps:
            fig.add_trace(
                go.Scatter(
                    x=gpu_epochs,
                    y=gpu_temps,
                    mode="lines+markers",
                    name="GPU Temp",
                    line={"color": "#ff6692"},
                ),
                row=3,
                col=2,
            )
