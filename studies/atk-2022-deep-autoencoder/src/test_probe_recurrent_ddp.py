from __future__ import annotations

import os
import unittest

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
import torch.nn.functional as functional

from paper_literal_models import build_model
from probe_recurrent_ddp import (
    ProbeLossModule,
    _apply_optimizer_step,
    _optimizer_for_probe,
)


class _IdentityModel(torch.nn.Module):
    def __init__(self, loss: str = "mse") -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.loss = loss

    @property
    def losses(self) -> list[torch.Tensor]:
        return []

    def forward(
        self, inputs: torch.Tensor, training: bool | None = None
    ) -> torch.Tensor:
        del training
        return inputs * self.scale

    def compute_loss(
        self,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        y_pred: torch.Tensor,
        training: bool,
    ) -> torch.Tensor:
        del x, training
        if self.loss == "bce":
            return functional.binary_cross_entropy(
                y_pred.reshape(-1), y.to(dtype=y_pred.dtype).reshape(-1)
            )
        return functional.mse_loss(y_pred, y)


class ProbeLossTests(unittest.TestCase):
    def test_sae_probe_performs_finite_weight_update(self) -> None:
        model = _IdentityModel()
        wrapped = ProbeLossModule.build("lstm_sae", model)
        optimizer = _optimizer_for_probe(wrapped, "adam")
        values = torch.ones((4, 8), dtype=torch.float32)
        before = model.scale.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        loss = wrapped(values, values)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(torch.abs(model.scale.detach() - before)), 0.0)

    def test_supervised_probe_uses_binary_cross_entropy(self) -> None:
        model = _IdentityModel("bce")
        wrapped = ProbeLossModule.build("supervised_lstm", model)
        values = torch.full((4, 1), 0.5, dtype=torch.float32)
        labels = torch.ones((4, 1), dtype=torch.float32)
        loss = wrapped(values, labels)
        self.assertAlmostEqual(float(loss.detach()), -torch.log(torch.tensor(0.25)).item())

    def test_supervised_probe_rejects_invalid_probability(self) -> None:
        model = _IdentityModel("bce")
        model.scale.data.fill_(3.0)
        wrapped = ProbeLossModule.build("supervised_lstm", model)
        with self.assertRaisesRegex(FloatingPointError, r"left \[0, 1\]"):
            wrapped(torch.ones((2, 1)), torch.ones((2, 1)))

    def test_compiled_keras_optimizer_updates_paper_model(self) -> None:
        bundle = build_model(
            "lstm_sae",
            8,
            {"encoder_widths": (4, 2), "dropout": 0.0},
            seed=11,
        )
        wrapped = ProbeLossModule.build("lstm_sae", bundle.model)
        parameters = [parameter for parameter in wrapped.parameters() if parameter.requires_grad]
        device = parameters[0].device
        values = torch.rand((4, 8), device=device)
        before = [parameter.detach().clone() for parameter in parameters]

        wrapped.zero_grad(set_to_none=True)
        loss = wrapped(values, values)
        loss.backward()
        _apply_optimizer_step(
            optimizer_engine="keras",
            torch_optimizer=None,
            paper_model=bundle.model,
        )

        deltas = [
            torch.linalg.vector_norm(parameter.detach() - original).item()
            for parameter, original in zip(parameters, before, strict=True)
        ]
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(max(deltas), 0.0)
        self.assertEqual(int(bundle.model.optimizer.iterations.value), 1)


if __name__ == "__main__":
    unittest.main()
