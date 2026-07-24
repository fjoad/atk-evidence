"""Focused tests for the bounded FC-VAE first-step diagnostic."""

from __future__ import annotations

import unittest

import torch

from diagnose_fc_vae_first_step_ddp import (
    gradient_diagnostics,
    nonfinite_state_diagnostics,
)


class DiagnosticSummaryTests(unittest.TestCase):
    def test_finite_gradient_can_overflow_adam_square(self) -> None:
        summary = gradient_diagnostics(
            [
                ("ordinary/kernel", torch.tensor([1.0, -2.0])),
                ("z_log_var/kernel", torch.tensor([2.0e19])),
            ]
        )
        self.assertEqual(summary["gradient_nonfinite_elements"], 0)
        self.assertEqual(summary["max_gradient_path"], "z_log_var/kernel")
        self.assertEqual(summary["gradient_square_nonfinite_elements"], 1)
        self.assertEqual(
            summary["gradient_square_nonfinite_tensors"],
            [{"path": "z_log_var/kernel", "elements": 1}],
        )

    def test_parameter_and_optimizer_records_remain_separable(self) -> None:
        parameters = nonfinite_state_diagnostics(
            [("decoder/kernel", torch.tensor([1.0, 2.0]))]
        )
        optimizer = nonfinite_state_diagnostics(
            [
                ("adam/z_log_var_kernel_velocity", torch.tensor([float("inf")])),
                ("adam/z_mean_kernel_momentum", torch.tensor([float("nan")])),
            ]
        )
        self.assertEqual(parameters["nonfinite_tensor_count"], 0)
        self.assertEqual(optimizer["nonfinite_tensor_count"], 2)
        self.assertEqual(optimizer["nonfinite_element_count"], 2)
        self.assertEqual(
            [item["path"] for item in optimizer["nonfinite_tensors"]],
            ["adam/z_log_var_kernel_velocity", "adam/z_mean_kernel_momentum"],
        )


if __name__ == "__main__":
    unittest.main()
