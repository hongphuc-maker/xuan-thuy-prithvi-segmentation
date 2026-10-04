from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from prithvi_xtseg.permutation import align_probabilities, learn_class_permutation

if torch is not None:
    from prithvi_xtseg.dmi import (
        diagnostics_from_joint,
        exact_joint_upstream_gradient,
        joint_numerator_from_logits,
        validate_joint_for_gradient,
    )
    from prithvi_xtseg.morphology import LogitAdditiveClosing2d


class DirectDMICoreTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is verified in the Colab smoke test")
    def test_morphology_shape_parameters_and_gradient(self):
        layer = LogitAdditiveClosing2d(num_classes=11)
        inputs = torch.randn(2, 11, 8, 9, requires_grad=True)
        outputs = layer(inputs)
        self.assertEqual(outputs.shape, inputs.shape)
        self.assertEqual(layer.structuring_element.numel(), 99)
        outputs.square().mean().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertTrue(torch.isfinite(layer.structuring_element.grad).all())

    def test_train_only_assignment_recovers_channel_permutation(self):
        true_to_output = [2, 0, 1, 4, 3, 6, 5, 8, 7, 10, 9]
        confusion = np.ones((11, 11), dtype=np.int64)
        for true_class, output_channel in enumerate(true_to_output):
            confusion[true_class, output_channel] = 100 + true_class
        learned = learn_class_permutation(confusion)
        self.assertEqual(learned["true_class_to_output_channel"], true_to_output)
        raw = np.zeros((11, 3), dtype=np.float32)
        for true_class, output_channel in enumerate(true_to_output):
            raw[output_channel, true_class % 3] += 1.0
        aligned = align_probabilities(raw, learned)
        self.assertTrue(np.array_equal(aligned, raw[np.asarray(true_to_output)]))

    @unittest.skipIf(torch is None, "torch is verified in the Colab smoke test")
    def test_two_pass_upstream_matches_direct_autograd(self):
        torch.manual_seed(7)
        logits = torch.randn(2, 11, 2, 3, dtype=torch.float32, requires_grad=True)
        target = torch.tensor(
            [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 0]]],
            dtype=torch.int64,
        )
        numerator, count = joint_numerator_from_logits(logits, target)
        joint = numerator / float(count)
        loss, diagnostics = diagnostics_from_joint(joint, count, classes_present=11)
        validate_joint_for_gradient(joint, diagnostics)
        direct_gradient = torch.autograd.grad(loss, logits)[0]

        replay_logits = logits.detach().clone().requires_grad_(True)
        replay_numerator, replay_count = joint_numerator_from_logits(replay_logits, target)
        self.assertEqual(replay_count, count)
        upstream = exact_joint_upstream_gradient(joint.detach(), count)
        surrogate = (replay_numerator * upstream).sum()
        replay_gradient = torch.autograd.grad(surrogate, replay_logits)[0]
        self.assertTrue(torch.allclose(direct_gradient, replay_gradient, rtol=2e-4, atol=2e-5))


if __name__ == "__main__":
    unittest.main()
