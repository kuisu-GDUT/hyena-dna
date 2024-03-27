import unittest

import torch


def add(a, b):
    return a + b


class TestModels(unittest.TestCase):
    def test_hyena(self):
        from src.models.sequence.hyena import HyenaOperator

        model = HyenaOperator(
            d_model=512,
            l_max=1024,
            order=2,
            filter_order=64,
            num_heads=1,
            inner_factor=1,
            num_blocks=1,
            fused_bias_fc=False,
            outer_mixing=False,
            dropout=0.0,
            filter_dropout=0.0,
            filter_cls="hyena-filter",
            post_order_ffn=False,
            jit_filter=False,
            short_filter_order=3,
            activation="id",
            return_state=False,
        )

        input = torch.randn(size=(1, 1, 512))
        output = model(input)
        print(output.shape)

    def test_add2(self):
        self.assertEqual(add(3, 5), 10)


if __name__ == '__main__':
    unittest.main()
