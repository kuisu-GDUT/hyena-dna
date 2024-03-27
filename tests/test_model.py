import unittest

import torch


def add(a, b):
    return a + b


class TestModels(unittest.TestCase):
    def test_hyena(self):
        from src.models.sequence.hyena import HyenaOperator
        model = HyenaOperator(
            d_model=128,
            emb_dim=5,
            filter_order=64,
            short_filter_order=3,
            l_max=1026,
            modulate=True,
            w=10,
            lr=0.0006,
            wd=0.0,
            lr_pos_emb=0.0,
            bidirectional=False
        )
        print(model)
        input = torch.randn(size=(1, 1, 128))
        output = model(input)
        print(output.shape)

    def test_dna_embedding_model(self):
        from src.models.sequence.dna_embedding import DNAEmbeddingModel
        model = DNAEmbeddingModel(
            d_model=128,
            n_layer=2,
            d_inner=128 * 4,
            vocab_size=12
        )
        print(model)

    def test_conv_lm_head_mdoel(self):
        from src.models.sequence.long_conv_lm import ConvLMHeadModel

        model = ConvLMHeadModel(
            d_model=128,
            n_layer=2,
            d_inner=128 * 4,
            vocab_size=12
        )

        print(model)

    def test_long_conv(self):
        from src.models.sequence.long_conv import LongConv

        model = LongConv(
            d_model=128,
            n_layer=2,
            d_inner=128 * 4,
            vocab_size=12
        )
        print(model)
        pass

    def test_simple_model(self):
        from src.models.sequence.simple_lm import SimpleLMHeadModel

        model = SimpleLMHeadModel(
            d_model=128,
            n_layer=2,
            d_inner=128 * 4,
            vocab_size=12
        )
        print(model)
        pass

    def test_basic_cnn(self):
        from transformers import AutoConfig, AutoModelForCausalLM

        model_name = 'togethercomputer/evo-1-8k-base'

        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_config.use_cache = True

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=model_config,
            trust_remote_code=True,
        )

if __name__ == '__main__':
    unittest.main()
