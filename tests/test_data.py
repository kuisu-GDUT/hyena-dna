import unittest


class TestModels(unittest.TestCase):
    def test_character_tokenizer(self):
        from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer

        max_length = 6  # max len of seq grabbed
        padding_side = 'right'
        tokenizer = CharacterTokenizer(
            characters=['A', 'C', 'G', 'T', 'N'],
            model_max_length=max_length,
            add_special_tokens=False,
            padding_side=padding_side,
        )
        token = tokenizer(
            ["ACGTACGT", "AAA"]
        )
        print(token)

    def test_genomic_benchmark_dataset(self):
        from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
        from src.dataloaders.datasets.genomic_bench_dataset import GenomicBenchmarkDataset

        max_length = 300  # max len of seq grabbed
        use_padding = True
        dest_path = "/home/sukui/4_project/08_BioLLM/hyena-dna/data/genomic_benchmark"
        return_mask = True
        add_eos = True
        padding_side = 'right'

        tokenizer = CharacterTokenizer(
            characters=['A', 'C', 'G', 'T', 'N'],
            model_max_length=max_length,
            add_special_tokens=False,
            padding_side=padding_side,
        )

        ds = GenomicBenchmarkDataset(
            max_length=max_length,
            use_padding=use_padding,
            split='train',  #
            tokenizer=tokenizer,
            tokenizer_name='char',
            dest_path=dest_path,
            return_mask=return_mask,
            add_eos=add_eos,
        )

        it = iter(ds)
        elem = next(it)
        print('elem[0].shape', elem[0].shape)
        print(elem)


if __name__ == '__main__':
    unittest.main()
