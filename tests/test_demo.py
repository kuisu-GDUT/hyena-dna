import unittest


def add(a, b):
    return a + b


class SimpleTest(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(3, 7), 10)

    def test_add2(self):
        self.assertEqual(add(3, 5), 10)

    def test_torch(self):
        import torch

        print(torch.cuda.is_available())



if __name__ == '__main__':
    unittest.main()
