import unittest


class BPEmbEmbeddingTest(unittest.TestCase):

    def test(self):
        embedding:BPEmbEmbedding = BPEmbEmbedding("de")
        self.assertEqual(embedding.raw_embedding().padding_idx, 100000)


if __name__ == "__main__":
    unittest.main()
