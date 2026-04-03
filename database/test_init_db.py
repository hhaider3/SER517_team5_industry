import unittest
from unittest.mock import MagicMock, patch


class TestInitDb(unittest.TestCase):

    @patch("init_db.chromadb.PersistentClient")
    @patch("init_db.OpenCLIPEmbeddingFunction")
    def test_db_initializes_successfully(self, mock_embedder, mock_client):
        mock_collection = MagicMock()
        mock_collection.name = "k12_education_images"
        mock_collection.count.return_value = 0
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        from init_db import main
        main()

    @patch("init_db.chromadb.PersistentClient")
    @patch("init_db.OpenCLIPEmbeddingFunction")
    def test_db_failure_raises_exception(self, mock_embedder, mock_client):
        mock_client.return_value.get_or_create_collection.side_effect = Exception("DB failed")

        from init_db import main
        with self.assertRaises(Exception):
            main()


if __name__ == "__main__":
    unittest.main()