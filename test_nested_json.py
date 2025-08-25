import unittest
import pandas as pd
from flexible_column_detector import FlexibleColumnDetector

class TestFlexibleColumnDetector(unittest.TestCase):

    def setUp(self):
        self.detector = FlexibleColumnDetector()

    def test_with_flat_list(self):
        """Test with a simple, flat list of records."""
        data = [
            {'merchant': 'Vendor A', 'amount': 100},
            {'merchant': 'Vendor B', 'amount': 200}
        ]
        df = self.detector.normalize_to_dataframe(data)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df.columns), ['merchant', 'amount'])

    def test_with_nested_list(self):
        """Test with a list of records nested under a key."""
        data = {
            "some_metadata": "value",
            "transactions": [
                {'merchant': 'Vendor C', 'amount': 300},
                {'merchant': 'Vendor D', 'amount': 400}
            ]
        }
        df = self.detector.normalize_to_dataframe(data)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df.columns), ['merchant', 'amount'])

    def test_with_columnar_dict(self):
        """Test with a dictionary of lists (columnar format)."""
        data = {
            'merchant': ['Vendor E', 'Vendor F'],
            'amount': [500, 600]
        }
        df = self.detector.normalize_to_dataframe(data)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df.columns), ['merchant', 'amount'])

    def test_with_invalid_format(self):
        """Test with an unsupported data format."""
        data = {"key": "value"} # Not a list of records or dict of lists
        df = self.detector.normalize_to_dataframe(data)
        self.assertTrue(df.empty)

    def test_with_empty_list(self):
        """Test with an empty list."""
        df = self.detector.normalize_to_dataframe([])
        self.assertTrue(df.empty)

    def test_with_malformed_nested_list(self):
        """Test with a nested structure that doesn't contain a list of dicts."""
        data = {"transactions": "not a list"}
        df = self.detector.normalize_to_dataframe(data)
        self.assertTrue(df.empty)

if __name__ == '__main__':
    unittest.main()
