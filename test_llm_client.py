import unittest
import json
from llm_client_v2 import LLMClient

class TestProductionGradeLLMClient(unittest.TestCase):

    def setUp(self):
        """Set up a mock LLMClient for each test."""
        # We explicitly use_mock=True to ensure we are testing the mock AI
        self.client = LLMClient(use_mock=True)
        print("\n--- Starting Test: {} ---".format(self.id()))

    def test_01_standard_operation(self):
        """Tests a batch of typical, clean transactions."""
        print("Scenario: Standard, clean data.")
        standard_data = [
            {'merchant': 'uber trip', 'amount': 25.50},
            {'merchant': 'GOOGLE *GSUITE', 'amount': 12.00},
            {'merchant': 'Amazon Web Services', 'amount': 150.75},
        ]
        results = self.client.process_transaction_batch(standard_data)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['standardized_vendor'], 'Uber')
        self.assertEqual(results[0]['category'], 'Travel & Transportation')
        self.assertEqual(results[1]['standardized_vendor'], 'Google Workspace')
        self.assertEqual(results[1]['category'], 'Software & Technology')
        print("✅ Standard operation test passed.")

    def test_02_messy_data_challenge(self):
        """Tests real-world, messy merchant strings."""
        print("Scenario: Messy, real-world data.")
        messy_data = [
            {'merchant': 'SQ *SQ *PATRIOT CAFE', 'amount': 15.25},
            {'merchant': 'AMZ*Amazon.com', 'amount': 49.99},
            {'merchant': 'apple.com/bill', 'amount': 9.99},
        ]
        results = self.client.process_transaction_batch(messy_data)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['standardized_vendor'], 'Square')
        self.assertEqual(results[0]['category'], 'Meals & Entertainment')
        self.assertEqual(results[1]['standardized_vendor'], 'Amazon')
        self.assertEqual(results[1]['category'], 'Office Supplies & Equipment')
        self.assertEqual(results[2]['standardized_vendor'], 'Apple')
        print("✅ Messy data challenge passed.")

    def test_03_empty_and_null_stress_test(self):
        """Tests handling of empty strings and None values."""
        print("Scenario: Empty and null data.")
        edge_case_data = [
            {'merchant': '', 'amount': 100.00},
            {'merchant': None, 'amount': 200.00},
            {}, # Entirely empty record
        ]
        results = self.client.process_transaction_batch(edge_case_data)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['standardized_vendor'], 'Unknown Vendor')
        self.assertEqual(results[0]['category'], 'Other')
        self.assertEqual(results[1]['standardized_vendor'], 'Unknown Vendor')
        self.assertEqual(results[1]['category'], 'Other')
        self.assertEqual(results[2]['standardized_vendor'], 'Unknown Vendor')
        self.assertEqual(results[2]['category'], 'Other')
        print("✅ Empty and null stress test passed.")
        
    def test_04_batch_consistency(self):
        """Ensures that a larger batch is processed correctly without errors."""
        print("Scenario: Larger batch processing (10 records).")
        large_batch = [
            {'merchant': 'uber trip', 'amount': 1}, {'merchant': 'GOOGLE *GSUITE', 'amount': 2},
            {'merchant': 'SQ *SQ *PATRIOT CAFE', 'amount': 3}, {'merchant': 'AMZ*Amazon.com', 'amount': 4},
            {'merchant': 'apple.com/bill', 'amount': 5}, {'merchant': '', 'amount': 6},
            {'merchant': None, 'amount': 7}, {'merchant': 'random place', 'amount': 8},
            {'merchant': 'Amazon', 'amount': 9}, {'merchant': 'uber eats', 'amount': 10}
        ]
        # The main goal here is to ensure it runs without crashing and returns the correct number of results.
        results = self.client.process_transaction_batch(large_batch)
        self.assertEqual(len(results), 10)
        print("✅ Batch consistency test passed.")

if __name__ == '__main__':
    print("--- Running Production Grade AI Core Test Suite ---")
    unittest.main()
    print("--- Test Suite Finished ---")

