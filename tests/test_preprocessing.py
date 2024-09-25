import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))

import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from unittest.mock import patch
from scripts. preprocessing import calculate_missing_percentage, check_missing_values, outlier_box_plots

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [np.nan, 2, 3, 4, 5],
            'C': [1, 2, 3, 4, np.nan]
        })

    def test_calculate_missing_percentage(self):
        expected_output = "The dataset has 20.0% missing values.\n"
        with patch('sys.stdout', new=StringIO()) as fake_out:
            calculate_missing_percentage(self.df)
            self.assertEqual(fake_out.getvalue(), expected_output)

    def test_check_missing_values(self):
        result = check_missing_values(self.df)
        self.assertEqual(result.shape, (3, 3))
        self.assertEqual(result.index.tolist(), ['A', 'B', 'C'])
        self.assertEqual(result['Missing Values'].tolist(), [1, 1, 1])
        self.assertEqual(result['% of Total Values'].tolist(), [20.0, 20.0, 20.0])

    @patch('matplotlib.pyplot.show')
    def test_outlier_box_plots(self, mock_show):
        outlier_box_plots(self.df)
        self.assertEqual(mock_show.call_count, 3)

if __name__ == '__main__':
    unittest.main()