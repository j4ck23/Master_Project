import unittest
import pandas as pd
from LSTM import Model1

class Test(unittest.TestCase):
    @classmethod
    def test_data(self):
        """Gathers Data used in project"""
        self.df = pd.read_csv("Processed.csv")

    def test_non_empty(self):
        """Checks data is presnet"""
        self.assertNotEqual(len(self.df.index), 0)
        print("Data Present")

    def test_for_columns(self):
       """Checks required columns for porject are present"""
       self.assertIn("PM2.5", self.df.columns)
       self.assertIn("PM10", self.df.columns)
       self.assertIn("NO2", self.df.columns)
       self.assertIn("CO", self.df.columns)
       self.assertIn("O3", self.df.columns)
       self.assertIn("SO2", self.df.columns)
       self.assertIn("AQI", self.df.columns)
       self.assertIn("Time", self.df.columns)
       self.assertIn("Date", self.df.columns)
       print("All Required Columns Present")

    def test_not_null(self):
        """Check no null data is present in required columns"""
        self.assertTrue(self.df['PM2.5'].isnull().sum() == 0)
        self.assertTrue(self.df['PM10'].isnull().sum() == 0)
        self.assertTrue(self.df['NO2'].isnull().sum() == 0)
        self.assertTrue(self.df['CO'].isnull().sum() == 0)
        self.assertTrue(self.df['O3'].isnull().sum() == 0)
        self.assertTrue(self.df['SO2'].isnull().sum() == 0)
        self.assertTrue(self.df['AQI'].isnull().sum() == 0)
        self.assertTrue(self.df['Time'].isnull().sum() == 0)
        self.assertTrue(self.df['Date'].isnull().sum() == 0)
        print("No Null Data")

    def test_LSTM(self):
        """Checks the LSTM model can be built"""
        self.assertTrue(Model1())
        print("LSTM Built")

if __name__ == '__main__':
    unittest.main()