import unittest
from stats import ztest
import pandas as pd
from  pathlib import Path



class TestStatsMethods( unittest.TestCase ):
    df = pd.read_csv('Smoker_R.csv')
    df.head()

    def test_zscore(self):
        # z_score, pval = 
        # self.assertEqual(  )
        pass