# ztest.py

import pandas as pd
import math

from scipy.stats import norm

def ztest(pop_mean: float , pop_std: float, sample: pd.Series ):
    """
    ztest is used when :
    1. We need to test the value of mean, given that the population mean is known.
    2. When sample is large and population variance is known.

    Arguments:
    pop_mean: mean of the population
    pop_std: standard deviation of the population
    sample: sample dataframe
    """

    z_score = (sample.sample - pop_mean) / (pop_std / math.sqrt( sample.shape[0]))

    return z_score, norm.cdf(sample) 