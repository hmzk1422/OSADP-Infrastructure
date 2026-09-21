from config.ingestion import config
import pandas as pd
import numpy as np

def macro_features(df, timeframe):
    """
    Returns higher timeframe (HTF) feature computation DF.
    - Runs reconstruct(df, timeframe) first before computing the features
    - returns dataframe reconstructed based on the timeframe
    """
    
    return 0

def micro_features(df):
    """
    Returns the lower timeframe (LTF) feature computation DF.
    - Runs reconstruct(df, timeframe) first before computing the feature.
    - Run macro_features first before running micro_features as this will be the final output
    """
    return 0