import pandas as pd
import geopandas as gpd
import numpy as np
import os
from glob import glob
from functools import reduce
import copy

class GetThresholds:
    def __init__(self, thresholds, risk_df, invert_vars = ['Comprehensive Medical Treatment Accessibility']):
        """
        Get thresholds mapped to normalised df
        
        :param self: 
        :param thresholds: pd.DataFrame. either daytime or nighttime threshold
        :param risk_df: pd.DataFrame that describes all the raw unscaled values (consolidated df of all the variables) 
        """
        self.thresholds = thresholds
        self.risk_df = risk_df
        self.invert_vars = invert_vars

    def min_max_threshold(self, row):
        return_dict = dict() # to store data
        var = row['Variable']
        lower_break = row['lower_break']
        upper_break = row['upper_break']
        try:
            min_val = self.risk_df[var].min()
            max_val = self.risk_df[var].max()
            # print(var)
            return_dict['lower_break'] = (lower_break-min_val)/(max_val - min_val)*100
            return_dict['upper_break'] = (upper_break-min_val)/(max_val - min_val)*100
            return_dict['Variable'] = var
            if var in self.invert_vars:
                return_dict['lower_break'] = 100 - return_dict['lower_break']
                return_dict['upper_break'] = 100 - return_dict['upper_break']
        except:
            print(var)
            pass
        # print(var)
        return pd.Series(return_dict, index = list(return_dict))

    def max_scaling_threshold(self, row):
        """ Note that for future HRI, the max_val must be 2019's max_value, otherwise your future HRI will not be comparable against previous HRI"""
        return_dict = dict() # to store data
        var = row['Variable']
        lower_break = row['lower_break']
        upper_break = row['upper_break']
        try:
            max_val = self.risk_df[var].max()
            # print(var)
            return_dict['lower_break'] = (lower_break)/(max_val)*100
            return_dict['upper_break'] = (upper_break)/(max_val)*100
            return_dict['Variable'] = var
            if var in self.invert_vars:
                return_dict['lower_break'] = 100 - return_dict['lower_break']
                return_dict['upper_break'] = 100 - return_dict['upper_break']
        except:
            print(var)
            pass
        # print(var)
        return pd.Series(return_dict, index = list(return_dict))

    def to_percentile_threshold(self, row):
        """
        use this function to transform from raw threshold values to percentiles thresholds
        e.g. convert 0.1, 0.2 threshold values to 50,60 percentile values
        """
        return_dict = dict() # to store data
        var = row['Variable']
        lower_break = row['lower_break']
        upper_break = row['upper_break']
        try:
            xVal = np.sort(self.risk_df[var].values)
            yVal = np.sort(self.risk_df[var].rank(pct=True).values*100) #risk_df[var].apply(lambda x: x.rank(pct=True)*100)
            ynew = np.interp(np.array([lower_break,upper_break]), xVal, yVal,
                             left=0, right=100) # value to return if new x val exceeds xVal range
            return_dict['lower_break'] = ynew[0]
            return_dict['upper_break'] = ynew[1]
            return_dict['Variable'] = var
            if var in self.invert_vars:
                return_dict['lower_break'] = np.max(yVal) - return_dict['lower_break']
                return_dict['upper_break'] = np.max(yVal) - return_dict['upper_break']
        except:
            print(var)
            pass
        
        return pd.Series(return_dict, index = list(return_dict))

    def from_percentile_threshold(self, row):
        """use this function to transform percentile to another normalised values"""
        return_dict = dict() # to store data
        var = row['Variable']
        lower_break = row['lower_break']
        upper_break = row['upper_break']
        try:
            # use percentile ignoring NaN
            return_dict['lower_break'] = np.nanpercentile(self.risk_df[var], lower_break)
            return_dict['upper_break'] = np.nanpercentile(self.risk_df[var], upper_break)
            return_dict['Variable'] = var
        except:
            print(var)
            pass
        
        return pd.Series(return_dict, index = list(return_dict))
    
    def mapping_thresholds_specified(self, thresholds, normalise="max_scaling"):
        """
        given specified thresholds, map it to the normalised transform value
        Args:
            thresholds (pd.DataFrame): df of thresholds
            normalise (str): max_scaling or min_max, perc
        Returns:
            pd.DataFrame: df that provides the remapped threshold values
        """
        # use linear interpolation for percentile for perc threshold type
        # use normalisation formula for  guideline-based threshold type
        thresholds_specified_vars = thresholds[thresholds['Threshold_type']=="guideline-based"]
        perc_vars = thresholds[thresholds['Threshold_type']=="perc"]
        if normalise == "perc":
            # convert guideline based raw threshold values to percentile values
            guideline_threshold = thresholds_specified_vars.apply(lambda x: self.to_percentile_threshold(x),axis=1)
            # select common column names prior to concat dfs
            perc_threshold = perc_vars[guideline_threshold.columns.to_list()]
 
        elif normalise == "min_max":
            # convert guideline based raw threshold values to normalised min_max values
            guideline_threshold = thresholds_specified_vars.apply(lambda x: self.min_max_threshold(x),axis=1)
            # convert percentile values to the raw values corresponding to 25th and 75th percentile
            perc_threshold = perc_vars.apply(lambda x: self.from_percentile_threshold(x),axis=1)
            # transform raw values (derived from percentiles) using min-max function so they all correspond to the normalised threshold values
            perc_threshold = perc_threshold.apply(lambda x: self.min_max_threshold(x),axis=1)
        else:
            # convert guideline based raw threshold values to normalised min_max values
            guideline_threshold = thresholds_specified_vars.apply(lambda x: self.max_scaling_threshold(x),axis=1)
            # convert percentile values to the raw values corresponding to 25th and 75th percentile
            perc_threshold = perc_vars.apply(lambda x: self.from_percentile_threshold(x),axis=1)
            # transform raw values (derived from percentiles) using min-max function so they all correspond to the normalised threshold values
            perc_threshold = perc_threshold.apply(lambda x: self.max_scaling_threshold(x),axis=1)
        
        concat_thresholds = pd.concat([guideline_threshold, perc_threshold])
        return concat_thresholds.dropna()
    
def convert_threshold_to_HRI_df(threshold_df, HRI_df):
    """
    Convert threshold_df to a format similar to the input df for HRICalulation
    :param threshold_df: pd.DataFrame describing the Variables and the lower and upper break thresholds
    :param HRI_df: dict of dataframe where keys are petals and values are the df describing the normalised values for each variable
    """
    threshold = copy.deepcopy(threshold_df)
    threshold = threshold.set_index('Variable').T # transpose df
    # threshold = threshold.rename(index={"Variable":"PLN_AREA_N"}) 
    threshold.index.names = ["PLN_AREA_N"]
    threshold_dict = dict()
    for petal, df in HRI_df.items():
        columns_to_select = HRI_df[petal].select_dtypes(np.number).columns.to_list()
        threshold_dict[petal] = threshold[columns_to_select].reset_index()#.drop(columns=["Variable"])

    return threshold_dict