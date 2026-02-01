import utils.data as Data
import pandas as pd
import geopandas as gpd
import numpy as np
import os
from glob import glob
from functools import reduce
import copy
from math import ceil
import matplotlib.pyplot as plt

class HRICalculation:
    def __init__(self, risk_df, weights_df, petal_weights):
        """
        Calculate HRI

        :param risk_df: dict of pd.DataFrame that describes normalised df values for each petal
        :param weights_df: pd.DataFrame that describes the weights for each variable
        :param petal_weights: pd.DataFrame that describes the weights for each petal
        """
        self.risk_df = risk_df
        self.weights_df = weights_df
        self.petal_weights = petal_weights

    def calculate_subindex(self, petal):
        """
        calculates capacity subindex - summation of c_i x w_i 
        Args:
            petal (str): petal name e.g. Sensitivity, Capacity
        """
        capacity_df = self.risk_df[petal]
        # select numeric columns only
        capacity_columns = capacity_df.select_dtypes(np.number).columns.to_list()
        # get number of variables
        n_vars = len(capacity_columns)
        # make copy of weights_df
        weights_df = copy.deepcopy(self.weights_df)
        # make variable the index
        weights_df = weights_df.set_index(['Variable'])
        # convert weights to values between 0 to 1
        capacity_weights = weights_df.loc[capacity_columns,"Weight"].values.reshape((n_vars,1))/100
        # convert capacity_df to matrix
        capacity_val = capacity_df[capacity_columns].values
        assert capacity_val.shape[1] == capacity_weights.shape[0], "shapes must tally for matrix dot product"
        # print(capacity_val.shape, capacity_weights.shape)
        return pd.DataFrame(data=np.dot(capacity_val, capacity_weights),
                            index = capacity_df["PLN_AREA_N"], columns=[petal])
    
    def calculate_vulnerability_subindex(self):
        """calculates sensitivity and capacity subindex """
        vul_SI = dict()
        for v in ["Sensitivity", "Capacity"]:
            SI = self.calculate_subindex(petal=v)
            vul_SI[v] = SI
        return vul_SI
    
    def get_petal_weights(self):
        """ get H, E, V weights
        Returns:
            tuple: H, E, V weights
        """
        # get petal weights
        petal_weight = self.petal_weights.set_index(["Petal"])
        # convert weights to values between 0 to 1
        petal_weight = petal_weight["Weight"]/100
        # get individual petals' weights
        hazard = petal_weight.loc["Hazard"]
        exposure = petal_weight.loc["Exposure"]
        vulnerability = petal_weight.loc["Vulnerability"]
        return hazard, exposure, vulnerability
    
    def HRI_formula(self, H, E, S, C):
        """
        HRI_formula

        :param H: Hazard array
        :param E: Exposure array
        :param S: Sensitivity array
        :param C: Capacity array
        """
        # calculate V
        V = S/C
        hazard, exposure, vulnerability = self.get_petal_weights()
        return (H**hazard)*(E**exposure)*(V**vulnerability)
    
    
    def get_HRI_df(self):
        """get unweighted HRI for each petal (without W_H, W_E, W_V)"""
        vul_SI = self.calculate_vulnerability_subindex()
        # add data to SI
        for p in ["Hazard","Exposure"]:
            df = self.risk_df[p].set_index(["PLN_AREA_N"])
            df.columns = [p]
            vul_SI[p] = df
        # concat all petals to one df
        HRI_df = reduce(lambda x, y: pd.merge(x, y, 
                                              how='outer',
                                              right_index=True,
                                              left_index=True), list(vul_SI.values()))
        # add it as an attribute to class
        self.HRI_df = HRI_df
        return HRI_df
    
    def calculate_HRI_verbose(self):
        """ 
        obtain the granular scores for H, E, S, C for each planning area
        """
        HRI_df = self.get_HRI_df()
        HRI_df["Vulnerability"] = HRI_df["Sensitivity"]/HRI_df["Capacity"]
        nrows = len(HRI_df)
        hazard, exposure, vulnerability = self.get_petal_weights()
        petal_weights = np.array([[hazard, exposure, vulnerability]])
        # repeat along rows
        petal_weights = np.tile(petal_weights,(nrows,1))
        # element wise power
        HRI_df_power = np.power(HRI_df[["Hazard","Exposure","Vulnerability"]].values,
                          petal_weights)
        return pd.DataFrame(data=HRI_df_power, index=HRI_df.index, columns=["Hazard","Exposure","Vulnerability"])
        
    def plot_HRI_breakdown(self, HRI_df_verbose=None,
                           ax=None,nrows=5,
                           colors=['#fdbf6f','#a6cee3','#fff7bc'],
                           save_fp = None,
                           **kwargs):
        """
        Calculated based on rescaled weighted HRI
        Args:
            HRI_df (pd.DataFrame)
            ax (mpl.Axes): if None, plot on new figure, otherwise plot on supplied axes
            **kwargs: other keyword arguments for ax.pie
        """
        if HRI_df_verbose is None:
            HRI_df_verbose = self.calculate_HRI_verbose()
        
        # remove NA rows
        HRI_df_verbose = HRI_df_verbose.dropna()
        nObs = len(HRI_df_verbose)
        if ax is None:
            fig, axes = plt.subplots(nrows=nrows, ncols=ceil(nObs/nrows),
                                     figsize=(12,10))
        
        # labels = ["Hazard", "Exposure", "Vulnerability"]
        labels = ["H","E","V"]
        # maximum values in Hazard, Exposure and Vulnerability
        baseline_max = HRI_df_verbose.max(axis=0).values
        for (row_ix, row), ax in zip(HRI_df_verbose.iterrows(), axes.flatten()):
            # print(row_ix, row.values)
            ax.pie(row.values/baseline_max, labels=labels, colors=colors, **kwargs)
            ax.set_title(row_ix)
        
        if save_fp is not None:
            plt.savefig(save_fp, bbox_inches = 'tight')
        if ax is None:
            plt.show()
        
        # rescaled HRI df verbose by using baselinemax
        rescaled_HRI_df_verbose = HRI_df_verbose.values/baseline_max
        return pd.DataFrame(data=rescaled_HRI_df_verbose, index=HRI_df_verbose.index, columns=HRI_df_verbose.columns)


    def calculate_HRI(self):
        """
        calculate HRI
        The higher the HRI, the higher the overall risk
        if H increase, HRI increase
        if E increase, HRI increase
        if V increase because S>C, HRI increase
        Note that HRI is an unbounded score
        Returns:
            pd.DataFrame with just one HRI column
        """
        HRI_df = self.get_HRI_df()
        # calculate HRI
        HRI = self.HRI_formula(H= HRI_df["Hazard"], E=HRI_df["Exposure"], 
                         S=HRI_df["Sensitivity"], C=HRI_df["Capacity"])
        # because actual HRI only ranges from 0 to 2, we will need to do some rescaling
        # assume that H, E, may take on 100 and S/C may take on 100/100=1
        baseline_max = np.array([100])
        baseline_HRI = self.HRI_formula(H= baseline_max, E=baseline_max, 
                         S=baseline_max, C=baseline_max)
        return pd.DataFrame(HRI/baseline_HRI*100, columns=["rescaled_HRI"])
    
    def plot_HRI(self, HRI_df=None, planningArea=Data.PLANNING_AREA,
                 ax=None,**kwargs):
        """
        Args:
            HRI_df (pd.DataFrame)
            ax (mpl.Axes): if None, plot on new figure, otherwise plot on supplied axes
            **kwargs: other keyword arguments for gpd.plot
        """
        if HRI_df is None:
            HRI_df = self.calculate_HRI()
        HRI_df = HRI_df.reset_index()
        # join HRI with planning area
        HRI_gdf = planningArea.merge(HRI_df, how="inner", on="PLN_AREA_N")
        HRI_gdf.plot(ax=ax, column="rescaled_HRI",**kwargs)


