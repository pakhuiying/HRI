import pandas as pd
import geopandas as gpd
import numpy as np
import os
from glob import glob
from functools import reduce
import copy
from utils import data as Data

# planning area data
PLANNING_AREA_FP = r"C:\Users\hypak\OneDrive - Singapore Management University\Documents\Data\SG_Map\planningArea.shp"
# vulnreability weights fp
DAYTIME_WEIGHTS_FP = r"Weights_Thresholds\DayTime_Weights.csv"
NIGHTTIME_WEIGHTS_FP = r"Weights_Thresholds\NightTime_Weights.csv"
PETAL_WEIGHTS_FP = r"Weights_Thresholds\Petal_Weights.csv"
# Thresholds fp
THRESHOLDS_FP = r"Weights_Thresholds\Thresholds.csv"

# import planning area
def import_planning_area(fp = PLANNING_AREA_FP):
    return gpd.read_file(fp)

PLANNING_AREA = import_planning_area()
# # import data
# def import_data_fps(data_dir = r"Risk_Data"):

select_columns = {
    # capacity
    'DayTime_ACHouseholds': 'ACHouseholdsProp',
    'DayTime_Clinic': 'Clinic/DaytimePopulationPerPlanningArea',
    'DayTime_Household median income': 'Median Income Ratio (Straits Times)',
    'DayTime_PublicTransportFactor':'proportionBuildingFootprintAccessibileToPT',
    'DayTime_ShadeFactor':'ShadeFactor',
    'DayTime_minMean_travelTime_Hospital': 'min_travel_time', # invert this
    # exposure
    'DayTime_Pop':'TotalDayTimePop',
    # hazard
    'DayTime_WBGT_hottest':'WBGT_hottest',
    # sensitivity
    'DayTime_FunctionalDisability':'FunctionallyDisabledHouseholds:HouseholdPerPlanningArea',
    'DayTime_Social_Isolation': 'SociallyIsolatedHouseholds:HouseholdsPerPlanningArea',
    'DayTime_Vulnerable_Children':'Age<14:PlanningAreaPopulation',
    'DayTime_Vulnerable_Elderly':'Age>60:PlanningAreaPopulation',
    # capacity
    'NightTime_ACHouseholds':'ACHouseholdsProp',
    'NightTime_Clinic':'Clinic/NighttimePopulationPerPlanningArea',
    'NightTime_Household median income':'Median Income Ratio (Straits Times)',
    'NightTime_PublicTransportFactor':'proportionBuildingFootprintAccessibileToPT',
    'NightTime_ShadeFactor':'ShadeFactor',
    'NightTime_minMean_travelTime_Hospital': 'min_travel_time', # invert this
    # exposure
    'NightTime_Pop':'ResidentPopTotalAll',
    # hazard
    'NightTime_TA_coolest': 'TA_coolest',
    # sensitivity
    'NightTime_FunctionalDisability':'FunctionallyDisabledHouseholds:HouseholdPerPlanningArea',
    'NightTime_Social_Isolation': 'SociallyIsolatedHouseholds:HouseholdsPerPlanningArea',
    'NightTime_Vulnerable_Children': 'Age<14:PlanningAreaNighttimePopulation',
    'NightTime_Vulnerable_Elderly': 'Age>60:PlanningAreaNighttimePopulation'
}

rename_columns = {
    'ACHouseholdsProp': 'AC Ownership',
    'Clinic/DaytimePopulationPerPlanningArea': 'Minor Medical Treatment Availability',
    'Median Income Ratio (Straits Times)': 'Financial Capacity',
    'proportionBuildingFootprintAccessibileToPT': 'Public Transport Accessibility',
    'ShadeFactor': 'Shade Potential',
    'TotalDayTimePop': 'Daytime Population',
    'WBGT_hottest': 'Hottest WBGT',
    'FunctionallyDisabledHouseholds:HouseholdPerPlanningArea': 'Functionally Disabled',
    'SociallyIsolatedHouseholds:HouseholdsPerPlanningArea': 'Socially Isolated',
    'Age<14:PlanningAreaPopulation': 'Prepubescent Population',
    'Age>60:PlanningAreaPopulation': 'Elderly Population',
    'Clinic/NighttimePopulationPerPlanningArea': 'Minor Medical Treatment Availability',
    'ResidentPopTotalAll': 'Nighttime Population',
    'TA_coolest': 'Min TA',
    'Age<14:PlanningAreaNighttimePopulation': 'Prepubescent Population',
    'Age>60:PlanningAreaNighttimePopulation': 'Elderly Population',
    'min_travel_time': 'Comprehensive Medical Treatment Accessibility'
}

def import_data(select_columns=select_columns,rename_columns=rename_columns,data_dir = r"Risk_Data"):
    """
    import_data and store it  in a nested dict, where first level keys: represent Day/Night Time HRI,
    second level keys represent petal names, third
    
    :param select_columns: dictionary describing which column names to select from each dataframe
    :param data_dir: directory of where the risk data is stored. Find all data recursively
    """
    fps = glob(os.path.join(data_dir,"**/*.csv"), recursive=True)
    # remove the Risk_Data folder from the fps
    fps = [os.path.join(*(fp.split(os.path.sep)[1:])) for fp in fps]
    # initialise empty dict to store data
    risk_data = dict()
    for fp in fps:
        petal_name = os.path.dirname(fp)
        fp_name = os.path.splitext(os.path.basename(fp))[0]
        # read file
        df = pd.read_csv(os.path.join(data_dir,fp))
        try:
            # subset columns
            keep_columns = ["PLN_AREA_N",select_columns[fp_name]]
            df = df[keep_columns]
            # rename columns
            df = df.rename(columns={select_columns[fp_name]: rename_columns[select_columns[fp_name]]})
            # capitalise all PLN_AREA_N
            df['PLN_AREA_N'] = df['PLN_AREA_N'].str.strip().str.upper()
            # cast everything to numeric
            df[df.columns[-1]] = pd.to_numeric(df[df.columns[-1]], errors='coerce').astype(np.float64)
            HRI_type, var_name = fp_name.split('_', maxsplit=1)
            if HRI_type not in risk_data:
                risk_data[HRI_type] = {petal_name: dict()}
            if petal_name not in risk_data[HRI_type]:
                risk_data[HRI_type][petal_name] = {var_name: df}
            risk_data[HRI_type][petal_name][var_name] = df
        except Exception as e:
            print(e,fp_name)

    return risk_data


def consolidate_df(imported_data_dict, planningArea=PLANNING_AREA, normalise=None):
    """  
    Args:
        imported_data_dict (dict): output from import_data
        planning_area (gpd.GeoDataFrame)
        normalise (None or perc or min_max or max_scaling): whether to convert all numeric columns to percentile values (not encouraged because not comparable across time)
    """
    store_data = dict()
    for HRI_time, HRI_time_dict in imported_data_dict.items():
        if HRI_time not in store_data:
            store_data[HRI_time] = dict()
        for petal, petal_dict in HRI_time_dict.items():
            if petal not in store_data[HRI_time]:
                store_data[HRI_time][petal] = dict()
            # merge all columns
            petal_df = reduce(lambda x, y: pd.merge(x, y, on = 'PLN_AREA_N',how='outer'), list(petal_dict.values()))
            # select rows corresponding to planning areas
            petal_df = petal_df[petal_df['PLN_AREA_N'].isin(planningArea['PLN_AREA_N'].to_list())]
            if normalise is not None:
                # get numeric columns
                numeric_columns = petal_df.select_dtypes(np.number).columns
                if normalise == "perc":
                    # normalise columns using percentile rank
                    # disadvantage: HRI will not be comparable across time
                    petal_df.loc[:,numeric_columns] = petal_df[numeric_columns].apply(lambda x: x.rank(pct=True)*100)
                elif normalise == "min_max":
                    # disadvantage: value with the min value will turn into 0 which will cause complication e.g.
                    # if exposure min value is 100, after normalisation, it will transform into 0, and cause HRI to be 0 also, but min exposure is not 0 exposure!
                    petal_df.loc[:,numeric_columns] = petal_df[numeric_columns].apply(lambda x: ((x-x.min())/(x.max()-x.min()))*100)
                else:
                    petal_df.loc[:,numeric_columns] = petal_df[numeric_columns].apply(lambda x: (x/x.max())*100)
            # print(len(petal_df))
            store_data[HRI_time][petal] = petal_df
    return store_data

def consolidate_petals(risk_df):
    """consolidates all the sensitivity, capacity, exposure, hazard into one df
    Args:
        risk_df (dict)
    """
    consolidated_df = {i: None for i in list(risk_df)}
    for HRI_time, HRI_time_dict in risk_df.items():
        petal_df = reduce(lambda x, y: pd.merge(x, y, on = 'PLN_AREA_N',how='outer'), list(HRI_time_dict.values()))
        consolidated_df[HRI_time] = petal_df
    return consolidated_df

def normalisation_imputation_df(risk_imported_df, normalise="max_scaling"):
    # initialise df for storing normalised and imputed data
    risk_norm_df = Data.consolidate_df(risk_imported_df, normalise=normalise)
    risk_norm = copy.deepcopy(risk_norm_df)
    risk_norm_imputed = copy.deepcopy(risk_norm_df)
    # invert 'Comprehensive Medical Treatment Accessibility' because higher travel time means lower capacity
    for HRI_time, HRI_time_dict in risk_norm_df.items():
        for petal, petal_df in HRI_time_dict.items():
            if petal == "Capacity":
                # flip relationship
                risk_norm[HRI_time][petal]['Comprehensive Medical Treatment Accessibility'] = 100 - petal_df['Comprehensive Medical Treatment Accessibility']
                risk_norm_imputed[HRI_time][petal]['Comprehensive Medical Treatment Accessibility'] = 100 - petal_df['Comprehensive Medical Treatment Accessibility']
                # print(f"{HRI_time}: {risk_norm[HRI_time][petal]['Comprehensive Medical Treatment Accessibility'].values[0]}")
            # then impute NAs with 0th normentile
            risk_norm_imputed[HRI_time][petal] = risk_norm_imputed[HRI_time][petal].fillna(0)

    return {"raw_df":risk_norm_df, "norm_df":risk_norm, "norm_df_imputed": risk_norm_imputed}
