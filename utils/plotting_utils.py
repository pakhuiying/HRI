import geopandas as gpd
import utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
# import planning area

PLANNING_AREA = Data.import_planning_area()

def get_sgMainland_bbox(planningArea_shp = PLANNING_AREA):
    planningArea_shp_mainland = planningArea_shp[~planningArea_shp["PLN_AREA_N"].str.contains("ISLAND")]
    minx = planningArea_shp_mainland.bounds["minx"].min()
    miny = planningArea_shp_mainland.bounds["miny"].min()
    maxx = planningArea_shp_mainland.bounds["maxx"].max()
    maxy = planningArea_shp_mainland.bounds["maxy"].max()

    bbox = (minx, miny, maxx - 4e-4*maxx, maxy)
    return planningArea_shp_mainland, bbox

def correlation_plot(corr_matrix, title='Correlation Matrix',
                     figsize=(12,12),cmap='coolwarm',labelsize=20,save_fp=None):
    """
    Plot the correlation matrix using seaborn heatmap.
    """
    corr_array = corr_matrix.values
    # mask out lower triangle
    # mask =  np.tri(corr_array.shape[0], k=-1)
    mask =  np.triu(corr_array, k=0)
    corr_array = np.ma.array(corr_array, mask=mask)
    # set cmap
    cmap = cm.get_cmap(cmap, 10) # jet doesn't have white color
    cmap.set_bad('w') # default value is 'k'

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_array,cmap=cmap, aspect='auto')

    xlabels = corr_matrix.columns
    ylabels = corr_matrix.index

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(range(len(xlabels)), labels=xlabels,
                rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(ylabels)), labels=ylabels)

    # Loop over data dimensions and create text annotations.
    for i in range(len(ylabels)):
        for j in range(len(xlabels)):
            text = ax.text(j, i, f"{corr_array[i, j]:.3f}", 
                        ha="center", va="center", color="w",fontweight='bold')
    ax.spines[:].set_visible(False)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax_text_style = dict(horizontalalignment='center', verticalalignment='center',
                        fontsize=labelsize,weight='bold')
    ax.set_title(title,**ax_text_style)
    cbar = fig.colorbar(im, ax=ax, fraction=0.1)
    cbar.ax.tick_params(labelsize=labelsize)
    cbar.set_label(label="Correlation value",size=labelsize)
    fig.tight_layout()
    if save_fp is not None:
        fig.savefig(save_fp,bbox_inches='tight')
    
    plt.show()
    return corr_array

def plot_sensitivity_test(res, x_labels, figsize=(9, 4), save_fp = None):
    """
    plot_sesitivity_test by showing Sobol' results
    
    :param res: output from scipy's sobol_indices
    :param save_fp (str)
    """
    # Set the global random seed to ensure same output everytime
    np.random.seed(42)
    boot = res.bootstrap()

    x = list(range(len(x_labels)))
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    _ = axes[0].errorbar(
        x, res.first_order, fmt='o',
        yerr=[
            res.first_order - boot.first_order.confidence_interval.low,
            boot.first_order.confidence_interval.high - res.first_order
        ],
    )
    
    _ = axes[1].errorbar(
        x, res.total_order, fmt='o',
        yerr=[
            res.total_order - boot.total_order.confidence_interval.low,
            boot.total_order.confidence_interval.high - res.total_order
        ],
    )

    for ax in axes:
        ax.set_xlabel('HRI weights')
        ax.set_xticks(x, x_labels, rotation=45, ha='right')

    axes[0].set_ylabel("First order Sobol' indices")
    axes[1].set_ylabel("Total order Sobol' indices")

    plt.tight_layout()
    if save_fp is not None:
        plt.savefig(save_fp, bbox_inches = 'tight')
    plt.show()