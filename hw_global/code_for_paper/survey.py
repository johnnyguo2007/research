import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os

SUMMARY_DIR = '/Trex/case_results/i.e215.I2000Clm50SpGs.hw_production.05/research_results/summary'
FIGURE_OUTPUT_DIR = '/home/jguo/tmp/output2'
TWOKG_FEATHER_PATH = os.path.join(SUMMARY_DIR, 'twoKG_local_hour_adjusted_variables_HW98.feather')

def normalize_longitude(lons: np.ndarray) -> np.ndarray:
    return ((lons + 180) % 360) - 180

def create_base_map(ax):
    """Creates a base map with common settings."""
    m = Basemap(projection='cyl', lon_0=0, ax=ax, fix_aspect=False,
                llcrnrlat=-44.94133, urcrnrlat=65.12386)
    m.drawcoastlines(color='0.15', linewidth=0.5, zorder=3)
    m.drawcountries(linewidth=0.1)
    m.fillcontinents(color='white', lake_color='lightcyan')
    m.drawmapboundary(fill_color='lightcyan')
    m.drawparallels(np.arange(-90., 91., 30.), labels=[1,0,0,0], fontsize=10)
    m.drawmeridians(np.arange(-180., 181., 60.), labels=[0,0,0,1], fontsize=10)
    return m

def plot_kg_differences():
    """Plots global maps showing KG classifications."""
    # Load data
    df = pd.read_feather(TWOKG_FEATHER_PATH)
    
    # Fill NaN values and get all unique locations
    df['KGMainGroup'] = df['KGMainGroup'].fillna('Unknown')
    df['KGMajorClass'] = df['KGMajorClass'].fillna('Unknown')
    all_locations = df.groupby('location_ID').first().reset_index()
    
    # Create figure with four subplots
    fig = plt.figure(figsize=(24, 32))  # Doubled height for 4 plots
    
    # Create a unified color mapping for all climate types
    all_climate_types = sorted(set(list(all_locations['KGMainGroup'].unique()) + 
                                 list(all_locations['KGMajorClass'].unique())))
    
    # Ensure 'Unknown' is included and at the end of the list
    if 'Unknown' in all_climate_types:
        all_climate_types.remove('Unknown')
    all_climate_types = sorted(all_climate_types) + ['Unknown']
    
    # Create color dictionary for all types
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_climate_types)))
    color_dict = dict(zip(all_climate_types, colors))
    
    # Add special color for matching classifications
    color_dict['Matching'] = (0.7, 0.7, 0.7, 1.0)  # Gray color for matching points
    
    # First plot - All KGMainGroup classifications
    ax1 = plt.subplot(411)
    m1 = create_base_map(ax1)
    
    # Prepare coordinates
    normalized_lons = normalize_longitude(all_locations['lon'].values)
    x, y = m1(normalized_lons, all_locations['lat'].values)
    
    # Plot all points for KGMainGroup
    for kg_type in all_climate_types:
        mask = all_locations['KGMainGroup'] == kg_type
        if mask.any():
            m1.scatter(x[mask], y[mask], 
                      color=color_dict[kg_type],
                      label=kg_type,
                      marker='o', 
                      s=50, 
                      alpha=0.75, 
                      zorder=4)
    
    ax1.set_title('All Locations - KGMainGroup Classification', fontsize=14, pad=20)
    leg1 = ax1.legend(title='KGMainGroup',
                     title_fontsize=12,
                     fontsize=10,
                     loc='center left',
                     bbox_to_anchor=(1.02, 0.5),
                     frameon=True,
                     edgecolor='black',
                     fancybox=True,
                     shadow=True)
    
    # Second plot - All KGMajorClass classifications
    ax2 = plt.subplot(412)
    m2 = create_base_map(ax2)
    
    # Plot all points for KGMajorClass
    for kg_type in all_climate_types:
        mask = all_locations['KGMajorClass'] == kg_type
        if mask.any():
            m2.scatter(x[mask], y[mask], 
                      color=color_dict[kg_type],
                      label=kg_type,
                      marker='o', 
                      s=50, 
                      alpha=0.75, 
                      zorder=4)
    
    ax2.set_title('All Locations - KGMajorClass Classification', fontsize=14, pad=20)
    leg2 = ax2.legend(title='KGMajorClass',
                     title_fontsize=12,
                     fontsize=10,
                     loc='center left',
                     bbox_to_anchor=(1.02, 0.5),
                     frameon=True,
                     edgecolor='black',
                     fancybox=True,
                     shadow=True)
    
    # Third map - KGMainGroup with comparison
    ax3 = plt.subplot(413)
    m3 = create_base_map(ax3)
    
    # First plot matching points
    matching_mask = all_locations['KGMainGroup'] == all_locations['KGMajorClass']
    if matching_mask.any():
        m3.scatter(x[matching_mask], y[matching_mask],
                  color=color_dict['Matching'],
                  label='Matching Classifications',
                  marker='o',
                  s=50,
                  alpha=0.75,
                  zorder=3)
    
    # Then plot differing points for KGMainGroup
    for kg_type in all_climate_types:
        mask = (all_locations['KGMainGroup'] == kg_type) & ~matching_mask
        if mask.any():
            m3.scatter(x[mask], y[mask], 
                      color=color_dict[kg_type],
                      label=kg_type,
                      marker='o', 
                      s=50, 
                      alpha=0.75, 
                      zorder=4)
    
    ax3.set_title('KGMainGroup Classification (Highlighting Differences)', fontsize=14, pad=20)
    leg3 = ax3.legend(title='KGMainGroup',
                     title_fontsize=12,
                     fontsize=10,
                     loc='center left',
                     bbox_to_anchor=(1.02, 0.5),
                     frameon=True,
                     edgecolor='black',
                     fancybox=True,
                     shadow=True)
    
    # Fourth map - KGMajorClass with comparison
    ax4 = plt.subplot(414)
    m4 = create_base_map(ax4)
    
    # First plot matching points
    if matching_mask.any():
        m4.scatter(x[matching_mask], y[matching_mask],
                  color=color_dict['Matching'],
                  label='Matching Classifications',
                  marker='o',
                  s=50,
                  alpha=0.75,
                  zorder=3)
    
    # Then plot differing points for KGMajorClass
    for kg_type in all_climate_types:
        mask = (all_locations['KGMajorClass'] == kg_type) & ~matching_mask
        if mask.any():
            m4.scatter(x[mask], y[mask], 
                      color=color_dict[kg_type],
                      label=kg_type,
                      marker='o', 
                      s=50, 
                      alpha=0.75, 
                      zorder=4)
    
    ax4.set_title('KGMajorClass Classification (Highlighting Differences)', fontsize=14, pad=20)
    leg4 = ax4.legend(title='KGMajorClass',
                     title_fontsize=12,
                     fontsize=10,
                     loc='center left',
                     bbox_to_anchor=(1.02, 0.5),
                     frameon=True,
                     edgecolor='black',
                     fancybox=True,
                     shadow=True)
    
    # Add overall title
    plt.suptitle('Global Distribution of Köppen-Geiger Classifications', fontsize=16, y=0.95)
    
    # Adjust layout to prevent legend cutoff
    plt.subplots_adjust(right=0.85, hspace=0.3)
    
    # Save figure with extra space for legends
    plt.savefig(os.path.join(FIGURE_OUTPUT_DIR, 'kg_classification_all.png'), 
                bbox_inches='tight',
                dpi=300,
                pad_inches=0.5)
    plt.close()
    
    # Print summary statistics
    print(f"Total number of locations: {len(all_locations)}")
    print(f"Number of locations with matching classifications: {matching_mask.sum()}")
    print(f"Number of locations with different classifications: {(~matching_mask).sum()}")
    
    print("\nKGMainGroup distribution:")
    print(all_locations['KGMainGroup'].value_counts())
    print("\nKGMajorClass distribution:")
    print(all_locations['KGMajorClass'].value_counts())
    
    # Print the mapping between classifications
    print("\nMapping between KGMainGroup and KGMajorClass:")
    print(pd.crosstab(all_locations['KGMainGroup'], all_locations['KGMajorClass']))

if __name__ == "__main__":
    plot_kg_differences()
