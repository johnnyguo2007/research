"""
Generate Extended Data Figure 1
"""
import numpy as np
import netCDF4 as nc 
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import BoundaryNorm
# Load site meta data and summer precipitation
FilePath='B:\\HeatStressPaper2022_v2\\Revision2\\Upload_20221130\\Data\\'
Meta=pd.read_csv(FilePath+'SiteMetaData.csv')

# Load summer mean Tw and summer precipitation
Precipitation=nc.Dataset(FilePath+'SummerPrecipitation.nc')
Rain=Precipitation.variables['Precip'][:]
     
TobePlot=Rain
fig = plt.figure(figsize=(10, 5.8))
lat = 538#[145:683]
lon = 1152
m = Basemap(projection='cyl',lon_0=0.,lat_0=0.,lat_ts=0.,fix_aspect=False,\
            llcrnrlat=-55.97132,urcrnrlat=70.05215,\
            llcrnrlon=-180,urcrnrlon=180.0,\
            rsphere=6371200.,resolution='l',area_thresh=10000)   
m.drawcountries(linewidth=0.15)
m.drawmapboundary(fill_color='lightcyan') #lightcyan
m.fillcontinents(color='none',lake_color='lightcyan')
m.drawcoastlines(color = '0.15',linewidth=0.5)

# draw parallels.
parallels = np.arange(-90.,100.,30.) 
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=14,linewidth=0.5)
# draw meridians
meridians = np.arange(0.,360.,60.)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=14,linewidth=0.5)

ny = lat; nx = lon
lons, lats = m.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
# lats=-lats
x, y = m(lons, lats) # compute map proj coordinates.

clevsVPDif = np.arange(-80 ,900 ,80)
TobePlot0=np.ma.concatenate((TobePlot[:,576:1152],TobePlot[:,0:576]),axis=1)    
TobePlot0=TobePlot0[145:683,:]
cmap = plt.get_cmap('BrBG')
norm = BoundaryNorm(clevsVPDif, ncolors=cmap.N, clip=False) 
cs = m.pcolormesh(x,y,TobePlot0,cmap=cmap, norm=norm, shading='auto') 
cbar = m.colorbar(cs,location='bottom',pad="13%")
cbar.ax.tick_params(labelsize=14) 
# cbar.ax.set_xticklabels(labels=clevsVPDif, weight='bold', fontsize=15)  
plt.figtext(0.905, 0.115,'(mm)', fontsize=15)

lons = np.array(Meta['Longitude'].iloc[::2])
lats =  np.array(Meta['Latitude'].iloc[::2])
x, y = m(lons, lats)
m.scatter(x,y,s=20,facecolor='black',edgecolors='white',zorder=3)
# plt.savefig(FilePath+'\\EDFigure1.png', dpi=600)
# plt.savefig(FilePath+'\\EDFigure1.eps')
# plt.close()
