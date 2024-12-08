"""
Updated on 2024/11/21
@author:
"""
import netCDF4 as nc 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import BoundaryNorm

# Read urban fraction map from 2015 to 2070
Folder='C:\\Users\\Ciel\\OneDrive - Princeton University\\2024Work\\WorkingPapers\\JGRA_DynamicUrbanRe1\\24111_\\data_241118\\'
FileUrbanFraction=nc.Dataset(Folder+'UrbanFraction2015-2070.nc')
UrbanFrac=FileUrbanFraction.variables['UrbanFrac'][:,:,:]
MeanDeltaUrbAll_BNU=np.nanmean(UrbanFrac[46:56],axis=0)-UrbanFrac[0]
landfrac=FileUrbanFraction.variables['landfrac'][:,:]
area0=FileUrbanFraction.variables['area'][:,:]
# Ignore the overflow error when multiplying two masked arrays 
np.seterr(over='ignore') 
area=area0*landfrac

# Read climate effects on air temperatrue and downward shortwave radiation
EnsembleMean=nc.Dataset(Folder+'EnsembleMean.nc')
DirectImpactTSA=EnsembleMean.variables['Direct_TSA'][:,:]
IndirectImpactTSA=EnsembleMean.variables['Indirect_TSA'][:,:]
IndirectPvalueTSA=EnsembleMean.variables['IndirectPvalue_TSA'][:,:]
TotalPvalueTSA=EnsembleMean.variables['TotalPvalue_TSA'][:,:]

IndirectImpactFSDS=EnsembleMean.variables['Indirect_FSDS'][:,:]
IndirectPvalueFSDS=EnsembleMean.variables['IndirectPvalue_FSDS'][:,:]

thre=0.05 # the confidence level is 95%
IndirectPvalueTSA_S=np.ma.masked_where(IndirectPvalueTSA>thre,IndirectPvalueTSA)
TotalPvalueTSA_S=np.ma.masked_where(TotalPvalueTSA>thre,TotalPvalueTSA)
IndirectPvalueFSDS_S=np.ma.masked_where(IndirectPvalueFSDS>thre,IndirectPvalueFSDS)

IndirectPvalueTSA_S_U=np.ma.masked_where(MeanDeltaUrbAll_BNU<0.01,IndirectPvalueTSA_S)
print(len(IndirectPvalueTSA_S_U.compressed()))
plt.rcParams.update({'hatch.color': '#585858'})
plt.rcParams.update({'font.sans-serif':'Arial'})

""" 
PlotContour
Description: a function to visualize 2D global data
Input:  TobePlot --2D data to be visualized
        Lim1, Lim2, Interval -- defines the upper and lower limit, as well as intervals of color levels
        Title -- map title
        Lable -- if this is direct/indirect/total climate effect
        colorscheme --colorscheme
"""
def PlotContour(TobePlot, Lim1, Lim2, Interval,Title,Lable,colorscheme):
    lat = 150 # no need to plot the Antarctic and Arctic 
    lon = 288
    m = Basemap(projection='cyl',lon_0=0.,lat_0=0.,lat_ts=0.,fix_aspect=False,\
                llcrnrlat=-56.0733,urcrnrlat=84.34555,\
                llcrnrlon=-180,urcrnrlon=180.0-1.25,\
                rsphere=6371200.,resolution='l',area_thresh=10000)    
    m.drawmapboundary(fill_color='lightcyan') 
    m.fillcontinents(color='white',lake_color='lightcyan')
    m.drawcoastlines(linewidth=0.3,zorder=3)
    
    # draw parallels.
    parallels = np.arange(-90.,100.,30.) 
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.3)
    # draw meridians
    meridians = np.arange(0.,360.,60.)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10,linewidth=0.3)
    
    ny = lat; nx = lon
    lons, lats = m.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
    x, y = m(lons, lats) # compute map proj coordinates.
    
    TobePlot2=np.ma.concatenate((TobePlot[:,144:288],TobePlot[:,0:144]),axis=1)
    TobePlot3=TobePlot2[36:186,:] # no need to plot the Antarctic and Arctic
    clevsVPDif = np.arange(Lim1, Lim2, Interval)
    cmap = plt.get_cmap(colorscheme)
    norm = BoundaryNorm(clevsVPDif, ncolors=cmap.N, clip=True)
    cs = m.pcolormesh(x, y, TobePlot3, cmap=cmap, norm=norm, zorder=2, shading='auto')
    cc='black' 
    if Title=='ΔTa (indirect effect)':
        Pvalue=np.ma.concatenate((IndirectPvalueTSA_S[36:186,144:288],IndirectPvalueTSA_S[36:186,0:144]),axis=1)
        plt.pcolor(x, y,Pvalue, alpha=0., hatch='.....',facecolor=cc,zorder=3,shading='auto')
    elif Title=='ΔK↓':
        Pvalue=np.ma.concatenate((IndirectPvalueFSDS_S[36:186,144:288],IndirectPvalueFSDS_S[36:186,0:144]),axis=1)
        plt.pcolor(x, y,Pvalue, alpha=0., hatch='.....', facecolor=cc,zorder=3,shading='auto')
    elif Title=='ΔTa (total effect)':
        Pvalue=np.ma.concatenate((TotalPvalueTSA_S[36:186,144:288],TotalPvalueTSA_S[36:186,0:144]),axis=1)
        plt.pcolor(x, y,Pvalue, alpha=0., hatch='.....', facecolor=cc,zorder=3,shading='auto')
    plt.title(Title,fontsize=14,fontweight='bold',pad=-1)  
    return cs

fig = plt.figure(figsize=(12,7.0))
heights = [1,1]
widths = [1,1]
spec5 = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths,
                  height_ratios=heights)

ax = fig.add_subplot(spec5[0,0])
cs1=PlotContour(DirectImpactTSA,-0.06, 0.06,0.01,'ΔTa (direct effect)','Direct','RdBu_r')
Deltax=plt.gca().get_xlim()[1]-plt.gca().get_xlim()[0];Deltay=plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0]
plt.text(plt.gca().get_xlim()[0]-Deltax*0.08,plt.gca().get_ylim()[1]+Deltay*0.055,'a',horizontalalignment='center',verticalalignment='center', size=16,fontweight='bold') 

ax = fig.add_subplot(spec5[0,1])
cs2=PlotContour(IndirectImpactTSA,-0.6, 0.7,0.1,'ΔTa (indirect effect)','Indirect','RdBu_r')
Deltax=plt.gca().get_xlim()[1]-plt.gca().get_xlim()[0];Deltay=plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0]
plt.text(plt.gca().get_xlim()[0]-Deltax*0.08,plt.gca().get_ylim()[1]+Deltay*0.055,'b',horizontalalignment='center',verticalalignment='center', size=16,fontweight='bold') 

ax = fig.add_subplot(spec5[1,0])
cs3=PlotContour(DirectImpactTSA+IndirectImpactTSA,-0.6, 0.7,0.1,'ΔTa (total effect)','Total','RdBu_r')
Deltax=plt.gca().get_xlim()[1]-plt.gca().get_xlim()[0];Deltay=plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0]
plt.text(plt.gca().get_xlim()[0]-Deltax*0.08,plt.gca().get_ylim()[1]+Deltay*0.055,'c',horizontalalignment='center',verticalalignment='center', size=16,fontweight='bold') 

ax = fig.add_subplot(spec5[1,1])
cs4=PlotContour(IndirectImpactFSDS, -6, 7, 1,'ΔK↓','Total','RdBu_r') 
Deltax=plt.gca().get_xlim()[1]-plt.gca().get_xlim()[0];Deltay=plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0]
plt.text(plt.gca().get_xlim()[0]-Deltax*0.08,plt.gca().get_ylim()[1]+Deltay*0.055,'d',horizontalalignment='center',verticalalignment='center', size=16,fontweight='bold') 
# Add legends
ax.tick_params(axis='both',direction='in')    
cbar_ax1 = fig.add_axes([0.105, 0.56, 0.35, 0.015])
cbar1 = fig.colorbar(cs1, cax=cbar_ax1, orientation="horizontal", extend="both")
cbar1.ax.tick_params(labelsize=11) 
cbar_ax1.text(0.0525,0.02,'°C', horizontalalignment='left',verticalalignment='center', size=13)

cbar_ax2 = fig.add_axes([0.54, 0.56, 0.35, 0.015])
cbar2 = fig.colorbar(cs2, cax=cbar_ax2, orientation="horizontal", extend="both")
cbar2.ax.tick_params(labelsize=11) 
cbar_ax2.text(0.66,-0.4,'°C', horizontalalignment='left',verticalalignment='center', size=13)

cbar_ax3 = fig.add_axes([0.105, 0.1, 0.35, 0.015])
cbar3 = fig.colorbar(cs3, cax=cbar_ax3, orientation="horizontal", extend="both")
cbar3.ax.tick_params(labelsize=11) 
cbar_ax3.text(0.66,-0.4,'°C', horizontalalignment='left',verticalalignment='center', size=13)

cbar_ax4 = fig.add_axes([0.54, 0.1, 0.35, 0.015])
cbar4 = fig.colorbar(cs4, cax=cbar_ax4, orientation="horizontal", extend="both")
cbar4.ax.tick_params(labelsize=11) 
cbar_ax4.text(6.7,0.4,'W $\mathregular{m^{-2}}$', horizontalalignment='left',verticalalignment='center', size=13)

plt.subplots_adjust(top=0.95,
bottom=0.15,
left=0.1,
right=0.9,
hspace=0.4,
wspace=0.25)

FigureFolder='C:\\Users\\Ciel\\OneDrive - Princeton University\\2024Work\\WorkingPapers\\JGRA_DynamicUrbanRe1\\24111_\\Figures_24111_\\'
plt.savefig(FigureFolder+'Figure3.pdf')
plt.savefig(FigureFolder+'Figure3.png', dpi=600)
plt.close()





