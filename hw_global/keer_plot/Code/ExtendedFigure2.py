"""
Generate Extended Data Figure 2
"""
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.ticker
import warnings
warnings.filterwarnings("ignore")

FilePath='B:\\HeatStressPaper2022_v2\\Revision2\\Upload_20221130\\Data\\'

PartitionTerms=nc.Dataset(FilePath+'PartitionTerms.nc')
DayDeltaTw=PartitionTerms.variables['DaytimeDeltaTw'][:]
NightDeltaTw=PartitionTerms.variables['NighttimeDeltaTw'][:]

UHIUDICom=nc.Dataset(FilePath+'ModelUHIUDIComponents_Figure4EDFigure2.nc')
UHIComDay=UHIUDICom.variables['DaytimeUHICom'][:]
UHIComNight=UHIUDICom.variables['NighttimeUHICom'][:]
UDIComDay=UHIUDICom.variables['DaytimeUDICom'][:]
UDIComNight=UHIUDICom.variables['NighttimeUDICom'][:]

""" 
Cal_latitudinal_meanstd
Description: a function to calculate the latitudinal mean values of a variable
Input: Var--data in the shape of (LatNum,LonNum)
Output: LatMeanVar--Latitudinal mean values in the shape of (LatNum)
""" 
def Cal_latitudinal_meanstd(Var):
    MeanVar=np.zeros(Var[:,1].shape)
    for i in range(0,len(Var[:,1])):
        if len(Var[i,:].compressed())>10:  
            MeanVar[i]=np.nanmean(Var[i,:])
        else:
            MeanVar[i]=np.nan
    MeanVar=np.flip(MeanVar, axis=0)
    return MeanVar

""" 
BasemapPlot
Description: a function to visualize 2D global data
Input:  TobePlot--2D data to be visualized
        Lim1,Lim2--The range of the colorbar
        Interval--The interval of the colorbar
        Title-- Figure title
Output: the figure object
""" 
def BasemapPlot(TobePlot, Label, Lim1 ,Lim2 ,Interval,DN,Zone):
    lat = 538
    lon = 1152
    
    m = Basemap(projection='cyl',lon_0=0.,lat_0=0.,lat_ts=0.,fix_aspect=False,\
                llcrnrlat=-55.97132,urcrnrlat=70.05215,\
                llcrnrlon=-180,urcrnrlon=180.0,\
                rsphere=6371200.,resolution='l',area_thresh=10000)    
    m.drawcountries(linewidth=0.15)
    m.drawmapboundary(fill_color='lightcyan') #lightcyan
    m.fillcontinents(color='white',lake_color='lightcyan')
    m.drawcoastlines(color = '0.15',linewidth=0.5,zorder=3) #,zorder=3
    
    # draw parallels.
    parallels = np.arange(-90.,100.,30.) 
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=14,linewidth=0.5)
    # draw meridians
    meridians = np.arange(0.,360.,60.)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=14,linewidth=0.5)
    
    ny = lat; nx = lon
    lons, lats = m.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
    x, y = m(lons, lats) # compute map proj coordinates.
    if DN=='Day':
        clevsVPDif = [-0.4,-0.14,-0.05,0,0.05,0.14,0.25]
        colors = ['#313695','#4575b4', '#74add1','#abd9e9','#fee090','#f46d43','#d73027','#a50026']
    elif DN=='Night':
        clevsVPDif = [-0.4,-0.2,-0.05,0,0.15,0.3,0.6]
        colors = ['#313695','#4575b4', '#74add1','#abd9e9','#fee090','#f46d43','#d73027','#a50026']    
        
    cmap, norm = matplotlib.colors.from_levels_and_colors(clevsVPDif, colors, extend='both')
    cs = m.pcolormesh(x,y,TobePlot,zorder=2,cmap=cmap, norm=norm, shading='auto')
    
    plt.title(Label,fontsize=14) 
    return cs
""" 
PlotLatitudinalMean
Description: a function to visualize latitudinal mean data
Input:  Latmean -- the latitudinal mean values to be visualized
        linecolor -- color of line plot
        ylabel--label for y axis
        DN-- Daytime, Nighttime designation
""" 
def PlotLatitudinalMean(Latmean,linecolor,ylabel,DN):   
    x=np.arange(70.05215,-55.97132,-0.234375)
    plt.plot(Latmean,x,linecolor,linewidth=1.4,label=ylabel)
    plt.plot([0]*len(x),x,'k--',linewidth=0.7)
    plt.ylim( (-55.97132,70.05215) )
    plt.yticks(ticks=np.arange(-30,70, 30),labels=['30S','0','30N','60N'],fontsize=12)
    plt.xticks(fontsize=12)
    if DN==0:
        plt.xlim( (-0.5, 0.5) ) 
    elif DN==1:
        plt.xlim( (-1, 1) )   

fig = plt.figure(figsize=(13, 11))
widths = [5, 0.9,1,5, 0.9]
heights = [1,1,1]
spec5 = fig.add_gridspec(ncols=5, nrows=3, width_ratios=widths,height_ratios=heights)
TobePlot=np.ma.array([DayDeltaTw,NightDeltaTw])
VarName=r'$ΔT_{w}$'
TobePlot=np.ma.concatenate((TobePlot[:,:,576:1152],TobePlot[:,:,0:576]),axis=2)   
TobePlot=TobePlot[:,145:683,:]
Zone='All'
LatMeanTobePlotDay=Cal_latitudinal_meanstd(TobePlot[0,:,:])
LatMeanTobePlotNight=Cal_latitudinal_meanstd(TobePlot[1,:,:])

ax1 = fig.add_subplot(spec5[0, 0])
cs1=BasemapPlot(TobePlot[0,:,:], VarName, -1.2 ,0.8 ,0.2,'Day',Zone)
ax1.text(-195,80,'a', horizontalalignment='center',verticalalignment='center', size=16,weight='bold')  
ax1 = fig.add_subplot(spec5[0, 1])
ax1.yaxis.tick_right()
ax1.tick_params(axis='both',direction='in',labelsize=15)
ax1.yaxis.set_label_position("right")
PlotLatitudinalMean(LatMeanTobePlotDay,'k',VarName,0)

ax2 = fig.add_subplot(spec5[0, 3])
cs2=BasemapPlot(TobePlot[1,:,:], VarName, -0.6 ,1.4 ,0.2,'Night',Zone)
ax2.text(-195,80,'b', horizontalalignment='center',verticalalignment='center', size=16,weight='bold')  
ax2 = fig.add_subplot(spec5[0, 4])
ax2.yaxis.tick_right()
ax2.tick_params(axis='both',direction='in',labelsize=15)
ax2.yaxis.set_label_position("right")
PlotLatitudinalMean(LatMeanTobePlotNight,'k',VarName,1)

TobePlot=np.ma.array([UHIComDay,UHIComNight])
VarName='UHI component'
TobePlot=np.ma.concatenate((TobePlot[:,:,576:1152],TobePlot[:,:,0:576]),axis=2)   
TobePlot=TobePlot[:,145:683,:]
LatMeanTobePlotDay=Cal_latitudinal_meanstd(TobePlot[0,:,:])
LatMeanTobePlotNight=Cal_latitudinal_meanstd(TobePlot[1,:,:])
ax1 = fig.add_subplot(spec5[1, 0])
BasemapPlot(TobePlot[0,:,:], VarName, -1.2 ,0.8 ,0.2,'Day',Zone)
ax1.text(-195,80,'c', horizontalalignment='center',verticalalignment='center', size=16,weight='bold')  
ax1 = fig.add_subplot(spec5[1, 1])
ax1.yaxis.tick_right()
ax1.tick_params(axis='both',direction='in',labelsize=15)
ax1.yaxis.set_label_position("right")
PlotLatitudinalMean(LatMeanTobePlotDay,'k',VarName,0)

ax2 = fig.add_subplot(spec5[1, 3])
BasemapPlot(TobePlot[1,:,:], VarName, -0.6 ,1.4 ,0.2,'Night',Zone)
ax2.text(-195,80,'d', horizontalalignment='center',verticalalignment='center', size=16,weight='bold')  
ax2 = fig.add_subplot(spec5[1, 4])
ax2.yaxis.tick_right()
ax2.tick_params(axis='both',direction='in',labelsize=15)
ax2.yaxis.set_label_position("right")
PlotLatitudinalMean(LatMeanTobePlotNight,'k',VarName,1)

TobePlot=np.ma.array([UDIComDay,UDIComNight])
VarName='UDI component'
TobePlot=np.ma.concatenate((TobePlot[:,:,576:1152],TobePlot[:,:,0:576]),axis=2)   
TobePlot=TobePlot[:,145:683,:]
LatMeanTobePlotDay=Cal_latitudinal_meanstd(TobePlot[0,:,:])
LatMeanTobePlotNight=Cal_latitudinal_meanstd(TobePlot[1,:,:])
ax1 = fig.add_subplot(spec5[2, 0])
BasemapPlot(TobePlot[0,:,:], VarName, -1.2 ,0.8 ,0.2,'Day',Zone)  #Spectral jet seismic
ax1.text(-195,80,'e', horizontalalignment='center',verticalalignment='center', size=16,weight='bold')  
ax1 = fig.add_subplot(spec5[2, 1])
ax1.yaxis.tick_right()
ax1.tick_params(axis='both',direction='in',labelsize=15)
ax1.yaxis.set_label_position("right")
PlotLatitudinalMean(LatMeanTobePlotDay,'k',VarName,0)

ax2 = fig.add_subplot(spec5[2, 3])
BasemapPlot(TobePlot[1,:,:], VarName, -0.6 ,1.4 ,0.2,'Night',Zone)
ax2.text(-195,80,'f', horizontalalignment='center',verticalalignment='center', size=16,weight='bold')  
ax2 = fig.add_subplot(spec5[2, 4])
ax2.yaxis.tick_right()
ax2.tick_params(axis='both',direction='in',labelsize=15)
ax2.yaxis.set_label_position("right")
PlotLatitudinalMean(LatMeanTobePlotNight,'k',VarName,1)

# Left Cbar
cbar_ax2 = fig.add_axes([0.06, 0.09, 0.38, 0.015])
cbar2 = fig.colorbar(cs1, cax=cbar_ax2, orientation="horizontal", extend="both")
cbar2.ax.tick_params(labelsize=14) 
cbar_ax2.text(0.315,0.05,'(°C)', horizontalalignment='center',verticalalignment='center', size=14)

# Right Cbar
cbar_ax2 = fig.add_axes([0.56, 0.09, 0.38, 0.015])
cbar2 = fig.colorbar(cs2, cax=cbar_ax2, orientation="horizontal", extend="both")
cbar2.ax.tick_params(labelsize=14) 
cbar_ax2.text(0.7,0.05,'(°C)', horizontalalignment='center',verticalalignment='center', size=14)

plt.subplots_adjust(top=0.92,
bottom=0.14,
left=0.06,
right=0.95,
hspace=0.39,
wspace=0.175)  
# plt.savefig(FilePath+'\\EDFigure2.png', dpi=600)
# plt.savefig(FilePath+'\\EDFigure2.eps')
# plt.close()
