"""
Plot Figure 5
"""
import numpy as np
import netCDF4 as nc 
import matplotlib.pyplot as plt
import seaborn as sb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde
import matplotlib.patches as patches
import warnings
warnings.filterwarnings("ignore")

Wet=570;Dry=180
# FilePath='B:\\HeatStressPaper2022_v2\\Revision2\\Upload_20221130\\Data\\'
FilePath='/home/jguo/research/hw_global/keer_plot/Data/'
DangerousDaysNights=nc.Dataset(FilePath+'DangerousDaysNights_Figure5.nc')
DangerousDays_U=DangerousDaysNights.variables['DangerousDays_U'][:]
DangerousDays_R=DangerousDaysNights.variables['DangerousDays_R'][:]
DangerousNights_U=DangerousDaysNights.variables['DangerousNights_U'][:]
DangerousNights_R=DangerousDaysNights.variables['DangerousNights_R'][:]
CoastalGrid=DangerousDaysNights.variables['CoastalGrid'][:]
area=DangerousDaysNights.variables['area'][:,:] #96:192 96:192

Precipitation=nc.Dataset(FilePath+'SummerPrecipitation.nc')
Rain=Precipitation.variables['Precip'][:]

"""
GlobalAreaMean
Description: a function to calculatet the area-weighted mean of gridded data
Input： VarData -- data to be calculated 
Output: GlobalMean -- area-weighted mean
""" 
def GlobalAreaMean(VarData):
    MaskedArea=np.ma.masked_where(np.ma.getmask(VarData),area)
    GlobalMean=np.nansum(VarData*(MaskedArea/np.sum(MaskedArea)),axis=(0,1))
    return GlobalMean

"""
PlotByClimateZone
Description: a function to plot exceedance days
Input： Exceed_U -- excedance days in urban areas
Exceed_R -- excedance days in rural areas
Output: G
"""
def PlotByClimateZone(Exceed_R,Exceed_U,Zone,clip0,clip1):
    Exceed_DeltaDay=Exceed_U-Exceed_R
    CoExceed_DeltaDay=np.ma.masked_where(CoastalGrid==0, Exceed_DeltaDay)
    InExceed_DeltaDay=np.ma.masked_where(CoastalGrid==99, Exceed_DeltaDay)
    if DN=='Day':
        print('Coastal urban-rural dangerous days in '+Zone+' climate: '+str(GlobalAreaMean(CoExceed_DeltaDay)))
        print('Interior urban-rural dangerous days in '+Zone+' climate: '+str(GlobalAreaMean(InExceed_DeltaDay)))
    elif DN=='Night':
        print('Coastal urban-rural dangerous nights in '+Zone+' climate: '+str(GlobalAreaMean(CoExceed_DeltaDay)))
        print('Interior urban-rural dangerous nights in '+Zone+' climate: '+str(GlobalAreaMean(InExceed_DeltaDay)))
        
    c='black'
    flierprops = dict(marker=".",markersize=0.8,alpha=0.3,color=c,markeredgecolor=c)
    boxprops = dict( linewidth=1.5, color=c);whiskerprops=dict( linewidth=1.5, color=c)
    medianprops = dict( linewidth=1.5, color='black');capprops=dict( linewidth=1.5, color=c)
    
    if Zone=='Wet':
        pos=0.25
    elif Zone=='Inter':
        pos=3.05
    elif Zone=='Dry':
        pos=5.83
    y =CoExceed_DeltaDay.compressed()
    # Add some random "jitter" to the x-axis
    x = np.random.normal(pos, 0.07, size=len(y))

    Density1 = np.vstack([x,y])
    Densityz1 = gaussian_kde(Density1)(Density1) 
    ax.scatter(x, y,c=np.log(Densityz1),cmap='viridis',s=0.4) #color the scatters by log(density)
    
    y =InExceed_DeltaDay.compressed()
    # Add some random "jitter" to the x-axis
    x = np.random.normal(pos+1.15, 0.08, size=len(y))
    Density1 = np.vstack([x,y])
    Densityz1 = gaussian_kde(Density1)(Density1) 
    ax.scatter(x, y,c=np.log(Densityz1),cmap='viridis',s=0.4) #color the scatters by log(density)
    
    ax.boxplot(CoExceed_DeltaDay.compressed(), positions=[pos],whis=(0, 100),widths=0.25,vert=True,flierprops=flierprops,capprops=capprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops)
    # plot coastal PDF on an inserted axis:inset1
    sb.kdeplot(y=CoExceed_DeltaDay.compressed(), color='red',fill=True,alpha=.8, label='Wet',linewidth=0,clip=(clip0,clip1),ax=inset1) #cornflowerblue ,Label=''
    sb.kdeplot(y=CoExceed_DeltaDay.compressed(), color='black',fill=False, label='Wet',linewidth=1,clip=(clip0,clip1),common_norm=True,ax=inset1) #cornflowerblue ,Label=''

    # plot interior PDF on an inserted axis:inset2
    ax.boxplot(InExceed_DeltaDay.compressed(), positions=[1.15+pos],whis=(0, 100),widths=0.25,vert=True,flierprops=flierprops,capprops=capprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops)
    sb.kdeplot(y=InExceed_DeltaDay.compressed(), color='orange',shade=True,alpha=.8, label='Wet',linewidth=0,clip=(clip0,clip1),common_norm=True,ax=inset2) #cornflowerblue ,Label=''
    sb.kdeplot(y=InExceed_DeltaDay.compressed(), color='black',shade=False, label='Wet',linewidth=1,clip=(clip0,clip1),common_norm=True,ax=inset2) #cornflowerblue ,Label=''

fig = plt.figure(figsize=(9,8))
heights = [1,1]
spec5 = fig.add_gridspec(ncols=1, nrows=2,height_ratios=heights)
# not need to plot the PDF curves beyond (-15,20)
Clip0=-15;Clip1=20
DN='Day'
ax = fig.add_subplot(spec5[0])
ax.set_xlim((0,8))
ax.set_ylim((-45,35))
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False) 
ax.hlines(0, 0, 8, colors='k', linestyles='solid',linewidth=0.8)
ax.set_yticks(np.arange(-30, 35, 15)) 
ax.axes.xaxis.set_visible(False)
# define some properties for the inserted axes
# note that you may need to adjust the hgt to make the y-axis of the pdf curves 
# overlap with the y-axis of the scatter plots
wid=0.6;top=1.025;hgt=2.915#2.87
# this is an inset axes over the main axes
inset1 = inset_axes(ax, width=wid,height=hgt,
                    bbox_to_anchor=(0.16,top), bbox_transform=ax.transAxes) 

inset2 = inset_axes(ax, width=wid, height=hgt,
                    bbox_to_anchor=(0.3,top), bbox_transform=ax.transAxes) 

inset1.set_ylim((-45,35));inset2.set_ylim((-45,35))
inset1.set_xlim((0,0.8));inset2.set_xlim((0,0.8))
inset1.axis('off');inset2.axis('off');

DangerousDays_RWet=np.ma.masked_where((Rain<Wet), DangerousDays_R)
DangerousDays_UWet=np.ma.masked_where((Rain<Wet), DangerousDays_U)
PlotByClimateZone(DangerousDays_RWet,DangerousDays_UWet,'Wet',Clip0,Clip1)
ax.text(0.1,39,'Day',horizontalalignment='left',verticalalignment='top',fontsize=14,weight='bold') 
ax.set_ylabel('$\it{ΔN}$',fontsize=14,labelpad=-9)

# this is an inset axes over the main axes
inset1 = inset_axes(ax, width=wid, height=hgt,
                    bbox_to_anchor=(0.51,top), bbox_transform=ax.transAxes) 

inset2 = inset_axes(ax, width=wid, height=hgt, 
                    bbox_to_anchor=(0.65,top), bbox_transform=ax.transAxes) 

inset1.set_ylim((-45,35));inset2.set_ylim((-45,35))
inset1.set_xlim((0,0.8));inset2.set_xlim((0,0.8))
inset1.axis('off');inset2.axis('off');

DangerousDays_RInter=np.ma.masked_where((Rain<=Dry)|(Rain>=Wet), DangerousDays_R)
DangerousDays_UInter=np.ma.masked_where((Rain<=Dry)|(Rain>=Wet), DangerousDays_U)
PlotByClimateZone(DangerousDays_RInter,DangerousDays_UInter,'Inter',Clip0,Clip1)
# this is an inset axes over the main axes
inset1 = inset_axes(ax, width=wid, height=hgt,
                    bbox_to_anchor=(0.86,top), bbox_transform=ax.transAxes) 
inset2 = inset_axes(ax, width=wid, height=hgt, 
                    bbox_to_anchor=(1,top), bbox_transform=ax.transAxes) 

inset1.set_ylim((-45,35));inset2.set_ylim((-45,35))
inset1.set_xlim((0,0.8));inset2.set_xlim((0,0.8))
inset1.axis('off');inset2.axis('off');

DangerousDays_RDry=np.ma.masked_where((Rain>Dry), DangerousDays_R)
DangerousDays_UDry=np.ma.masked_where((Rain>Dry), DangerousDays_U)
PlotByClimateZone(DangerousDays_RDry,DangerousDays_UDry,'Dry',Clip0,Clip1)
ax.tick_params(axis='both',labelsize=14,direction='in')
ax.text(-0.3,43,'a',horizontalalignment='left',verticalalignment='top',fontsize=14,weight='bold') 

DN='Night'
Clip0=-5;Clip1=5 # not need to plot the PDF curves beyond (-5,5)
ax = fig.add_subplot(spec5[1])
ax.set_xlim((0,8))
ax.set_ylim((-18,18))
ax.hlines(0, 0, 8, colors='k', linestyles='solid',linewidth=0.8)
ax.axes.xaxis.set_visible(False)
ax.set_yticks(np.arange(-8,17, 8)) 

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False) 
#define some properties for the inserted axes

wid=0.6;hgt=2.915;xlim0=0;xlim1=5;ylim0=-18;ylim1=18
# this is an inset axes over the main axes
inset1 = inset_axes(ax, width=wid, height=hgt,
                    bbox_to_anchor=(0.16,top), bbox_transform=ax.transAxes) 
inset2 = inset_axes(ax, width=wid, height=hgt, 
                    bbox_to_anchor=(0.3,top), bbox_transform=ax.transAxes) 

inset1.set_ylim((ylim0,ylim1));inset2.set_ylim((ylim0,ylim1))
inset1.set_xlim((xlim0,xlim1));inset2.set_xlim((xlim0,xlim1))
inset1.axis('off');inset2.axis('off');

DangerousNights_RWet=np.ma.masked_where((Rain<Wet), DangerousNights_R)
DangerousNights_UWet=np.ma.masked_where((Rain<Wet), DangerousNights_U)
#Delta=np.max(Exceed_U0_AllHighDay-Exceed_R0_AllHighDay)
PlotByClimateZone(DangerousNights_RWet,DangerousNights_UWet,'Wet',Clip0,Clip1)

ax.text(0.1,20,'Night',horizontalalignment='left',verticalalignment='top',fontsize=14,weight='bold') 
ax.set_ylabel('$\it{ΔN}$',fontsize=14,labelpad=-9)
# this is an inset axes over the main axes
inset1 = inset_axes(ax, width=wid, height=hgt,
                    bbox_to_anchor=(0.51,top), bbox_transform=ax.transAxes) 
inset2 = inset_axes(ax, width=wid, height=hgt, 
                    bbox_to_anchor=(0.65,top), bbox_transform=ax.transAxes) 
inset1.set_ylim((ylim0,ylim1));inset2.set_ylim((ylim0,ylim1))
inset1.set_xlim((xlim0,xlim1));inset2.set_xlim((xlim0,xlim1))
inset1.axis('off');inset2.axis('off');

DangerousNights_RInter=np.ma.masked_where((Rain<=Dry)|(Rain>=Wet), DangerousNights_R)
DangerousNights_UInter=np.ma.masked_where((Rain<=Dry)|(Rain>=Wet), DangerousNights_U)
PlotByClimateZone(DangerousNights_RInter,DangerousNights_UInter,'Inter',Clip0,Clip1)
# this is an inset axes over the main axes
inset1 = inset_axes(ax, width=wid, height=hgt,
                    bbox_to_anchor=(0.86,top), bbox_transform=ax.transAxes) 
inset2 = inset_axes(ax, width=wid, height=hgt, 
                    bbox_to_anchor=(1,top), bbox_transform=ax.transAxes) 
inset1.set_ylim((ylim0,ylim1));inset2.set_ylim((ylim0,ylim1))
inset1.set_xlim((xlim0,xlim1));inset2.set_xlim((xlim0,xlim1))
inset1.axis('off');inset2.axis('off');

DangerousNights_RDry=np.ma.masked_where((Rain>Dry), DangerousNights_R)
DangerousNights_UDry=np.ma.masked_where((Rain>Dry), DangerousNights_U)
PlotByClimateZone(DangerousNights_RDry,DangerousNights_UDry,'Dry',Clip0,Clip1)

ax.tick_params(axis='both',labelsize=14,direction='in')
ax.text(-0.3,21,'b',horizontalalignment='left',verticalalignment='top',fontsize=14,weight='bold') 

# create a white rectangle to cover (-12,-18) of y-axis in figure b
rect = patches.Rectangle((-0.8, -0.2), 0.1, 0.5, linewidth=1, edgecolor='k', facecolor='k')
fig.patches.extend([plt.Rectangle((0.086,0.10),0.025,0.05,fill=True, color='white',
                                  transform=fig.transFigure, figure=fig,zorder=1)])

# add some text annotations
baseline=-17.5
b=np.arange(ax.get_xlim()[0],ax.get_xlim()[1]*0.93,0.1)
ax.plot(b, b*0+baseline,'k',linewidth=1.5)
ax.scatter([0.4,1.55,3.2,3.2+1.15,6,6+1.15], [baseline+0.35,baseline+0.35,baseline+0.35,baseline+0.35,baseline+0.35,baseline+0.35],marker="^",s=14,c='k')
ax.text(0.42,baseline+2.8,'Coastal',horizontalalignment='center',verticalalignment='top',fontsize=14) 
ax.text(1.55,baseline+2.8,'Interior',horizontalalignment='center',verticalalignment='top',fontsize=14) 

ax.text(3.2,baseline+2.8,'Coastal',horizontalalignment='center',verticalalignment='top',fontsize=14) 
ax.text(3.2+1.15,baseline+2.8,'Interior',horizontalalignment='center',verticalalignment='top',fontsize=14) 

ax.text(6,baseline+2.8,'Coastal',horizontalalignment='center',verticalalignment='top',fontsize=14) 
ax.text(7.15,baseline+2.8,'Interior',horizontalalignment='center',verticalalignment='top',fontsize=14) 
ax.text((0.4+1.55)/2.0,baseline-0.36,'Wet',horizontalalignment='center', fontweight="bold",verticalalignment='top',fontsize=14) 
ax.text((3.2+3.2+1.15)/2.0,baseline-0.36,'Inter.',horizontalalignment='center', fontweight="bold",verticalalignment='top',fontsize=14) 
ax.text((6+7.15)/2.0,baseline-0.36,'Dry',horizontalalignment='center', fontweight="bold",verticalalignment='top',fontsize=14) 

plt.subplots_adjust(top=0.9,
bottom=0.1,
left=0.11,
right=0.9,
hspace=0.23,
wspace=0.21)
plt.savefig(FilePath+'Figure5.png', dpi=600)
plt.close()

