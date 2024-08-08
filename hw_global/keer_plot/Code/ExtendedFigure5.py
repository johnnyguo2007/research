"""
Generate Extended Figure 5
"""
import numpy as np
import netCDF4 as nc 
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
Wet=570;Dry=180
FilePath='B:\\HeatStressPaper2022_v2\\Revision2\\Upload_20221130\\Data\\'

# Load summer mean Tw and summer precipitation
Precipitation=nc.Dataset(FilePath+'SummerPrecipitation.nc')
Rain=Precipitation.variables['Precip'][:]

PartitionTerms=nc.Dataset(FilePath+'PartitionTerms.nc')
area=PartitionTerms.variables['area'][:,:] #96:192 96:192

DaytimeDeltaTw=PartitionTerms.variables['DaytimeDeltaTw'][:]
DaytimeConvectionTerm=PartitionTerms.variables['DaytimeConvectionTerm'][:]
DaytimeAnthropogenicHeatTerm=PartitionTerms.variables['DaytimeAnthropogenicHeatTerm'][:]
DaytimeHeatStorageTerm=PartitionTerms.variables['DaytimeHeatStorageTerm'][:]
DaytimeSolarAbsorptionTerm=PartitionTerms.variables['DaytimeSolarAbsorptionTerm'][:]
DaytimeLongwaveRadTerm=PartitionTerms.variables['DaytimeLongwaveRadTerm'][:]

NighttimeDeltaTw=PartitionTerms.variables['NighttimeDeltaTw'][:]
NighttimeConvectionTerm=PartitionTerms.variables['NighttimeConvectionTerm'][:]
NighttimeAnthropogenicHeatTerm=PartitionTerms.variables['NighttimeAnthropogenicHeatTerm'][:]
NighttimeHeatStorageTerm=PartitionTerms.variables['NighttimeHeatStorageTerm'][:]
NighttimeSolarAbsorptionTerm=PartitionTerms.variables['NighttimeSolarAbsorptionTerm'][:]
NighttimeLongwaveRadTerm=PartitionTerms.variables['NighttimeLongwaveRadTerm'][:]

"""
GlobalAreaMean
Description: a function to calculatet the area-weighted mean of gridded data
Input：  VarData -- data to be calculated 
Output: GlobalMean -- area-weighted mean
"""  
def GlobalAreaMean(VarData):
    MaskedArea=np.ma.masked_where(np.ma.getmask(VarData),area)
    GlobalMean=np.nansum(VarData*(MaskedArea/np.sum(MaskedArea)),axis=(0,1))
    return GlobalMean

"""
DivideClimateZone
Description: a function to devide a partition term into wet, inter, and dry 
            results and calculate regional mean.
Input：  Term -- a global partition term 
Output: DividedTerm -- a array of the gridded partition term in the wet, inter, and dry regions
        Mean -- a array of the regional mean value (wet, inter, and dry)
"""  
def DivideClimateZone(Term):
    TermWet=np.ma.masked_where(Rain<Wet,Term)
    TermDry=np.ma.masked_where(Rain>Dry,Term)
    TermInter=np.ma.masked_where((Rain<=Dry)|(Rain>=Wet),Term)
    DividedTerm=np.ma.array([TermWet.compressed(),TermInter.compressed(),TermDry.compressed()])
    Mean=np.ma.array([GlobalAreaMean(TermWet),GlobalAreaMean(TermInter),GlobalAreaMean(TermDry)])   
    return DividedTerm,Mean

[AnthropogenicHeatTerm3RegDay,AnthropogenicHeatMeanDay]=DivideClimateZone(DaytimeAnthropogenicHeatTerm)
[HeatStorageTerm3RegDay,HeatStorageMeanDay]=DivideClimateZone(DaytimeHeatStorageTerm)
[SolarAbsorptionTerm3RegDay,SolarAbsorptionMeanDay]=DivideClimateZone(DaytimeSolarAbsorptionTerm)
[LongwaveRadTerm3RegDay,LongwaveRadMeanDay]=DivideClimateZone(DaytimeLongwaveRadTerm)

[AnthropogenicHeatTerm3RegNight,AnthropogenicHeatMeanNight]=DivideClimateZone(NighttimeAnthropogenicHeatTerm)
[HeatStorageTerm3RegNight,HeatStorageMeanNight]=DivideClimateZone(NighttimeHeatStorageTerm)
[SolarAbsorptionTerm3RegNight,SolarAbsorptionMeanNight]=DivideClimateZone(NighttimeSolarAbsorptionTerm)
[LongwaveRadTerm3RegNight,LongwaveRadMeanNight]=DivideClimateZone(NighttimeLongwaveRadTerm)
# define the bar colors for each term
StorageColor='black';LongwaveColor='#3366FF';AlbedoColor='grey';AnthroColor='#CC0000'

def PlotBoxplot(HeatStorage,Albedo,LongwaveRad,Anthropogenic,HeatStorageMean,AlbedoMean,LongwaveRadMean,AnthropogenicMean,a,PlotNum):
    #viridis
    Wid=0.6 # define cap size and bar width
    x_pos = np.arange(1,12,4)
    w1=1;w2=99;Linew=1.5
    plt.hlines(0, -0.7, 12.2, colors='k', linestyles='solid')

    c=StorageColor;alp=0.4 # define the box plot properties
    flierprops = dict(marker=".",markersize=1,alpha=alp,color=c,markeredgecolor=c)
    boxprops = dict( linewidth=Linew, color=c);whiskerprops=dict( linewidth=Linew, color=c)
    medianprops = dict( linewidth=Linew, color=StorageColor);capprops=dict( linewidth=Linew, color=c)
    ax.boxplot(HeatStorage,positions=x_pos+0.0,whis=(w1, w2),widths=Wid\
    ,flierprops=flierprops,capprops=capprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops)
    ax.scatter(x_pos+0.0, HeatStorageMean,marker='x',s=20,c=c,edgecolors=c)
    
    c=AlbedoColor;alp=0.4 # define the box plot properties
    flierprops = dict(marker=".",markersize=1,alpha=alp,color=c,markeredgecolor=c)
    boxprops = dict( linewidth=Linew, color=c);whiskerprops=dict( linewidth=Linew, color=c)
    medianprops = dict( linewidth=Linew, color=AlbedoColor);capprops=dict( linewidth=Linew, color=c)
    ax.boxplot(Albedo,positions=x_pos+0.8,whis=(w1, w2),widths=Wid\
    ,flierprops=flierprops,capprops=capprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops)
    ax.scatter(x_pos+0.8, AlbedoMean,marker='x',s=20,c=c,edgecolors=c)
    
    c=LongwaveColor;alp=0.4 # define the box plot properties
    flierprops = dict(marker=".",markersize=1,alpha=alp,color=c,markeredgecolor=c)
    boxprops = dict( linewidth=Linew, color=c);whiskerprops=dict( linewidth=Linew, color=c)
    medianprops = dict( linewidth=Linew, color=LongwaveColor);capprops=dict( linewidth=Linew, color=c)
    ax.boxplot(LongwaveRad,positions=x_pos+1.6,whis=(w1, w2),widths=Wid\
    ,flierprops=flierprops,capprops=capprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops)
    ax.scatter(x_pos+1.6, LongwaveRadMean,marker='x',s=20,c=c,edgecolors=c)
    
    c=AnthroColor;alp=0.4 # define the box plot properties
    flierprops = dict(marker=".",markersize=1,alpha=alp,color=c,markeredgecolor=c)
    boxprops = dict( linewidth=Linew, color=c);whiskerprops=dict( linewidth=Linew, color=c)
    medianprops = dict( linewidth=Linew, color=AnthroColor);capprops=dict( linewidth=Linew, color=c)
    ax.boxplot(Anthropogenic,positions=x_pos+2.4,whis=(w1, w2),widths=Wid\
    ,flierprops=flierprops,capprops=capprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops)
    ax.scatter(x_pos+2.4, AnthropogenicMean,marker='x',s=20,c=c,edgecolors=c)
    
    plt.ylabel(r'$ΔT_{w}$'+' (°C)',fontsize=14,labelpad=-5)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.tick_params(axis='x',bottom=False)
    ticks_loc = ax.get_xticks().tolist()
    ax.set_xticks(ax.get_xticks().tolist()) # to use set_xticklabels we have to set xtick loc first 
    ax.set_xticklabels([' ']*len(ticks_loc),fontsize=9,fontweight='bold')
    ax.set_yticks(np.arange(-1.2, 1.9, 0.6)) 
    ax.tick_params(axis='y',labelsize=14,direction='in')
    plt.ylim((-1.5,2))
    if a==1:
        ax.set_ylim((-1.7,2)) 
        b=np.arange(ax.get_xlim()[0]+1.5,ax.get_xlim()[1]-1.5,0.1)
        plt.plot(b, b*0-1.2,'k',linewidth=1.5, zorder=1)
        plt.scatter([2.2,2.2+4,2.2+8], [-1.12,-1.12,-1.12],marker="^",s=50,c='k')
        plt.text(2.2,-1.25,'Wet',horizontalalignment='center',verticalalignment='top',fontsize=14) 
        plt.text(2.2+4,-1.25,'Inter.',horizontalalignment='center',verticalalignment='top',fontsize=14) 
        plt.text(2.2+8,-1.25,'Dry',horizontalalignment='center',verticalalignment='top',fontsize=14) 
    plt.text(-0.9,2.3,PlotNum,horizontalalignment='center',verticalalignment='top',fontsize=14,weight='bold') 
    if a==0:
        plt.text(0.1,2,'Day',horizontalalignment='left',verticalalignment='top',fontsize=14,weight='bold') 
    elif a==1:
        plt.text(0.1,2,'Night',horizontalalignment='left',verticalalignment='top',fontsize=14,weight='bold') 
    ax.set_xlim((0,12.2)) 

fig = plt.figure(figsize=(9.2,6))
heights = [1,1.2]
spec5 = fig.add_gridspec(ncols=1, nrows=2,
height_ratios=heights)

ax = fig.add_subplot(spec5[0])
PlotBoxplot(HeatStorageTerm3RegDay,SolarAbsorptionTerm3RegDay,LongwaveRadTerm3RegDay,AnthropogenicHeatTerm3RegDay,\
            HeatStorageMeanDay,SolarAbsorptionMeanDay,LongwaveRadMeanDay,AnthropogenicHeatMeanDay,0,'a')
    
ax = fig.add_subplot(spec5[1])
PlotBoxplot(HeatStorageTerm3RegNight,SolarAbsorptionTerm3RegNight,LongwaveRadTerm3RegNight,AnthropogenicHeatTerm3RegNight,\
            HeatStorageMeanNight,SolarAbsorptionMeanNight,LongwaveRadMeanNight,AnthropogenicHeatMeanNight,1,'b')

fig.patches.extend([plt.Rectangle((0.04,0.095),0.07,0.075,
fill=True, color='white',
transform=fig.transFigure, figure=fig)])

legend_elementsDTw = [Patch(facecolor=StorageColor, edgecolor='k',label='Heat storage'),
Patch(facecolor=AlbedoColor, edgecolor='k',label='Solar absorption'),
Patch(facecolor=LongwaveColor, edgecolor='k',label='Longwave radiation'),
Patch(facecolor=AnthroColor, edgecolor='k',label='Anthro. heat')
] 

fig.legend(handles=legend_elementsDTw,bbox_to_anchor=(0.65,0.5),frameon=False,loc='center left', prop={'size': 12})
# plt.text(0.96,0.48,'}',horizontalalignment='right',verticalalignment='top',fontsize=59,color='black', transform=plt.gcf().transFigure) 

plt.subplots_adjust(top=0.88,
bottom=0.11,
left=0.095,
right=0.640,
hspace=0.315,
wspace=0.2)

# plt.savefig(FilePath+'\\EDFigure5.png', dpi=600)
# # plt.savefig(FilePath+'\\EDFigure5.eps')
# plt.close()
