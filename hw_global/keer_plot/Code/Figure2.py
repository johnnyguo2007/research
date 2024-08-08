"""
Generate Figure 2
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

# The diagnostic analysis excluded some grids with unreasonable ra
# The grid number of DeltaTw may be larger than that of diagnostic analysis results
DaytimeDeltaTw=np.ma.masked_where(np.ma.getmask(DaytimeConvectionTerm),DaytimeDeltaTw)
NighttimeDeltaTw=np.ma.masked_where(np.ma.getmask(NighttimeConvectionTerm),NighttimeDeltaTw)

DaytimeCalculatedTw=DaytimeConvectionTerm+DaytimeAnthropogenicHeatTerm+DaytimeHeatStorageTerm+DaytimeSolarAbsorptionTerm+DaytimeLongwaveRadTerm
NighttimeCalculatedTw=NighttimeConvectionTerm+NighttimeAnthropogenicHeatTerm+NighttimeHeatStorageTerm+NighttimeSolarAbsorptionTerm+NighttimeLongwaveRadTerm

DaytimeDiabaticTerm=DaytimeAnthropogenicHeatTerm+DaytimeHeatStorageTerm+DaytimeSolarAbsorptionTerm+DaytimeLongwaveRadTerm
NighttimeDiabaticTerm=NighttimeAnthropogenicHeatTerm+NighttimeHeatStorageTerm+NighttimeSolarAbsorptionTerm+NighttimeLongwaveRadTerm

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
    DividedTerm=np.ma.array([TermWet.compressed(),TermInter.compressed(),TermDry.compressed()],dtype=object)
    Mean=np.ma.array([GlobalAreaMean(TermWet),GlobalAreaMean(TermInter),GlobalAreaMean(TermDry)],dtype=object) 
    return DividedTerm,Mean

[ConvectionTerm3RegDay,ConvectionMeanDay]=DivideClimateZone(DaytimeConvectionTerm)
[DeltaTw3RegDay,DeltaTwMeanDay]=DivideClimateZone(DaytimeDeltaTw)
[DiabaticTerm3RegDay,DiabaticMeanDay]=DivideClimateZone(DaytimeDiabaticTerm)
[CalculatedTw3RegDay,CalculatedTwMeanDay]=DivideClimateZone(DaytimeCalculatedTw)

[ConvectionTerm3RegNight,ConvectionMeanNight]=DivideClimateZone(NighttimeConvectionTerm)
[DeltaTw3RegNight,DeltaTwMeanNight]=DivideClimateZone(NighttimeDeltaTw)
[DiabaticTerm3RegNight,DiabaticMeanNight]=DivideClimateZone(NighttimeDiabaticTerm)
[CalculatedTw3RegNight,CalculatedTwMeanNight]=DivideClimateZone(NighttimeCalculatedTw)

def PlotBoxplot(WBA,CalculatedTw,Convection,Diabatic,WBAMean,CalculatedTwMean,ConvectionMean,DiabaticMean,a,PlotNum):
     #viridis plasma magma cividis
    alp=0.9
    Wid=0.6 # define cap size and bar width
    x_pos = np.arange(1,12,4)
    w1=1;w2=99;Linew=1.5
    # v1=-3;v2=3
    plt.hlines(0, -0.7, 12.2, colors='k', linestyles='solid')
    
    c='black';# define the box plot properties
    flierprops = dict(marker=".",markersize=1,alpha=alp,color=c,markeredgecolor=c)
    boxprops = dict( linewidth=Linew, color=c);whiskerprops=dict( linewidth=Linew, color=c)
    medianprops = dict( linewidth=Linew, color='black');capprops=dict( linewidth=Linew, color=c)
    ax.boxplot(WBA,positions=x_pos+0.0,whis=(w1, w2),widths=Wid\
    ,flierprops=flierprops,capprops=capprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops)
    ax.scatter(x_pos+0.0, WBAMean,marker='x',s=20,c=c,edgecolors=c)
    
    c='grey'; # define the box plot properties
    flierprops = dict(marker=".",markersize=1,alpha=alp,color=c,markeredgecolor=c)
    boxprops = dict( linewidth=Linew, color=c);whiskerprops=dict( linewidth=Linew, color=c)
    medianprops = dict( linewidth=Linew, color='black');capprops=dict( linewidth=Linew, color=c)
    ax.boxplot(CalculatedTw,positions=x_pos+0.8,whis=(w1, w2),widths=Wid\
    ,flierprops=flierprops,capprops=capprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops)
    ax.scatter(x_pos+0.8, CalculatedTwMean,marker='x',s=20,c=c,edgecolors=c)
     
    c='#3366FF';# define the box plot properties
    flierprops = dict(marker=".",markersize=1,alpha=alp,color=c,markeredgecolor=c)
    boxprops = dict( linewidth=Linew, color=c);whiskerprops=dict( linewidth=Linew, color=c)
    medianprops = dict( linewidth=Linew, color='black');capprops=dict( linewidth=Linew, color=c)
    ax.boxplot(Convection,positions=x_pos+1.6,whis=(w1, w2),widths=Wid\
    ,flierprops=flierprops,capprops=capprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops)
    ax.scatter(x_pos+1.6, ConvectionMean,marker='x',s=20,c=c,edgecolors=c)

    c='#CC0000'; # define the box plot properties
    flierprops = dict(marker=".",markersize=1,alpha=alp,color=c,markeredgecolor=c)
    boxprops = dict( linewidth=Linew, color=c);whiskerprops=dict( linewidth=Linew, color=c)
    medianprops = dict( linewidth=Linew, color='black');capprops=dict( linewidth=Linew, color=c)
    ax.boxplot(Diabatic,positions=x_pos+2.4,whis=(w1, w2),widths=Wid\
    ,flierprops=flierprops,capprops=capprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops)
    ax.scatter(x_pos+2.4, DiabaticMean,marker='x',s=20,c=c,edgecolors=c)

    plt.ylabel(r'$ΔT_{w}$'+' (°C)',fontsize=14,labelpad=-6)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.tick_params(axis='x',bottom=False)
    ticks_loc = ax.get_xticks().tolist()
    ax.set_xticks(ax.get_xticks().tolist()) # to use set_xticklabels we have to set xtick loc first 
    ax.set_xticklabels([' ']*len(ticks_loc),fontsize=9,fontweight='bold')
    ax.set_yticks(np.arange(-1.2, 1.9, 0.6)) 
    ax.tick_params(axis='y',labelsize=14,direction='in')
    plt.ylim((-2,2))
    if a==1:
        ax.set_ylim((-2,2)) 
        b=np.arange(ax.get_xlim()[0]+1.5,ax.get_xlim()[1]-1.5,0.1)
        plt.plot(b, b*0-2,'k',linewidth=1.5, zorder=1)
        plt.scatter([2.2,2.2+4,2.2+8], [-1.95,-1.95,-1.95],marker="^",s=50,c='k')
        plt.text(2.2,-2.05,'Wet',horizontalalignment='center',verticalalignment='top',fontsize=14) 
        plt.text(2.2+4,-2.05,'Inter.',horizontalalignment='center',verticalalignment='top',fontsize=14) 
        plt.text(2.2+8,-2.05,'Dry',horizontalalignment='center',verticalalignment='top',fontsize=14) 
        
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
PlotBoxplot(DeltaTw3RegDay,CalculatedTw3RegDay,ConvectionTerm3RegDay,DiabaticTerm3RegDay,DeltaTwMeanDay,CalculatedTwMeanDay,ConvectionMeanDay,DiabaticMeanDay,0,'a')

ax = fig.add_subplot(spec5[1])
PlotBoxplot(DeltaTw3RegNight,CalculatedTw3RegNight,ConvectionTerm3RegNight,DiabaticTerm3RegNight,DeltaTwMeanNight,CalculatedTwMeanNight,ConvectionMeanNight,DiabaticMeanNight,1,'b')

fig.patches.extend([plt.Rectangle((0.05,0.095),0.05,0.075,
fill=True, color='white',
transform=fig.transFigure, figure=fig)])

legend_elementsDTw = [Patch(facecolor='black', edgecolor='k',label='Modelled '+r'$ΔT_{w}$'),
Patch(facecolor='dimgray', edgecolor='k',label='Calculated '+r'$ΔT_{w}$'),
Patch(facecolor='blue', edgecolor='k',label='Dynamic mixing'),
Patch(facecolor='red', edgecolor='k',label='Diabatic heating')] 

fig.legend(handles=legend_elementsDTw,bbox_to_anchor=(0.65,0.5),frameon=False,loc='center left', prop={'size': 12})

plt.subplots_adjust(top=0.88,
bottom=0.11,
left=0.095,
right=0.640,
hspace=0.315,
wspace=0.2)
# plt.savefig(FilePath+'\\Figure2.png', dpi=600)
# plt.savefig(FilePath+'\\Figure2.eps')
# plt.close()
