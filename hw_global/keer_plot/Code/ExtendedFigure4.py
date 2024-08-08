"""
Generate Extended Data Figure 4
"""
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc 
from scipy import stats
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

FilePath='B:\\HeatStressPaper2022_v2\\Revision2\\Upload_20221130\\Data\\'
Precipitation=nc.Dataset(FilePath+'SummerPrecipitation.nc')
Rain=Precipitation.variables['Precip'][:]

PartitionTerms=nc.Dataset(FilePath+'PartitionTerms.nc')

DayDeltaTw=PartitionTerms.variables['DaytimeDeltaTw'][:]
DayDynamic=PartitionTerms.variables['DaytimeConvectionTerm'][:]
DaytimeAnthropogenicHeatTerm=PartitionTerms.variables['DaytimeAnthropogenicHeatTerm'][:]
DaytimeHeatStorageTerm=PartitionTerms.variables['DaytimeHeatStorageTerm'][:]
DaytimeSolarAbsorptionTerm=PartitionTerms.variables['DaytimeSolarAbsorptionTerm'][:]
DaytimeLongwaveRadTerm=PartitionTerms.variables['DaytimeLongwaveRadTerm'][:]

NightDeltaTw=PartitionTerms.variables['NighttimeDeltaTw'][:]
NightDynamic=PartitionTerms.variables['NighttimeConvectionTerm'][:]
NighttimeAnthropogenicHeatTerm=PartitionTerms.variables['NighttimeAnthropogenicHeatTerm'][:]
NighttimeHeatStorageTerm=PartitionTerms.variables['NighttimeHeatStorageTerm'][:]
NighttimeSolarAbsorptionTerm=PartitionTerms.variables['NighttimeSolarAbsorptionTerm'][:]
NighttimeLongwaveRadTerm=PartitionTerms.variables['NighttimeLongwaveRadTerm'][:]

DayDiabatic=DaytimeAnthropogenicHeatTerm+DaytimeHeatStorageTerm+DaytimeSolarAbsorptionTerm+DaytimeLongwaveRadTerm
NightDiabatic=NighttimeAnthropogenicHeatTerm+NighttimeHeatStorageTerm+NighttimeSolarAbsorptionTerm+NighttimeLongwaveRadTerm

# the dynamic/diagatic terms from partition analysis has fewer grids than the CLM delta Tw
# because some grids with negative aerodynamic resistence was masked in the partition analysis.
DayDeltaTw=np.ma.array(DayDeltaTw, mask = DayDynamic.mask)
NightDeltaTw=np.ma.array(NightDeltaTw, mask = NightDynamic.mask)

RainDaymask=np.ma.array(Rain, mask = DayDynamic.mask)
RainNightmask=np.ma.array(Rain, mask = NightDynamic.mask)

Wet=570;Dry=180

def BinScatter_Tw(X,Y,STRX,STRY,Color,BinNum,Zone):        
    lenth=len(X.compressed());
    # calculate binned average
    # sort X and Y using X values
    SortY=np.array([x for _,x in sorted(zip(X.compressed(),Y.compressed()))])[0:int(lenth/BinNum)*BinNum] 
    SortX=np.sort(X.compressed())[0:int(lenth/BinNum)*BinNum] 
    # print(int(lenth/BinNum))
    MeanPrecip=np.nanmean(np.array(np.split(SortX,BinNum)),axis=1)
    MeanDeltaTw=np.nanmean(np.array(np.split(SortY,BinNum)),axis=1)
    CalculateLinear=stats.linregress(MeanPrecip,MeanDeltaTw)
    Cov=np.cov(MeanPrecip,MeanDeltaTw)[0,1]
    if STRY == 'ΔTw':
        plt.scatter(MeanPrecip,MeanDeltaTw, s=16, facecolors='none', edgecolors='black',zorder=3)
    elif STRY == 'Dynamic mixing':
        plt.errorbar(MeanPrecip,MeanDeltaTw, ms=6,capsize=3, fmt="x",c=Color)
    elif STRY =='Diabatic heating':
        plt.errorbar(MeanPrecip,MeanDeltaTw, ms=6,capsize=3, fmt="^",c=Color)

    plt.xlim(-90,1500);plt.ylim(-1.0,1.1)
    b=np.arange(-90,1500,10*3)
    plt.plot(b, b*0,'k:',linewidth=1)
    ax.set_yticks(np.arange(-0.8, 1, 0.4))    
    plt.xticks(fontsize=12);plt.yticks(fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)  
    ax.set_xticks(np.arange(0, 1550, 300))
    ax.tick_params(axis='x',labelsize=16)
    ax.tick_params(axis='y',labelsize=16)
    yoffset=-1.17
    if STRY=='ΔTw':
        plt.text(60*3,0.98,'180',horizontalalignment='center',verticalalignment='bottom',fontsize=14)        
        plt.text(190*3,0.98,'570',horizontalalignment='center',verticalalignment='bottom',fontsize=14)        
        plt.text(60*3/2,yoffset+0.34,'Dry',horizontalalignment='center',verticalalignment='top',fontsize=16)    
        plt.text((190*3+60*3)/2,yoffset+0.34,'Inter.',horizontalalignment='center',verticalalignment='top',fontsize=16)    
        plt.text((190*3+411*3)/2,yoffset+0.34,'Wet',horizontalalignment='center',verticalalignment='top',fontsize=16)    
        a=np.arange(-1.12,0.98,0.02)
        plt.plot(a*0+60*3, a,'k--',linewidth=0.8)
        plt.plot(a*0+190*3, a,'k--',linewidth=0.8)
    return CalculateLinear,Cov

Bin=20
Zone='All'

fig = plt.figure(figsize=(13.5,4))
widths = [1,1]
spec5 = fig.add_gridspec(ncols=2, nrows=1, width_ratios=widths)

ax = fig.add_subplot(spec5[0])
[TwDay,TwDayCov]=BinScatter_Tw(RainDaymask,DayDeltaTw,'Precipitation','ΔTw','black',Bin,Zone)
[DynamicDay,DynamicDayCov]=BinScatter_Tw(RainDaymask,DayDynamic,'Precipitation','Dynamic mixing','blue',Bin,Zone)
[DiabaticDay,DiabaticDayCov]=BinScatter_Tw(RainDaymask,DayDiabatic,'Precipitation','Diabatic heating','red',Bin,Zone)

ax.text(-200,1.2,'a',horizontalalignment='left',verticalalignment='top',fontsize=18,weight='bold')   
ax.text(1140,1.03,'Day',horizontalalignment='left',verticalalignment='top',fontsize=18,weight='bold')
ax.tick_params(axis='both',direction='in')    
         
plt.ylabel(r'$ΔT_{w}$'+' (°C)',fontsize=18,labelpad=1)
plt.xlabel(r'$P_{s}$'+' (mm)',fontsize=18,labelpad=1)

ax = fig.add_subplot(spec5[1])
[TwNight,TwNightCov]=BinScatter_Tw(RainNightmask,NightDeltaTw,'Precipitation','ΔTw','black',Bin,Zone)
[DynamicNight,DynamicNightCov]=BinScatter_Tw(RainNightmask,NightDynamic,'Precipitation','Dynamic mixing','blue',Bin,Zone)
[DiabaticNight,DiabaticNightCov]=BinScatter_Tw(RainNightmask,NightDiabatic,'Precipitation','Diabatic heating','red',Bin,Zone)

plt.ylabel(r'$ΔT_{w}$'+' (°C)',fontsize=18,labelpad=2)
plt.xlabel(r'$P_{s}$'+' (mm)',fontsize=18,labelpad=2)

ax.text(-200,1.2,'b',horizontalalignment='left',verticalalignment='top',fontsize=16,weight='bold')      
ax.text(1140,1.03,'Night',horizontalalignment='left',verticalalignment='top',fontsize=16,weight='bold')  
ax.tick_params(axis='both',direction='in')     
   
legend_elements = [Line2D([0], [0], marker='o',lw=0, color='black', label=r'$ΔT_{w}$',markerfacecolor='none', markersize=6),
Line2D([0], [0], marker='x',lw=0, color='blue', label='Dynamic mixing',markerfacecolor='blue', markersize=6),
Line2D([0], [0], marker='^',lw=0, color='red', label='Diabatic heating',markerfacecolor='red', markersize=6)] 

fig.legend(handles=legend_elements,bbox_to_anchor=(0.78,0.55),frameon=False,loc='center left', prop={'size': 13.5})
plt.subplots_adjust(top=0.915,
bottom=0.155,
left=0.095,
right=0.775,
hspace=0.2,
wspace=0.325)
# plt.savefig(FilePath+'\\EDFigure4.png', dpi=600)
# # plt.savefig(FilePath+'\\EDFigure4.eps')
# plt.close()
