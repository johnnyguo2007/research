"""
Generate Figure 3 b,d
"""
import numpy as np
import netCDF4 as nc 
import matplotlib.pyplot as plt
import seaborn as sb

# define precipitation thresholds for three climate zones
Wet=570;Dry=180

# FilePath='B:\\HeatStressPaper2022_v2\\Revision2\\Upload_20221130\\Data\\'
FilePath='/home/jguo/research/hw_global/keer_plot/Data/'
Precipitation=nc.Dataset(FilePath+'SummerPrecipitation.nc')
Rain=Precipitation.variables['Precip'][:]

PartitionTerms=nc.Dataset(FilePath+'PartitionTerms.nc')
DayDeltaTw=PartitionTerms.variables['DaytimeDeltaTw'][:]
NightDeltaTw=PartitionTerms.variables['NighttimeDeltaTw'][:]
area=PartitionTerms.variables['area'][:,:] #96:192 96:192

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

# calculate the mean delta Tw in three climate zones
DayDeltaWBAWet=np.ma.masked_where((Rain<=Wet) , DayDeltaTw) 
DayDeltaWBAWetRound=str(np.round(GlobalAreaMean(DayDeltaWBAWet),2))
DayDeltaWBADry=np.ma.masked_where((Rain>=Dry) , DayDeltaTw)
DayDeltaWBADryRound=str(np.round(GlobalAreaMean(DayDeltaWBADry),2))
DayDeltaWBAInter=np.ma.masked_where((Rain<Dry)| (Rain>Wet), DayDeltaTw) 
DayDeltaWBAInterRound=str(np.round(GlobalAreaMean(DayDeltaWBAInter),2))

NightDeltaWBAWet=np.ma.masked_where((Rain<=Wet) , NightDeltaTw)
NightDeltaWBAWetRound=str(np.round(GlobalAreaMean(NightDeltaWBAWet),2))
NightDeltaWBADry=np.ma.masked_where((Rain>=Dry) , NightDeltaTw)
NightDeltaWBADryRound=str(np.round(GlobalAreaMean(NightDeltaWBADry),2))
NightDeltaWBAInter=np.ma.masked_where((Rain<Dry)| (Rain>Wet),NightDeltaTw)
NightDeltaWBAInterRound=str(np.round(GlobalAreaMean(NightDeltaWBAInter),2))

DataDay = [DayDeltaWBADry.compressed(),DayDeltaWBAInter.compressed(),DayDeltaWBAWet.compressed()]
DataNight = [NightDeltaWBADry.compressed(),NightDeltaWBAInter.compressed(),NightDeltaWBAWet.compressed()]

fig = plt.figure(figsize=(12,4.2))

heights = [1,3]
widths = [1,1]

spec5 = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths,
                          height_ratios=heights)
wetcolor='#CC0000';intercolor='black';drycolor='#3366FF'
# plot the first box plot
ax = fig.add_subplot(spec5[0,0])

c=wetcolor # define the box plot properties
flierprops = dict(marker=".",markersize=0.8,alpha=0.3,color=c,markeredgecolor=c)
boxprops = dict( linewidth=1.5, color=c);whiskerprops=dict( linewidth=1.5, color=c)
medianprops = dict( linewidth=1.5, color='black');capprops=dict( linewidth=1.5, color=c)
ax.boxplot(DayDeltaWBAWet.compressed(), positions=[3],whis=(1, 99),widths=0.4,vert=False,flierprops=flierprops,capprops=capprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops)

c=intercolor # define the box plot properties
flierprops = dict(marker=".", markersize=0.8,alpha=0.3, color=c,markeredgecolor=c)
boxprops = dict( linewidth=1.5, color=c);whiskerprops=dict( linewidth=1.5, color=c)
medianprops = dict( linewidth=1.5, color='black');capprops=dict( linewidth=1.5, color=c)
ax.boxplot(DayDeltaWBAInter.compressed(), positions=[2],whis=(1, 99),widths=0.4,vert=False,flierprops=flierprops,capprops=capprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops)

c=drycolor # define the box plot properties
flierprops = dict(marker=".", markersize=0.8,alpha=0.3, color=c,markeredgecolor=c)
boxprops = dict( linewidth=1.5, color=c);whiskerprops=dict( linewidth=1.5, color=c)
medianprops = dict( linewidth=1.5, color='black');capprops=dict( linewidth=1.5, color=c)
ax.boxplot(DayDeltaWBADry.compressed(), positions=[1],whis=(1, 99),widths=0.4,vert=False,flierprops=flierprops,capprops=capprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops)

plt.text(1.46,3,'Wet',color='red',horizontalalignment='left',verticalalignment='center',weight='bold', fontsize=10) 
plt.text(1.46,2,'Inter.',color='black',horizontalalignment='left',verticalalignment='center',weight='bold', fontsize=10) 
plt.text(1.46,1,'Dry',color='blue',horizontalalignment='left',verticalalignment='center',weight='bold', fontsize=10) 

plt.text(-1.82,2.7,'Day',color='black',horizontalalignment='left',verticalalignment='center',weight='bold', fontsize=14) 

Labels = [ 'Dry','Inter.','Wet' ]
ax.axis('off')
y_pos = [i+1 for i, _ in enumerate(Labels)]
plt.xlim((-1.8,1.4))
plt.yticks(y_pos, Labels)
plt.tick_params(axis='both',direction='in')
ax.patch.set_facecolor('white')
fig.patch.set_alpha(0)

# plot the first PDF distribution
ax = fig.add_subplot(spec5[1,0])
ax=sb.kdeplot(DayDeltaWBAWet.compressed(), color=wetcolor,fill=False,label='Wet',linewidth=1.5) 
ax=sb.kdeplot(DayDeltaWBAInter.compressed(), color=intercolor,fill=False, label='InterInteriate',linewidth=1.5) 
ax=sb.kdeplot(DayDeltaWBADry.compressed(), color=drycolor,fill=False, label='Dry',linewidth=1.5) 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
plt.xlim((-1.8,1.4));plt.ylim((0,3.7))
plt.yticks(np.arange(0, 3.5,1),size=12)
plt.xticks(np.arange(-1, 2,1),size=12)
Deltax=plt.gca().get_xlim()[1]-plt.gca().get_xlim()[0];Deltay=plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0]
b=np.arange(0,ax.get_ylim()[1],0.1)
# annotate the mean delta Tw values of three climate zones
plt.text(plt.gca().get_xlim()[0]+Deltax*0.05,plt.gca().get_ylim()[1]-Deltay*0.075,DayDeltaWBAWetRound+'°C',color='red',horizontalalignment='left',verticalalignment='bottom',weight='bold', fontsize=9) 
plt.text(plt.gca().get_xlim()[0]+Deltax*0.05,plt.gca().get_ylim()[1]-Deltay*0.075*2,DayDeltaWBAInterRound+'°C',color='black',horizontalalignment='left',verticalalignment='bottom',weight='bold', fontsize=9) 
plt.text(plt.gca().get_xlim()[0]+Deltax*0.05,plt.gca().get_ylim()[1]-Deltay*0.075*3,DayDeltaWBADryRound+'0°C',color='blue',horizontalalignment='left',verticalalignment='bottom',weight='bold', fontsize=9) 

plt.xlabel(r'$ΔT_{w}$'+' (°C)',fontsize=13,labelpad=1) #r'$ΔT_{w}$'+' (°C)' ,fontweight='bold'
plt.ylabel('PDF',fontsize=13,labelpad=3)
plt.tick_params(axis='both',direction='in',labelsize=14)
plt.text(-2.0,5.5,'c',horizontalalignment='center',verticalalignment='center', size=13,fontweight='bold') 

# plot the second box plot
ax = fig.add_subplot(spec5[0,1])
c=wetcolor# define the box plot properties
flierprops = dict(marker=".",markersize=0.8,alpha=0.3,color=c,markeredgecolor=c)
boxprops = dict( linewidth=1.5, color=c);whiskerprops=dict( linewidth=1.5, color=c)
medianprops = dict( linewidth=1.5, color='black');capprops=dict( linewidth=1.5, color=c)
ax.boxplot(NightDeltaWBAWet.compressed(), positions=[3],whis=(1, 99),widths=0.4,vert=False,flierprops=flierprops,capprops=capprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops)

c=intercolor# define the box plot properties
flierprops = dict(marker=".", markersize=0.8,alpha=0.3, color=c,markeredgecolor=c)
boxprops = dict( linewidth=1.5, color=c);whiskerprops=dict( linewidth=1.5, color=c)
medianprops = dict( linewidth=1.5, color='black');capprops=dict( linewidth=1.5, color=c)
ax.boxplot(NightDeltaWBAInter.compressed(), positions=[2],whis=(1, 99),widths=0.4,vert=False,flierprops=flierprops,capprops=capprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops)

c=drycolor# define the box plot properties
flierprops = dict(marker=".", markersize=0.8,alpha=0.3, color=c,markeredgecolor=c)
boxprops = dict( linewidth=1.5, color=c);whiskerprops=dict( linewidth=1.5, color=c)
medianprops = dict( linewidth=1.5, color='black');capprops=dict( linewidth=1.5, color=c)
ax.boxplot(NightDeltaWBADry.compressed(), positions=[1],whis=(1, 99),widths=0.4,vert=False,flierprops=flierprops,capprops=capprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops)

plt.text(1.48,3,'Wet',color='red',horizontalalignment='left',verticalalignment='center',weight='bold', fontsize=10) 
plt.text(1.48,2,'Inter.',color='black',horizontalalignment='left',verticalalignment='center',weight='bold', fontsize=10) 
plt.text(1.48,1,'Dry',color='blue',horizontalalignment='left',verticalalignment='center',weight='bold', fontsize=10) 

plt.text(-1.82,2.7,'Night',color='black',horizontalalignment='left',verticalalignment='center',weight='bold', fontsize=14) 

ax.axis('off')
Labels = [ 'Dry','Inter.','Wet' ]
y_pos = [i+1 for i, _ in enumerate(Labels)]
plt.xlim((-1.8,1.4))
plt.yticks(y_pos, Labels)
plt.tick_params(axis='both',direction='in')

# plot the second PDF distribution
ax = fig.add_subplot(spec5[1,1])
ax=sb.kdeplot(NightDeltaWBAWet.compressed(), color=wetcolor,fill=False, label='Wet',linewidth=1.5) 
ax=sb.kdeplot(NightDeltaWBAInter.compressed(), color=intercolor,fill=False, label='InterInteriate',linewidth=1.5) 
ax=sb.kdeplot(NightDeltaWBADry.compressed(), color=drycolor,fill=False, label='Dry',linewidth=1.5) 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 

plt.xlim((-1.8,1.4));plt.ylim((0,3.7))
plt.yticks(np.arange(0, 3.5,1),size=12)
plt.xticks(np.arange(-1, 2,1),size=12)
plt.xticks(size=12)
Deltax=plt.gca().get_xlim()[1]-plt.gca().get_xlim()[0];Deltay=plt.gca().get_ylim()[1]-plt.gca().get_ylim()[0]
plt.text(plt.gca().get_xlim()[0]+Deltax*0.05,plt.gca().get_ylim()[1]-Deltay*0.075,NightDeltaWBAWetRound+'°C',color='red',horizontalalignment='left',verticalalignment='bottom',weight='bold', fontsize=9) 
plt.text(plt.gca().get_xlim()[0]+Deltax*0.05,plt.gca().get_ylim()[1]-Deltay*0.075*2,NightDeltaWBAInterRound+'°C',color='black',horizontalalignment='left',verticalalignment='bottom',weight='bold', fontsize=9) 
plt.text(plt.gca().get_xlim()[0]+Deltax*0.05,plt.gca().get_ylim()[1]-Deltay*0.075*3,NightDeltaWBADryRound+'°C',color='blue',horizontalalignment='left',verticalalignment='bottom',weight='bold', fontsize=9) 
b=np.arange(0,ax.get_ylim()[1],0.1)
plt.xlabel(r'$ΔT_{w}$'+' (°C)',fontsize=13,labelpad=1) #r'$ΔT_{w}$'+' (°C)' ,fontweight='bold'
plt.ylabel('PDF',fontsize=13,labelpad=3)
plt.tick_params(axis='both',direction='in',labelsize=14)
plt.text(-2.0,5.5,'d',horizontalalignment='center',verticalalignment='center', size=13,fontweight='bold') 

plt.subplots_adjust(top=0.909,
bottom=0.176,
left=0.166,
right=0.85,
hspace=0.194,
wspace=0.34)

plt.savefig(FilePath+'Figure3bd.png', dpi=600)
plt.close()
