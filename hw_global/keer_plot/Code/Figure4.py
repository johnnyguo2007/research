"""
Generate Figure 4
"""
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc 
FilePath='B:\\HeatStressPaper2022_v2\\Revision2\\Upload_20221130\\Data\\'

# Load summer precipitation
Precipitation=nc.Dataset(FilePath+'SummerPrecipitation.nc')
Rain=Precipitation.variables['Precip'][:]

# Load UDI component
UHIUDICom=nc.Dataset(FilePath+'ModelUHIUDIComponents_Figure4EDFigure2.nc')
UDIComDay=UHIUDICom.variables['DaytimeUDICom'][:]
UDIComNight=UHIUDICom.variables['NighttimeUDICom'][:]

Rain=np.ma.array(Rain, mask = UDIComDay.mask)

Wet=570;Dry=180
#define font and marker sizes
fs8=8;fs7=7;fs6=6;fs65=6.5
ms1=0.1;ms2=1.2;linew=0.4

# set the font globally
plt.rcParams.update({'font.sans-serif':'Arial'})

# plot Figure 5
BinNum=20

cm = 1/2.54 
fig = plt.figure(figsize=(8.9*cm, 5.476*cm))
widths = [1]
spec5 = fig.add_gridspec(ncols=1, nrows=1, width_ratios=widths)

ax = fig.add_subplot(spec5[0])
lenth=len(Rain.compressed());
SortUDI=np.array([x for _,x in sorted(zip(Rain.compressed(),UDIComDay.compressed()))])[0:int(lenth/BinNum)*BinNum]
SortRain=np.sort(Rain.compressed())[0:int(lenth/BinNum)*BinNum] 
print(int(lenth/BinNum)) # check the number of grid in each bin
MeanPrecip=np.nanmean(np.array(np.split(SortRain,BinNum)),axis=1)
UDICom=np.array(np.split(SortUDI,BinNum))
UDICom2=UDICom.tolist()
# define cap size and bar width
Wid=18;alp=0.8 
w1=1;w2=99;Linew=0.5
# define the box plot properties
c='black';
flierprops = dict(marker=".",markersize=ms1,alpha=alp,color=c,markeredgecolor=c)
boxprops = dict( linewidth=Linew, color=c);whiskerprops=dict( linewidth=Linew, color=c)
medianprops = dict( linewidth=Linew, color='black');capprops=dict( linewidth=Linew, color=c)
meanprops= dict(marker='x',markersize=ms2,markeredgewidth=linew,color='red',markeredgecolor='red',markerfacecolor='none')
ax.boxplot(UDICom2,positions=MeanPrecip,whis=(w1, w2),widths=Wid\
, showmeans=True,meanprops=meanprops,flierprops=flierprops,capprops=capprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops)

plt.xlim(-90,1500)  
plt.ylim(-3,1.1)
b=np.arange(-90,1500,10*3)
ax.set_yticks(np.arange(-3, 1.2, 1))   
ax.set_xticks(np.arange(0, 1600, 300))   
ax.set_xticklabels(['0','300','600','900','1200','1500'],fontsize=fs7)

plt.xticks(fontsize=fs7)
plt.yticks(fontsize=fs7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)  
ax.tick_params(axis='x',labelsize=fs7)
ax.tick_params(axis='y',labelsize=fs7)
yoffset=-2.9

plt.text(60*3,0.98,'180',horizontalalignment='center',verticalalignment='bottom',fontsize=fs7)        
plt.text(190*3,0.98,'570',horizontalalignment='center',verticalalignment='bottom',fontsize=fs7)        
plt.text(60*3/2,yoffset+0.34,'dry',horizontalalignment='center',verticalalignment='top',fontsize=fs7)    
plt.text((190*3+60*3)/2,yoffset+0.34,'inter.',horizontalalignment='center',verticalalignment='top',fontsize=fs7)    
plt.text((190*3+411*3)/2,yoffset+0.34,'wet',horizontalalignment='center',verticalalignment='top',fontsize=fs7) 
   
a=np.arange(-3,0.98,0.02)
plt.plot(a*0+60*3, a,'k--',linewidth=linew)
plt.plot(a*0+190*3, a,'k--',linewidth=linew)
ax.tick_params(axis='both',direction='in')    
         
plt.ylabel('UDI component'+' (°C)',fontsize=fs7,labelpad=1)
plt.xlabel(r'$P_{s}$'+' (mm)',fontsize=fs7,labelpad=1)
 
plt.subplots_adjust(top=0.93,
bottom=0.155,
left=0.14,
right=0.9,
hspace=0.2,
wspace=0.325)

# plt.savefig(FilePath+'Figure4.png', dpi=800)
# plt.savefig(FilePath+'Figure5.eps')
