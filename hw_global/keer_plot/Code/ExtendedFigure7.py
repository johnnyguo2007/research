"""
Generate Extended Figure 7
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
Gamma = 0.66
FilePath='B:\\HeatStressPaper2022_v2\\Revision2\\Upload_20221130\\Data\\'

WetSiteModel=pd.read_csv(FilePath+'EDFigure7_WetModel.csv',sep=',')
WetSiteObs=pd.read_csv(FilePath+'EDFigure7_WetObs.csv',sep=',')

fs8=8;fs7=7;fs6=6;
ms1=4;ms2=2;linew=1;linew2=0.8
# set the font globally
plt.rcParams.update({'font.sans-serif':'Arial'})

#plot ED Figure 7
cm = 1/2.54 
fig = plt.figure(figsize=(18.3*cm, 4.3*cm))
widths = [1,1,1,1]
heights = [1]

spec5 = fig.add_gridspec(ncols=4,nrows=1, width_ratios=widths,height_ratios=heights)
time0=np.arange(2,21,6)
time=np.arange(0,24)

ax = fig.add_subplot(spec5[0])
plt.plot(time,WetSiteModel.loc[:,'ModelTw_U'],color='Red', label="Modelled urban "+r'$T_{w}$', linewidth=linew)
plt.scatter(time0,WetSiteObs.loc[:,'ObsTw_U'],color='Red',s=ms1, label="Observed urban "+r'$T_{w}$')#,color='Red', linestyle='dashed', label="Observed urban "+r'$T_{w}$'
plt.plot(time,WetSiteModel.loc[:,'ModelTw_R'],color='blue', label="Modelled rural "+r'$T_{w}$', linewidth=linew)
plt.scatter(time0,WetSiteObs.loc[:,'ObsTw_R'],color='blue',s=ms1, label="Observed rural "+r'$T_{w}$')#,color='blue', linestyle='dashed', label="Observed rural "+r'$T_{w}$'
ax.set_ylim((21,31)) 
ax.set_ylabel(r'$T_{w}$'+' (°C)',fontsize=fs7,labelpad=0)
ax.set_xlabel('Local time',fontsize=fs7,labelpad=0)
ax.set_xticks(np.arange(0, 25, 6))  
plt.text(0,31.2,'Wet region',horizontalalignment='left',verticalalignment='bottom',fontsize=fs7,weight='bold') 
ax.tick_params(axis='both',direction='in', labelsize= fs7) 
plt.legend( prop={'size': 5.8},ncol=1,loc='upper left')#bbox_to_anchor=(0.67,0.86),frameon=False,
plt.text(-4.1,31.2,'a',horizontalalignment='right',verticalalignment='bottom',fontsize=fs8,weight='bold')         
ax.set_xlim((-0.5,24.5))   

ax = fig.add_subplot(spec5[1])
Upper=WetSiteModel.loc[:,'ModelTw_U']-WetSiteModel.loc[:,'ModelTw_R']+WetSiteModel.loc[:,'ModelDeltaTw_std']
Lower=WetSiteModel.loc[:,'ModelTw_U']-WetSiteModel.loc[:,'ModelTw_R']-WetSiteModel.loc[:,'ModelDeltaTw_std']
plt.fill_between(time,Lower, Upper, alpha=0.2,color='grey')    
plt.plot(time,WetSiteModel.loc[:,'ModelTw_U']-WetSiteModel.loc[:,'ModelTw_R'],color='black', label="Modelled", linewidth=linew)
plt.errorbar(time0,WetSiteObs.loc[:,'ObsTw_U']-WetSiteObs.loc[:,'ObsTw_R'], yerr=WetSiteObs.loc[:,'ObsDeltaTw_std'],ms=ms2,color='Red',elinewidth=linew2, fmt="o", label="Observed ")   
ax.set_ylabel(r'$ΔT_{w}$'+' (°C)',fontsize=fs7,labelpad=0)
ax.set_xlabel('Local time',fontsize=fs7,labelpad=0)
ax.set_xticks(np.arange(0, 25, 6))  
ax.set_ylim((-1.8,1.8))   
ax.tick_params(axis='both',direction='in', labelsize= fs7) 
plt.legend( prop={'size': fs6})#frameon=False,
plt.text(-4.1,1.86,'b',horizontalalignment='right',verticalalignment='bottom',fontsize=fs8,weight='bold')         
ax.set_xlim((-0.5,24.5))   

ax = fig.add_subplot(spec5[2])
Upper=WetSiteModel.loc[:,'ModelUHICom']+WetSiteModel.loc[:,'ModelUHICom_std']
Lower=WetSiteModel.loc[:,'ModelUHICom']-WetSiteModel.loc[:,'ModelUHICom_std']
plt.fill_between(time,Lower, Upper, alpha=0.2,color='grey')    
plt.plot(time,WetSiteModel.loc[:,'ModelUHICom'],color='black', label='Modelled', linewidth=linew)
plt.errorbar(time0,WetSiteObs.loc[:,'ObsUHICom'], yerr=WetSiteObs.loc[:,'ObsUHICom_std'],ms=ms2,color='Red',elinewidth=linew2, fmt="o", label='Observed')   
ax.set_ylabel('UHI component (°C)',fontsize=fs7,labelpad=0)
ax.set_xlabel('Local time',fontsize=fs7,labelpad=0)
ax.set_xticks(np.arange(0, 25, 6))  
ax.set_ylim((-1.8,1.8))   
ax.tick_params(axis='both',direction='in', labelsize= fs7) 
plt.legend(prop={'size': fs6},loc='lower center')  
plt.text(-4.1,1.86,'c',horizontalalignment='right',verticalalignment='bottom',fontsize=fs8,weight='bold')         
ax.set_xlim((-0.5,24.5))   

ax = fig.add_subplot(spec5[3])
Upper=WetSiteModel.loc[:,'ModelUDICom']+WetSiteModel.loc[:,'ModelUDICom_std']
Lower=WetSiteModel.loc[:,'ModelUDICom']-WetSiteModel.loc[:,'ModelUDICom_std']
plt.fill_between(time,Lower, Upper, alpha=0.2,color='grey')    
plt.plot(time,WetSiteModel.loc[:,'ModelUDICom'],color='black', label='Modelled', linewidth=linew)
plt.errorbar(time0,WetSiteObs.loc[:,'ObsUDICom'], yerr=WetSiteObs.loc[:,'ObsUDICom_std'],ms=ms2,color='Red',elinewidth=linew2, fmt="o", label='Observed')   
ax.set_ylabel('UDI component (°C)',fontsize=fs7,labelpad=0)
ax.set_xlabel('Local time',fontsize=fs7,labelpad=0)
ax.set_xticks(np.arange(0, 25, 6))  
ax.set_ylim((-1.8,1.8))   
ax.tick_params(axis='both',direction='in', labelsize= fs7) 
plt.legend(prop={'size': fs6})   
plt.text(-4.1,1.86,'d',horizontalalignment='right',verticalalignment='bottom',fontsize=fs8,weight='bold')         
ax.set_xlim((-0.5,24.5))   

plt.subplots_adjust(top=0.9,
bottom=0.15,
left=0.06,
right=0.99,
hspace=0.37,
wspace=0.31) 
# plt.savefig(FilePath+'EDfig7.jpg', dpi=1200)

