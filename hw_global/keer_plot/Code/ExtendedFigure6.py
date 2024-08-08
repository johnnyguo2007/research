"""
Generate Extended Figure 6
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
Gamma = 0.66
FilePath='B:\\HeatStressPaper2022_v2\\Revision2\\Upload_20221130\\Data\\'

BerlinDiurnal = pd.read_csv(FilePath+'EDFigure6_Berlin.csv',sep=',')
PhoenixDiurnal = pd.read_csv(FilePath+'EDFigure6_Phoenix.csv',sep=',')

#font sizes
fs8=8;fs7=7;fs6=6;
linew=1
# set the font globally
plt.rcParams.update({'font.sans-serif':'Arial'})

#plot ED figure 6
cm = 1/2.54 
fig = plt.figure(figsize=(18.3*cm, 9.15*cm))
widths = [1,1,1,1]
heights = [1,1]

spec5 = fig.add_gridspec(ncols=4,nrows=2, width_ratios=widths,height_ratios=heights)
time=np.arange(0,24)

ax = fig.add_subplot(spec5[0,0])    
plt.plot(time,BerlinDiurnal.loc[:,'ModelTw_U'],color='Red', label="Modelled urban "+r'$T_{w}$', linewidth=linew)
plt.plot(time,BerlinDiurnal.loc[:,'ObsTw_U'],color='Red', linestyle='dashed', label="Observed urban "+r'$T_{w}$', linewidth=linew)
plt.plot(time,BerlinDiurnal.loc[:,'ModelTw_R'],color='blue', label="Modelled rural "+r'$T_{w}$', linewidth=linew)
plt.plot(time,BerlinDiurnal.loc[:,'ObsTw_R'],color='blue', linestyle='dashed', label="Observed rural "+r'$T_{w}$', linewidth=linew)
ax.set_ylabel(r'$T_{w}$'+' (°C)',fontsize=fs7,labelpad=0)
ax.set_xticks(np.arange(0, 25, 6))  
ax.set_ylim((11,28))  
ax.set_xlim((-0.5,24.5))   
plt.text(0,28.2,'Berlin',horizontalalignment='left',verticalalignment='bottom',fontsize=fs7,weight='bold') 
ax.tick_params(axis='both',direction='in', labelsize= fs7) 
plt.legend(prop={'size': 5.8},ncol=1,loc='upper left')#bbox_to_anchor=(0.67,0.86),frameon=False,
plt.text(-4.1,28.4,'a',horizontalalignment='right',verticalalignment='bottom',fontsize=fs8,weight='bold')         

ax = fig.add_subplot(spec5[0,1])
Upper=BerlinDiurnal.loc[:,'ObsTw_U']-BerlinDiurnal.loc[:,'ObsTw_R']+BerlinDiurnal.loc[:,'ObsDeltaTw_std']
Lower=BerlinDiurnal.loc[:,'ObsTw_U']-BerlinDiurnal.loc[:,'ObsTw_R']-BerlinDiurnal.loc[:,'ObsDeltaTw_std']
plt.fill_between(time,Lower, Upper, alpha=0.2,color='red')    
plt.plot(time,BerlinDiurnal.loc[:,'ModelTw_U']-BerlinDiurnal.loc[:,'ModelTw_R'],color='black', label="Modelled", linewidth=linew)
plt.plot(time,BerlinDiurnal.loc[:,'ObsTw_U']-BerlinDiurnal.loc[:,'ObsTw_R'],color='Red', linestyle='dashed', label="Observed", linewidth=linew)
ax.set_ylabel(r'$ΔT_{w}$'+' (°C)',fontsize=fs7,labelpad=-3)
ax.set_xticks(np.arange(0, 25, 6))  
ax.set_ylim((-1.8,1.8))   
ax.tick_params(axis='both',direction='in', labelsize= fs7) 
plt.legend(prop={'size': fs6},loc='upper right')
plt.text(-4.1,1.86,'b',horizontalalignment='right',verticalalignment='bottom',fontsize=fs8,weight='bold')         
ax.set_xlim((-0.5,24.5))   

ax = fig.add_subplot(spec5[0,2])
Upper=BerlinDiurnal.loc[:,'ObsUHICom']+BerlinDiurnal.loc[:,'ObsUHICom_std']
Lower=BerlinDiurnal.loc[:,'ObsUHICom']-BerlinDiurnal.loc[:,'ObsUHICom_std']
plt.fill_between(time,Lower, Upper, alpha=0.2,color='red')    
plt.plot(time,BerlinDiurnal.loc[:,'ModelUHICom'],color='black', label="Modelled", linewidth=linew)
plt.plot(time,BerlinDiurnal.loc[:,'ObsUHICom'],color='Red', linestyle='dashed', label="Observed", linewidth=linew)
ax.set_ylabel('UHI component (°C)',fontsize=fs7,labelpad=-0)
ax.set_xticks(np.arange(0, 25, 6))  
ax.set_ylim((-1.8,1.8))   
ax.tick_params(axis='both',direction='in', labelsize= fs7) 
plt.legend(prop={'size': fs6},loc='lower center')    
plt.text(-4.1,1.86,'c',horizontalalignment='right',verticalalignment='bottom',fontsize=fs8,weight='bold')         
ax.set_xlim((-0.5,24.5))   

ax = fig.add_subplot(spec5[0,3])
Upper=BerlinDiurnal.loc[:,'ObsUDICom']+BerlinDiurnal.loc[:,'ObsUDICom_std']
Lower=BerlinDiurnal.loc[:,'ObsUDICom']-BerlinDiurnal.loc[:,'ObsUDICom_std']
plt.fill_between(time,Lower, Upper, alpha=0.2,color='red')    
plt.plot(time,BerlinDiurnal.loc[:,'ModelUDICom'],color='black', label="Modelled", linewidth=linew)
plt.plot(time,BerlinDiurnal.loc[:,'ObsUDICom'],color='Red', linestyle='dashed', label="Observed", linewidth=linew)
ax.set_ylabel('UDI component (°C)',fontsize=fs7,labelpad=-0)
ax.set_xticks(np.arange(0, 25, 6))  
ax.set_ylim((-1.8,1.8))   
ax.tick_params(axis='both',direction='in', labelsize= fs7) 
plt.legend(prop={'size': fs6})     
plt.text(-4.1,1.86,'d',horizontalalignment='right',verticalalignment='bottom',fontsize=fs8,weight='bold')         
ax.set_xlim((-0.5,24.5))   

ax = fig.add_subplot(spec5[1,0])    
plt.plot(time,PhoenixDiurnal.loc[:,'ModelTw_U'],color='Red', label="Modelled urban "+r'$T_{w}$', linewidth=linew)
plt.plot(time,PhoenixDiurnal.loc[:,'ObsTw_U'],color='Red', linestyle='dashed', label="Observed urban "+r'$T_{w}$', linewidth=linew)
plt.plot(time,PhoenixDiurnal.loc[:,'ModelTw_R'],color='blue', label="Modelled rural "+r'$T_{w}$', linewidth=linew)
plt.plot(time,PhoenixDiurnal.loc[:,'ObsTw_R'],color='blue', linestyle='dashed', label="Observed rural "+r'$T_{w}$', linewidth=linew)
ax.set_ylabel(r'$T_{w}$'+' (°C)',fontsize=fs7,labelpad=0)
ax.set_xlabel('Local time',fontsize=fs7,labelpad=0)
ax.set_xticks(np.arange(0, 25, 6))  
ax.set_ylim((15,30))  ;ax.set_yticks(np.arange(15, 31, 5))  
plt.text(0,30.2,'Phoenix',horizontalalignment='left',verticalalignment='bottom',fontsize=fs7,weight='bold') 
ax.tick_params(axis='both',direction='in', labelsize= fs7) 
plt.legend(prop={'size': 5.8},ncol=1,loc='upper left')#bbox_to_anchor=(0.67,0.86),frameon=False,
plt.text(-4.1,30.4,'e',horizontalalignment='right',verticalalignment='bottom',fontsize=fs8,weight='bold')         
ax.set_xlim((-0.5,24.5))   

ax = fig.add_subplot(spec5[1,1])
Upper=PhoenixDiurnal.loc[:,'ObsTw_U']-PhoenixDiurnal.loc[:,'ObsTw_R']+PhoenixDiurnal.loc[:,'ObsDeltaTw_std']
Lower=PhoenixDiurnal.loc[:,'ObsTw_U']-PhoenixDiurnal.loc[:,'ObsTw_R']-PhoenixDiurnal.loc[:,'ObsDeltaTw_std']
plt.fill_between(time,Lower, Upper, alpha=0.2,color='red')    
plt.plot(time,PhoenixDiurnal.loc[:,'ModelTw_U']-PhoenixDiurnal.loc[:,'ModelTw_R'],color='black', label="Modelled", linewidth=linew)
plt.plot(time,PhoenixDiurnal.loc[:,'ObsTw_U']-PhoenixDiurnal.loc[:,'ObsTw_R'],color='Red', linestyle='dashed', linewidth=linew, label="Observed")
ax.set_ylabel(r'$ΔT_{w}$'+' (°C)',fontsize=fs7,labelpad=-3)
ax.set_xlabel('Local time',fontsize=fs7,labelpad=0)
ax.set_xticks(np.arange(0, 25, 6))  
ax.set_ylim((-1.8,1.8))   
ax.tick_params(axis='both',direction='in', labelsize=fs7) 
plt.legend(prop={'size': fs6})
plt.text(-4.1,1.86,'f',horizontalalignment='right',verticalalignment='bottom',fontsize=fs8,weight='bold')         
ax.set_xlim((-0.5,24.5))   

ax = fig.add_subplot(spec5[1,2])
Upper=PhoenixDiurnal.loc[:,'ObsUHICom']+PhoenixDiurnal.loc[:,'ObsUHICom_std']
Lower=PhoenixDiurnal.loc[:,'ObsUHICom']-PhoenixDiurnal.loc[:,'ObsUHICom_std']
plt.fill_between(time,Lower, Upper, alpha=0.2,color='red')    
plt.plot(time,PhoenixDiurnal.loc[:,'ModelUHICom'],color='black', label="Modelled", linewidth=linew)
plt.plot(time,PhoenixDiurnal.loc[:,'ObsUHICom'],color='Red', linestyle='dashed', label="Observed", linewidth=linew)
ax.set_ylabel('UHI component (°C)',fontsize=fs7,labelpad=-0)
ax.set_xlabel('Local time',fontsize=fs7,labelpad=0)
ax.set_xticks(np.arange(0, 25, 6))  
ax.set_ylim((-1.8,1.8))   
ax.tick_params(axis='both',direction='in', labelsize= fs7) 
plt.legend(prop={'size': fs6},loc='lower center')    
plt.text(-4.1,1.86,'g',horizontalalignment='right',verticalalignment='bottom',fontsize=fs8,weight='bold')         
ax.set_xlim((-0.5,24.5))   

ax = fig.add_subplot(spec5[1,3])
Upper=PhoenixDiurnal.loc[:,'ObsUDICom']+PhoenixDiurnal.loc[:,'ObsUDICom_std']
Lower=PhoenixDiurnal.loc[:,'ObsUDICom']-PhoenixDiurnal.loc[:,'ObsUDICom_std']
plt.fill_between(time,Lower, Upper, alpha=0.2,color='red')    
plt.plot(time,PhoenixDiurnal.loc[:,'ModelUDICom'],color='black', label="Modelled", linewidth=linew)
plt.plot(time,PhoenixDiurnal.loc[:,'ObsUDICom'],color='Red', linestyle='dashed', label="Observed", linewidth=linew)
ax.set_ylabel('UDI component (°C)',fontsize=fs7,labelpad=-0)
ax.set_xlabel('Local time',fontsize=fs7,labelpad=0)
ax.set_xticks(np.arange(0, 25, 6))  
ax.set_ylim((-1.8,1.8))   
ax.tick_params(axis='both',direction='in', labelsize= fs7) 
plt.legend(prop={'size': fs6})     
plt.text(-4.1,1.86,'h',horizontalalignment='right',verticalalignment='bottom',fontsize=fs8,weight='bold')         
ax.set_xlim((-0.5,24.5))   


plt.subplots_adjust(top=0.95,
bottom=0.1,
left=0.06,
right=0.99,
hspace=0.37,
wspace=0.31) 
# plt.savefig(FilePath+'EDfig6.jpg', dpi=1200)
# plt.close()

