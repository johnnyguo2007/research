"""
Generate Figure 1
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
from matplotlib.patches import Patch
FilePath='B:\\HeatStressPaper2022_v2\\Revision2\\Upload_20221130\\Data\\'

# Load observational daytime/nighttime results
ModelWet=pd.read_csv(FilePath+'Fig1EDFig9_model_wet.csv')
ModelInter=pd.read_csv(FilePath+'Fig1EDFig9_model_inter.csv')
ModelDry=pd.read_csv(FilePath+'Fig1EDFig9_model_dry.csv')

ObsWet=pd.read_csv(FilePath+'Fig1EDFig9_obs_wet.csv')
ObsInter=pd.read_csv(FilePath+'Fig1EDFig9_obs_inter.csv')
ObsDry=pd.read_csv(FilePath+'Fig1EDFig9_obs_dry.csv')

ModelWetDay=np.array(ModelWet[ModelWet['DN']=='Day'])
ModelInterDay=np.array(ModelInter[ModelInter['DN']=='Day'])
ModelDryDay=np.array(ModelDry[ModelDry['DN']=='Day'])
ModelWetNight=np.array(ModelWet[ModelWet['DN']=='Night'])
ModelInterNight=np.array(ModelInter[ModelInter['DN']=='Night'])
ModelDryNight=np.array(ModelDry[ModelDry['DN']=='Night'])

ObsWetDay=np.array(ObsWet[ObsWet['DN']=='Day'])
ObsInterDay=np.array(ObsInter[ObsInter['DN']=='Day'])
ObsDryDay=np.array(ObsDry[ObsDry['DN']=='Day'])
ObsWetNight=np.array(ObsWet[ObsWet['DN']=='Night'])
ObsInterNight=np.array(ObsInter[ObsInter['DN']=='Night'])
ObsDryNight=np.array(ObsDry[ObsDry['DN']=='Night'])

Twcolor='black';Tacolor='#CC0000'; Eacolor='#3366FF'

"""
BoxPlot
Description: a function to plot Figure 1 
""" 
def BoxPlot(Tw,Ta,Ea,Num,PlotNum):
    Wid=0.15 # define box width
    labels = ['Wet','Intermediate','Dry']
    x = np.arange(len(labels))*0.8
    w1=5;w2=95

    c=Twcolor;a=1 # define the box plot properties
    flierprops = dict(marker=".",markersize=2,alpha=a,color=c,markeredgecolor=c)
    boxprops = dict( linewidth=1, color=c);whiskerprops=dict( linewidth=1, color=c)
    medianprops = dict( linewidth=1, color='black');capprops=dict( linewidth=1, color=c)
    meanprops= dict(marker='x',markersize=4,color=c,markeredgecolor=c,markerfacecolor='none')
    ax.boxplot(Tw,positions=x+0.0,whis=(w1, w2),widths=Wid\
               , showmeans=True,meanprops=meanprops,flierprops=flierprops,capprops=capprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops)
   
    c=Tacolor; # define the box plot properties
    flierprops = dict(marker=".",markersize=2,alpha=a,color=c,markeredgecolor=c)
    boxprops = dict( linewidth=1, color=c);whiskerprops=dict( linewidth=1, color=c)
    medianprops = dict( linewidth=1, color='black');capprops=dict( linewidth=1, color=c)
    meanprops= dict(marker='x',markersize=4,color=c,markeredgecolor=c,markerfacecolor='none')
    ax.boxplot(Ta,positions=x+0.2,whis=(w1, w2),widths=Wid\
               , showmeans=True,meanprops=meanprops,flierprops=flierprops,capprops=capprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops)
    
    c=Eacolor; # define the box plot properties royalblue
    flierprops = dict(marker=".",markersize=2,alpha=a,color=c,markeredgecolor=c)
    boxprops = dict( linewidth=1, color=c);whiskerprops=dict( linewidth=1, color=c)
    medianprops = dict( linewidth=1, color='black');capprops=dict( linewidth=1, color=c)
    meanprops= dict(marker='x',markersize=4,color=c,markeredgecolor=c,markerfacecolor='none')
    ax.boxplot(Ea,positions=x+0.4,whis=(w1, w2),widths=Wid\
               , showmeans=True,meanprops=meanprops,flierprops=flierprops,capprops=capprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops)

    #ax.set_xlabel('Climate zone',fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    ax.axhline(y=0,linewidth=0.8,color = 'k')
    
    ax.set_yticks(np.arange(-0.7, 1.5, 0.7))    
    ax.set_ylabel(r'$ΔT_{w}$'+' (°C)',fontsize=11,labelpad=-6)
    ticks_loc = ax.get_xticks().tolist()
    ax.set_xticks(ax.get_xticks().tolist())     # to use set_xticklabels we have to set xtick loc first
    if Num in [1,2]:
        ax.set_xticklabels([' ']*len(ticks_loc),fontsize=9,fontweight='bold')#
        ax.tick_params(axis='x',bottom=False)
        ax.set_ylim((-1.3,1.8)) 
    elif Num in [3,4]: #
        ax.set_xticklabels([' ']*len(ticks_loc),fontsize=9,fontweight='bold')#
        ax.tick_params(axis='x',bottom=False)
        ax.set_ylim((-1.8,1.8)) 
        b=np.arange(ax.get_xlim()[0]+0.18,ax.get_xlim()[1]*0.94,0.1)
        offset=-1.6
        plt.plot(b, b*0+offset,'k',linewidth=1.5)
        plt.scatter([0.2,1,1.8], [offset+0.08,offset+0.08,offset+0.08],marker="^",s=14,c='k')
        plt.text(0.2,offset-0.06,'wet',horizontalalignment='center',verticalalignment='top',fontsize=12)        
        plt.text(1,offset-0.06,'inter.',horizontalalignment='center',verticalalignment='top',fontsize=12)   
        plt.text(1.8,offset-0.06,'dry',horizontalalignment='center',verticalalignment='top',fontsize=12)        
        if Num==3:
            plt.text(0.2,offset+0.4,'17',horizontalalignment='center',verticalalignment='top',fontsize=12)    
            plt.text(1,offset+0.4,'101',horizontalalignment='center',verticalalignment='top',fontsize=12)   
            plt.text(1.8,offset+0.4,'15',horizontalalignment='center',verticalalignment='top',fontsize=12)    
    plt.text(-0.5,1.9,PlotNum,horizontalalignment='center',verticalalignment='top',fontsize=12,weight='bold')         
    ax.set_xlim((-0.3,2.3))
        
      
    ax.tick_params(axis='y',direction='in', labelsize= 12)      
    
fig = plt.figure(figsize=(7.6,6.0))
widths = [1,1]
heights = [1,1.2]
spec5 = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths,
                          height_ratios=heights)

ax = fig.add_subplot(spec5[0, 0])

DataTw=np.array([ObsWetDay[:,1],ObsInterDay[:,1],ObsDryDay[:,1]], dtype=object)
DataTa=np.array([ObsWetDay[:,2],ObsInterDay[:,2],ObsDryDay[:,2]], dtype=object)
DataEa=np.array([ObsWetDay[:,3],ObsInterDay[:,3],ObsDryDay[:,3]], dtype=object)
BoxPlot(DataTw,DataTa,DataEa,1,'a')

ax = fig.add_subplot(spec5[0, 1])
DataTw=np.array([ModelWetDay[:,0],ModelInterDay[:,0],ModelDryDay[:,0]], dtype=object)
DataTa=np.array([ModelWetDay[:,1],ModelInterDay[:,1],ModelDryDay[:,1]], dtype=object)
DataEa=np.array([ModelWetDay[:,2],ModelInterDay[:,2],ModelDryDay[:,2]], dtype=object)
BoxPlot(DataTw,DataTa,DataEa,2,'b')

ax = fig.add_subplot(spec5[1, 0])
DataTw=np.array([ObsWetNight[:,1],ObsInterNight[:,1],ObsDryNight[:,1]], dtype=object)
DataTa=np.array([ObsWetNight[:,2],ObsInterNight[:,2],ObsDryNight[:,2]], dtype=object)
DataEa=np.array([ObsWetNight[:,3],ObsInterNight[:,3],ObsDryNight[:,3]], dtype=object)
BoxPlot(DataTw,DataTa,DataEa,3,'c')

ax = fig.add_subplot(spec5[1, 1])
DataTw=np.array([ModelWetNight[:,0],ModelInterNight[:,0],ModelDryNight[:,0]], dtype=object)
DataTa=np.array([ModelWetNight[:,1],ModelInterNight[:,1],ModelDryNight[:,1]], dtype=object)
DataEa=np.array([ModelWetNight[:,2],ModelInterNight[:,2],ModelDryNight[:,2]], dtype=object)
BoxPlot(DataTw,DataTa,DataEa,4,'d')

handles, labels = ax.get_legend_handles_labels()

rect = patches.Rectangle((-0.8, -0.2), 0.1, 0.5, linewidth=1, edgecolor='k', facecolor='k')
fig.patches.extend([plt.Rectangle((0.085,0.178),0.03,0.06,
                                  fill=True, color='white',
                                  transform=fig.transFigure, figure=fig)]) #
fig.patches.extend([plt.Rectangle((0.47,0.178),0.03,0.06,
                                  fill=True, color='white',
                                  transform=fig.transFigure, figure=fig)])

legend_elementsDTw = [Patch(facecolor=Twcolor, edgecolor='k',label=r'$ΔT_{w}$'),
Patch(facecolor=Tacolor, edgecolor='k',label='UHI component'),
Patch(facecolor=Eacolor, edgecolor='k',label='UDI component')
] 

fig.legend(handles=legend_elementsDTw,bbox_to_anchor=(0.78,0.5),frameon=False,loc='center left', prop={'size': 10})

plt.subplots_adjust(top=0.805,
bottom=0.195,
left=0.1,
right=0.775,
hspace=0.315,
wspace=0.36)
# plt.savefig(FilePath+'\\Figure1.png', dpi=600)
# plt.savefig(FilePath+'\\Figure1.eps')
# plt.close()
