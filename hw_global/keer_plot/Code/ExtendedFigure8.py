"""
Generate ED Figure 8
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
from matplotlib.patches import Patch

FilePath='B:\\HeatStressPaper2022_v2\\Revision2\\Upload_20221130\\Data\\'

# Load observational daytime/nighttime results
ModelWet=np.array(pd.read_csv(FilePath+'EDFigure8_model_wet.csv'))
ModelInter=np.array(pd.read_csv(FilePath+'EDFigure8_model_inter.csv'))
ModelDry=np.array(pd.read_csv(FilePath+'EDFigure8_model_dry.csv'))

ObsWet=np.array(pd.read_csv(FilePath+'EDFigure8_obs_wet.csv'))
ObsInter=np.array(pd.read_csv(FilePath+'EDFigure8_obs_inter.csv'))
ObsDry=np.array(pd.read_csv(FilePath+'EDFigure8_obs_dry.csv'))

Twcolor='black';Tacolor='#CC0000'; Eacolor='#3366FF'#dodgerblue
fs8=8;fs7=7;fs6=6;fs65=6.5
ms1=1;ms2=2;linew=0.7
# set the font globally
plt.rcParams.update({'font.sans-serif':'Arial'})

"""
BarPlot
Description: a function to plot ED figure 8
""" 
def BoxPlot(Tw,Ta,Ea,Num,PlotNum):
    Wid=0.15 # define cap size and bar width
    labels = ['Wet','Intermediate','Dry']
    x = np.arange(len(labels))*0.8
    w1=5;w2=95

    c=Twcolor;a=1 # define the box plot properties
    flierprops = dict(marker=".",markersize=ms1,alpha=a,color=c,markeredgecolor=c)
    boxprops = dict( linewidth=linew, color=c);whiskerprops=dict( linewidth=linew, color=c)
    medianprops = dict( linewidth=linew, color='black');capprops=dict( linewidth=linew, color=c)
    meanprops= dict(marker='x',markersize=ms2,markeredgewidth=linew,color=c,markeredgecolor=c,markerfacecolor='none')
    ax.boxplot(Tw,positions=x+0.0,whis=(w1, w2),widths=Wid\
               , showmeans=True,meanprops=meanprops,flierprops=flierprops,capprops=capprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops)
    
    c=Tacolor; # define the box plot properties
    flierprops = dict(marker=".",markersize=ms1,alpha=a,color=c,markeredgecolor=c)
    boxprops = dict( linewidth=linew, color=c);whiskerprops=dict( linewidth=linew, color=c)
    medianprops = dict( linewidth=linew, color='black');capprops=dict( linewidth=linew, color=c)
    meanprops= dict(marker='x',markersize=ms2,markeredgewidth=linew,color=c,markeredgecolor=c,markerfacecolor='none')
    ax.boxplot(Ta,positions=x+0.2,whis=(w1, w2),widths=Wid\
               , showmeans=True,meanprops=meanprops,flierprops=flierprops,capprops=capprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops)
    
    c=Eacolor; # define the box plot properties royalblue
    flierprops = dict(marker=".",markersize=ms1,alpha=a,color=c,markeredgecolor=c)
    boxprops = dict( linewidth=linew, color=c);whiskerprops=dict( linewidth=linew, color=c)
    medianprops = dict( linewidth=linew, color='black');capprops=dict( linewidth=linew, color=c)
    meanprops= dict(marker='x',markersize=ms2,markeredgewidth=linew,color=c,markeredgecolor=c,markerfacecolor='none')
    ax.boxplot(Ea,positions=x+0.4,whis=(w1, w2),widths=Wid\
               , showmeans=True,meanprops=meanprops,flierprops=flierprops,capprops=capprops,medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops)

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    ax.axhline(y=0,linewidth=linew,color = 'k')
    
    ax.set_yticks(np.arange(-0.7, 1.5, 0.7))    
    ax.set_ylabel(r'$ΔT_{w}$'+' (°C)',fontsize=fs7,labelpad=-0)
    
    ticks_loc = ax.get_xticks().tolist()
    ax.set_xticks(ax.get_xticks().tolist())     # to use set_xticklabels we have to set xtick loc first
    if Num in [1,2]:
        ax.set_xticklabels([' ']*len(ticks_loc),fontsize=fs7,fontweight='bold')#
        ax.tick_params(axis='x',bottom=False)
        ax.set_ylim((-1.8,1.8)) 
        b=np.arange(ax.get_xlim()[0]+0.18,ax.get_xlim()[1]*0.94,0.1)
        offset=-1.8
        plt.plot(b, b*0+offset,'k',linewidth=1.6)
        plt.scatter([0.2,1,1.8], [offset+0.08,offset+0.08,offset+0.08],marker="^",s=14,c='k')
        plt.text(0.2,offset-0.06,'wet',horizontalalignment='center',verticalalignment='top',fontsize=fs7)        
        plt.text(1,offset-0.06,'inter.',horizontalalignment='center',verticalalignment='top',fontsize=fs7) 
        plt.text(1.8,offset-0.06,'dry',horizontalalignment='center',verticalalignment='top',fontsize=fs7)  
        if Num ==1:
            plt.text(0.2,offset+0.4,'17',horizontalalignment='center',verticalalignment='top',fontsize=fs6)         
            plt.text(1,offset+0.4,'101',horizontalalignment='center',verticalalignment='top',fontsize=fs6)                 
            plt.text(1.8,offset+0.4,'15',horizontalalignment='center',verticalalignment='top',fontsize=fs6)         
    plt.text(-0.5,2,PlotNum,horizontalalignment='center',verticalalignment='top',fontsize=fs8,weight='bold')         
    ax.set_xlim((-0.3,2.3))

    ax.tick_params(axis='y',direction='in', labelsize= fs7)      
    

# fig = plt.figure(figsize=(10,8))

cm = 1/2.54 
fig = plt.figure(figsize=(12*cm, 4.5*cm))
widths = [1,1]
# heights = [1,1.2]
spec5 = fig.add_gridspec(ncols=2,nrows=1, width_ratios=widths)
#plt.clf()
ax = fig.add_subplot(spec5[0])
DataTw=np.array([ObsWet[:,1],ObsInter[:,1],ObsDry[:,1]]).tolist()
DataTa=np.array([ObsWet[:,2],ObsInter[:,2],ObsDry[:,2]]).tolist()
DataEa=np.array([ObsWet[:,3],ObsInter[:,3],ObsDry[:,3]]).tolist()
BoxPlot(DataTw,DataTa,DataEa,1,'a')

ax = fig.add_subplot(spec5[1])
DataTw=np.array([ModelWet[:,0],ModelInter[:,0],ModelDry[:,0]]).tolist()
DataTa=np.array([ModelWet[:,1],ModelInter[:,1],ModelDry[:,1]]).tolist()
DataEa=np.array([ModelWet[:,2],ModelInter[:,2],ModelDry[:,2]]).tolist()
BoxPlot(DataTw,DataTa,DataEa,2,'b')

handles, labels = ax.get_legend_handles_labels()
# Create a Rectangle patch
rect = patches.Rectangle((-0.8, -0.2), 0.1, 0.5, linewidth=1, edgecolor='k', facecolor='k')
fig.patches.extend([plt.Rectangle((0.06,0.1),0.03,0.1,
                                  fill=True, color='white',#
                                  transform=fig.transFigure, figure=fig)]) #
fig.patches.extend([plt.Rectangle((0.49,0.1),0.03,0.1,
                                  fill=True, color='white',
                                  transform=fig.transFigure, figure=fig)])
legend_elementsDTw = [Patch(facecolor=Twcolor, edgecolor='k',label=r'$ΔT_{wmax}$'),
Patch(facecolor=Tacolor, edgecolor='k',label='UHI component'),
Patch(facecolor=Eacolor, edgecolor='k',label='UDI component')
] 

fig.legend(handles=legend_elementsDTw,bbox_to_anchor=(0.8,0.5),frameon=False,loc='center left', prop={'size': fs6})

plt.subplots_adjust(top=0.9,
bottom=0.12,
left=0.08,
right=0.79,
hspace=0.315,
wspace=0.5)
# plt.savefig(FilePath+'EDfig8.jpg', dpi=1200)
