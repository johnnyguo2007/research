"""
Generate ED Figure 9
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
import random
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

Samplesize=1000
def Resample(Data,ModelObs,HighLowMed):
    if ModelObs=='Obs':
        Tw = Data[:,1].tolist()
        Ta = Data[:,2].tolist()
        Ea = Data[:,3].tolist()
    elif ModelObs=='Model':
        Tw = Data[:,0].tolist()
        Ta = Data[:,1].tolist()
        Ea = Data[:,2].tolist()
    if HighLowMed=='High':
        m=12 #resample size is 75% of original data size
    elif HighLowMed=='Med':
        m=61
    elif HighLowMed=='Low':
        m=10       
    RandomSampleTw=np.zeros(Samplesize)
    RandomSampleTa=np.zeros(Samplesize)
    RandomSampleEa=np.zeros(Samplesize)
    for i in range(0,Samplesize): # repeat resample for 1000 times
        random.seed(i)
        Tw_sub, Ta_sub, Ea_sub = zip(*random.sample(list(zip(Tw, Ta, Ea)),m))
        RandomSampleTw[i]=np.mean(Tw_sub)
        RandomSampleTa[i]=np.mean(Ta_sub)
        RandomSampleEa[i]=np.mean(Ea_sub)
    return RandomSampleTw,RandomSampleTa,RandomSampleEa 

[RandomSampleTwObsHighDay,RandomSampleTaObsHighDay,RandomSampleEaObsHighDay]=Resample(ObsWetDay,'Obs','High')
[RandomSampleTwObsMedDay,RandomSampleTaObsMedDay,RandomSampleEaObsMedDay]=Resample(ObsInterDay,'Obs','Med')
[RandomSampleTwObsLowDay,RandomSampleTaObsLowDay,RandomSampleEaObsLowDay]=Resample(ObsDryDay,'Obs','Low')

[RandomSampleTwObsHighNight,RandomSampleTaObsHighNight,RandomSampleEaObsHighNight]=Resample(ObsWetNight,'Obs','High')
[RandomSampleTwObsMedNight,RandomSampleTaObsMedNight,RandomSampleEaObsMedNight]=Resample(ObsInterNight,'Obs','Med')
[RandomSampleTwObsLowNight,RandomSampleTaObsLowNight,RandomSampleEaObsLowNight]=Resample(ObsDryNight,'Obs','Low')

[RandomSampleTwModelHighDay,RandomSampleTaModelHighDay,RandomSampleEaModelHighDay]=Resample(ModelWetDay,'Model','High')
[RandomSampleTwModelMedDay,RandomSampleTaModelMedDay,RandomSampleEaModelMedDay]=Resample(ModelInterDay,'Model','Med')
[RandomSampleTwModelLowDay,RandomSampleTaModelLowDay,RandomSampleEaModelLowDay]=Resample(ModelDryDay,'Model','Low')

[RandomSampleTwModelHighNight,RandomSampleTaModelHighNight,RandomSampleEaModelHighNight]=Resample(ModelWetNight,'Model','High')
[RandomSampleTwModelMedNight,RandomSampleTaModelMedNight,RandomSampleEaModelMedNight]=Resample(ModelInterNight,'Model','Med')
[RandomSampleTwModelLowNight,RandomSampleTaModelLowNight,RandomSampleEaModelLowNight]=Resample(ModelDryNight,'Model','Low')  

"""
BarPlot
Description: a function to plot ED Figure 9 
""" 
Twcolor='black';Tacolor='#CC0000'; Eacolor='#3366FF'#dodgerblue
fs8=8;fs7=7;fs6=6;fs65=6.5
ms1=1;ms2=2;linew=0.7
# set the font globally
plt.rcParams.update({'font.sans-serif':'Arial'})

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
    
    ax.set_yticks(np.arange(-0.3, 1.0, 0.3))    
    ax.set_ylabel(r'$ΔT_{w}$'+' (°C)',fontsize=fs7,labelpad=-1)
    ticks_loc = ax.get_xticks().tolist()
    ax.set_xticks(ax.get_xticks().tolist())     # to use set_xticklabels we have to set xtick loc first
    if Num in [1,2]:
        ax.set_xticklabels([' ']*len(ticks_loc),fontsize=fs7,fontweight='bold')#
        ax.tick_params(axis='x',bottom=False)
        ax.set_ylim((-0.6,0.85)) 
    elif Num in [3,4]:
        ax.set_xticklabels([' ']*len(ticks_loc),fontsize=fs7,fontweight='bold')#
        ax.tick_params(axis='x',bottom=False)
        ax.set_ylim((-0.6,0.85)) 
        b=np.arange(ax.get_xlim()[0]+0.18,ax.get_xlim()[1]*0.94,0.1)
        offset=-0.45
        plt.plot(b, b*0+offset,'k',linewidth=1)
        plt.scatter([0.2,1,1.8], [offset+0.03,offset+0.03,offset+0.03],marker="^",s=12,c='k')
        plt.text(0.2,offset-0.06,'Wet',horizontalalignment='center',verticalalignment='top',fontsize=fs7)        
        plt.text(1,offset-0.06,'Inter.',horizontalalignment='center',verticalalignment='top',fontsize=fs7)   
        plt.text(1.8,offset-0.06,'Dry',horizontalalignment='center',verticalalignment='top',fontsize=fs7)        
        
    plt.text(-0.5,1,PlotNum,horizontalalignment='center',verticalalignment='top',fontsize=fs8,weight='bold')         
    ax.set_xlim((-0.3,2.3))
        
      
    ax.tick_params(axis='y',direction='in', labelsize= fs7)      
    
cm = 1/2.54 
fig = plt.figure(figsize=(12*cm, 9.47368*cm))
widths = [1,1]
heights = [1,1.2]
spec5 = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths,
                          height_ratios=heights)
ax = fig.add_subplot(spec5[0, 0])
DataTw=np.array([RandomSampleTwObsHighDay[:],RandomSampleTwObsMedDay[:],RandomSampleTwObsLowDay[:]]).tolist()
DataTa=np.array([RandomSampleTaObsHighDay[:],RandomSampleTaObsMedDay[:],RandomSampleTaObsLowDay[:]]).tolist()
DataEa=np.array([RandomSampleEaObsHighDay[:],RandomSampleEaObsMedDay[:],RandomSampleEaObsLowDay[:]]).tolist()
BoxPlot(DataTw,DataTa,DataEa,1,'a')

ax = fig.add_subplot(spec5[0, 1])
DataTw=np.array([RandomSampleTwModelHighDay[:],RandomSampleTwModelMedDay[:],RandomSampleTwModelLowDay[:]]).tolist()
DataTa=np.array([RandomSampleTaModelHighDay[:],RandomSampleTaModelMedDay[:],RandomSampleTaModelLowDay[:]]).tolist()
DataEa=np.array([RandomSampleEaModelHighDay[:],RandomSampleEaModelMedDay[:],RandomSampleEaModelLowDay[:]]).tolist()
BoxPlot(DataTw,DataTa,DataEa,2,'b')

ax = fig.add_subplot(spec5[1, 0])
DataTw=np.array([RandomSampleTwObsHighNight[:],RandomSampleTwObsMedNight[:],RandomSampleTwObsLowNight[:]]).tolist()
DataTa=np.array([RandomSampleTaObsHighNight[:],RandomSampleTaObsMedNight[:],RandomSampleTaObsLowNight[:]]).tolist()
DataEa=np.array([RandomSampleEaObsHighNight[:],RandomSampleEaObsMedNight[:],RandomSampleEaObsLowNight[:]]).tolist()
BoxPlot(DataTw,DataTa,DataEa,3,'c')

ax = fig.add_subplot(spec5[1, 1])
DataTw=np.array([RandomSampleTwModelHighNight[:],RandomSampleTwModelMedNight[:],RandomSampleTwModelLowNight[:]]).tolist()
DataTa=np.array([RandomSampleTaModelHighNight[:],RandomSampleTaModelMedNight[:],RandomSampleTaModelLowNight[:]]).tolist()
DataEa=np.array([RandomSampleEaModelHighNight[:],RandomSampleEaModelMedNight[:],RandomSampleEaModelLowNight[:]]).tolist()
BoxPlot(DataTw,DataTa,DataEa,4,'d')

handles, labels = ax.get_legend_handles_labels()
# Create a Rectangle patch
rect = patches.Rectangle((-0.8, -0.2), 0.1, 0.5, linewidth=1, edgecolor='k', facecolor='k')
fig.patches.extend([plt.Rectangle((0.085,0.1),0.03,0.06,
                                  fill=True, color='white',
                                  transform=fig.transFigure, figure=fig)]) #
fig.patches.extend([plt.Rectangle((0.503,0.1),0.03,0.06,
                                  fill=True, color='white',
                                  transform=fig.transFigure, figure=fig)])
legend_elementsDTw = [Patch(facecolor=Twcolor, edgecolor='k',label=r'$ΔT_{w}$'),
Patch(facecolor=Tacolor, edgecolor='k',label='UHI component'),
Patch(facecolor=Eacolor, edgecolor='k',label='UDI component')
] 

fig.legend(handles=legend_elementsDTw,bbox_to_anchor=(0.8,0.5),frameon=False,loc='center left', prop={'size': fs6})

plt.subplots_adjust(top=0.92,
bottom=0.1,
left=0.095,
right=0.8,
hspace=0.315,
wspace=0.5)

# plt.savefig(FilePath+'EDfig9.jpg', dpi=1200)
