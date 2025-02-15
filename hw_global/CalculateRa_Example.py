import netCDF4 as nc 
import numpy as np
import gc

# constants
Lv=2466# J g-1 
Cp=1004 # J/(kg·K)
density=1.184 # kg/m3
SBConstant=5.67*1e-8# the Stefan Boltzmann constant = 5.670367 × 10-8 kg s-3 K-4
gamma=0.66 # hpa/C

def CalculateRc(CUTa,CUTBOT,CUFSH,CURH,CUVP,CUFSA,CUFIRA,CUFGR,CULE,DUTa,DUTBOT,DUFSH,DURH,DUVP,DUFSA,DUFIRA,DUFGR,DULE):
#------------Calculate Ra -----------------------#
    CURa=density*Cp*(CUTa-CUTBOT)/CUFSH
    DURa=density*Cp*(DUTa-DUTBOT)/DUFSH
    Thre_ra0=0;Thre_ra1=800
    CURa=np.ma.masked_where((CURa<Thre_ra0) | (CURa>Thre_ra1),CURa)
    DURa=np.ma.masked_where((DURa<Thre_ra0) | (DURa>Thre_ra1),DURa)
#------------Calculate Ra -----------------------#
    # Calculate VPD hPa
    CUVPD=(1/(CURH*0.01)-1)*CUVP*0.01
    DUVPD=(1/(DURH*0.01)-1)*DUVP*0.01
    # delta
    CUSlope = 17.27*237.3*(6.108*(np.exp((CUTa-273.15)*17.27/(237.3+CUTa-273.15))))/((237.3+CUTa-273.15)**2.0)# hPa K-1
    DUSlope = 17.27*237.3*(6.108*(np.exp((DUTa-273.15)*17.27/(237.3+DUTa-273.15))))/((237.3+DUTa-273.15)**2.0)# hPa K-1
    
    # Calculate Canopy Res
    # CURc=((CUSlope*(CUFSA-CUFIRA-CUFGR)+density*Cp*CUVPD/CURa)/CULE - CUSlope)/(gamma)*CURa-CURa
    # DURc=((DUSlope*(DUFSA-DUFIRA-DUFGR)+density*Cp*DUVPD/DURa)/DULE - DUSlope)/(gamma)*DURa-DURa
    CURc=(CURa*CUSlope*(CUFSA-CUFIRA-CUFGR)+density*Cp*CUVPD-CUSlope*CULE*CURa-gamma*CURa*CULE)/(CULE*gamma)
    DURc=(DURa*DUSlope*(DUFSA-DUFIRA-DUFGR)+density*Cp*DUVPD-DUSlope*DULE*DURa-gamma*DURa*DULE)/(DULE*gamma)
    
    CUcon=1/CURc*1000
    DUcon=1/DURc*1000
    # CUcon=CUcon[:,34:168,:]
    # DUcon=DUcon[:,34:168,:]
    return CUcon,DUcon
