module SoilFluxesMod

  !-----------------------------------------------------------------------
  ! !DESCRIPTION:
  ! Updates surface fluxes based on the new ground temperature.
  !
  ! !USES:
  use shr_kind_mod	, only : r8 => shr_kind_r8
  use shr_log_mod	, only : errMsg => shr_log_errMsg
  use decompMod		, only : bounds_type
  use abortutils	, only : endrun
  use perf_mod		, only : t_startf, t_stopf
  use clm_varctl	, only : iulog
  use clm_varpar	, only : nlevsno, nlevgrnd, nlevurb, max_patch_per_col
  use atm2lndType	, only : atm2lnd_type
  use CanopyStateType   , only : canopystate_type
  use EnergyFluxType    , only : energyflux_type
  use SolarAbsorbedType , only : solarabs_type
  use TemperatureType   , only : temperature_type
  use WaterstateType    , only : waterstate_type
  use WaterfluxType     , only : waterflux_type
  use LandunitType	, only : lun                
  use ColumnType	, only : col                
  use PatchType		, only : patch                
  !
  ! !PUBLIC TYPES:
  implicit none
  save
  !
  ! !PUBLIC MEMBER FUNCTIONS:
  public :: SoilFluxes   ! Calculate soil/snow and ground temperatures
  !-----------------------------------------------------------------------

contains

  !-----------------------------------------------------------------------
  subroutine SoilFluxes (bounds, num_urbanl, filter_urbanl, &
       num_urbanp, filter_urbanp, &
       num_nolakec, filter_nolakec, num_nolakep, filter_nolakep, &
       atm2lnd_inst, solarabs_inst, temperature_inst, canopystate_inst, &
       waterstate_inst, energyflux_inst, waterflux_inst)            
    !
    ! !DESCRIPTION:
    ! Update surface fluxes based on the new ground temperature
    !
    ! !USES:
    use clm_time_manager , only : get_step_size
    use clm_varcon       , only : hvap, cpair, grav, vkc, tfrz, sb 
    use landunit_varcon  , only : istsoil, istcrop
    use column_varcon    , only : icol_roof, icol_sunwall, icol_shadewall, icol_road_perv
    use subgridAveMod    , only : p2c
    !
    ! !ARGUMENTS:
    type(bounds_type)      , intent(in)    :: bounds    
    integer                , intent(in)    :: num_nolakec                      ! number of column non-lake points in column filter
    integer                , intent(in)    :: filter_nolakec(:)                ! column filter for non-lake points
    integer                , intent(in)    :: num_urbanl                       ! number of urban landunits in clump
    integer                , intent(in)    :: filter_urbanl(:)                 ! urban landunit filter
    integer                , intent(in)    :: num_urbanp                       ! number of urban pfts in clump
    integer                , intent(in)    :: filter_urbanp(:)                 ! urban pft filter
    integer                , intent(in)    :: num_nolakep                      ! number of column non-lake points in pft filter
    integer                , intent(in)    :: filter_nolakep(:)                ! patch filter for non-lake points
    type(atm2lnd_type)     , intent(in)    :: atm2lnd_inst
    type(solarabs_type)    , intent(in)    :: solarabs_inst
    type(temperature_type) , intent(in)    :: temperature_inst
    type(canopystate_type) , intent(in)    :: canopystate_inst
    type(waterstate_type)  , intent(in)    :: waterstate_inst
    type(waterflux_type)   , intent(inout) :: waterflux_inst
    type(energyflux_type)  , intent(inout) :: energyflux_inst
    !
    ! !LOCAL VARIABLES:
    integer  :: p,c,g,j,pi,l                                       ! indices
    integer  :: fc,fp                                              ! lake filtered column and pft indices
    real(r8) :: dtime                                              ! land model time step (sec)
    real(r8) :: egsmax(bounds%begc:bounds%endc)                    ! max. evaporation which soil can provide at one time step
    real(r8) :: egirat(bounds%begc:bounds%endc)                    ! ratio of topsoil_evap_tot : egsmax
    real(r8) :: tinc(bounds%begc:bounds%endc)                      ! temperature difference of two time step
    real(r8) :: sumwt(bounds%begc:bounds%endc)                     ! temporary
    real(r8) :: evaprat(bounds%begp:bounds%endp)                   ! ratio of qflx_evap_soi/topsoil_evap_tot
    real(r8) :: save_qflx_evap_soi                                 ! temporary storage for qflx_evap_soi
    real(r8) :: topsoil_evap_tot(bounds%begc:bounds%endc)          ! column-level total evaporation from top soil layer
    real(r8) :: eflx_lwrad_del(bounds%begp:bounds%endp)            ! update due to eflx_lwrad
    real(r8) :: t_grnd0(bounds%begc:bounds%endc)                   ! t_grnd of previous time step
    real(r8) :: lw_grnd
    real(r8) :: fsno_eff
    !-----------------------------------------------------------------------

    associate(                                                                & 
         eflx_h2osfc_to_snow_col => energyflux_inst%eflx_h2osfc_to_snow_col , & ! Input:  [real(r8) (:)   ]  col snow melt to h2osfc heat flux (W/m**2)

         forc_lwrad              => atm2lnd_inst%forc_lwrad_downscaled_col  , & ! Input:  [real(r8) (:)   ]  downward infrared (longwave) radiation (W/m**2)

         frac_veg_nosno          => canopystate_inst%frac_veg_nosno_patch   , & ! Input:  [integer (:)    ]  fraction of veg not covered by snow (0/1 now) [-]

         frac_sno_eff            => waterstate_inst%frac_sno_eff_col        , & ! Input:  [real(r8) (:)   ]  eff. fraction of ground covered by snow (0 to 1)
         frac_h2osfc             => waterstate_inst%frac_h2osfc_col         , & ! Input:  [real(r8) (:)   ]  fraction of ground covered by surface water (0 to 1)
         h2osoi_ice              => waterstate_inst%h2osoi_ice_col          , & ! Input:  [real(r8) (:,:) ]  ice lens (kg/m2) (new)                
         h2osoi_liq              => waterstate_inst%h2osoi_liq_col          , & ! Input:  [real(r8) (:,:) ]  liquid water (kg/m2) (new)            

         sabg_soil               => solarabs_inst%sabg_soil_patch           , & ! Input:  [real(r8) (:)   ]  solar radiation absorbed by soil (W/m**2)
         sabg_snow               => solarabs_inst%sabg_snow_patch           , & ! Input:  [real(r8) (:)   ]  solar radiation absorbed by snow (W/m**2)
         sabg                    => solarabs_inst%sabg_patch                , & ! Input:  [real(r8) (:)   ]  solar radiation absorbed by ground (W/m**2)

         emg                     => temperature_inst%emg_col                , & ! Input:  [real(r8) (:)   ]  ground emissivity                       
!         emv                     => temperature_inst%emv_patch              , & ! Input:  [real(r8) (:)   ]  vegetation emissivity
!         t_veg                   => temperature_inst%t_veg_patch            , & ! Output: [real(r8) (:)   ]  vegetation temperature (Kelvin) 
         t_skin_patch            => temperature_inst%t_skin_patch           , & ! Output: [real(r8) (:)   ]  patch skin temperature (K)
         t_skin_r_patch          => temperature_inst%t_skin_r_patch         , & ! Output: [real(r8) (:)   ]  rural skin temperature (K)
         t_skin_u_patch          => temperature_inst%t_skin_u_patch         , & ! Output: [real(r8) (:)   ]  urban skin temperature (K)
         t_h2osfc                => temperature_inst%t_h2osfc_col           , & ! Input:  [real(r8) (:)   ]  surface water temperature               
         tssbef                  => temperature_inst%t_ssbef_col            , & ! Input:  [real(r8) (:,:) ]  soil/snow temperature before update   
         t_h2osfc_bef            => temperature_inst%t_h2osfc_bef_col       , & ! Input:  [real(r8) (:)   ]  saved surface water temperature         
         t_grnd                  => temperature_inst%t_grnd_col             , & ! Input:  [real(r8) (:)   ]  ground temperature (Kelvin)             
         t_soisno                => temperature_inst%t_soisno_col           , & ! Input:  [real(r8) (:,:) ]  soil temperature (Kelvin)             
         xmf                     => temperature_inst%xmf_col                , & ! Input:  [real(r8) (:)   ]  
         xmf_h2osfc              => temperature_inst%xmf_h2osfc_col         , & ! Input:  [real(r8) (:)   ]  
         fact                    => temperature_inst%fact_col               , & ! Input:  [real(r8) (:)   ]  
         c_h2osfc                => temperature_inst%c_h2osfc_col           , & ! Input:  [real(r8) (:)   ]  

         htvp                    => energyflux_inst%htvp_col                , & ! Input:  [real(r8) (:)   ]  latent heat of vapor of water (or sublimation) [j/kg]
         eflx_building_heat_errsoi=> energyflux_inst%eflx_building_heat_errsoi_col  , & ! Input: [real(r8) (:)] heat flux to interior surface of walls and roof for errsoi check (W m-2)
         eflx_wasteheat_patch    => energyflux_inst%eflx_wasteheat_patch    , & ! Input:  [real(r8) (:)   ]  sensible heat flux from urban heating/cooling sources of waste heat (W/m**2)
         eflx_heat_from_ac_patch => energyflux_inst%eflx_heat_from_ac_patch , & ! Input:  [real(r8) (:)   ]  sensible heat flux put back into canyon due to removal by AC (W/m**2)
         eflx_traffic_patch      => energyflux_inst%eflx_traffic_patch      , & ! Input:  [real(r8) (:)   ]  traffic sensible heat flux (W/m**2)     
         dlrad                   => energyflux_inst%dlrad_patch             , & ! Input:  [real(r8) (:)   ]  downward longwave radiation below the canopy [W/m2]
         ulrad                   => energyflux_inst%ulrad_patch             , & ! Input:  [real(r8) (:)   ]  upward longwave radiation above the canopy [W/m2]
         cgrnds                  => energyflux_inst%cgrnds_patch            , & ! Input:  [real(r8) (:)   ]  deriv, of soil sensible heat flux wrt soil temp [w/m2/k]
         cgrndl                  => energyflux_inst%cgrndl_patch            , & ! Input:  [real(r8) (:)   ]  deriv of soil latent heat flux wrt soil temp [w/m**2/k]
         
         qflx_evap_can           => waterflux_inst%qflx_evap_can_patch      , & ! Output: [real(r8) (:)   ]  evaporation from leaves and stems (mm H2O/s) (+ = to atm)
         qflx_evap_soi           => waterflux_inst%qflx_evap_soi_patch      , & ! Output: [real(r8) (:)   ]  soil evaporation (mm H2O/s) (+ = to atm)
         qflx_evap_veg           => waterflux_inst%qflx_evap_veg_patch      , & ! Output: [real(r8) (:)   ]  vegetation evaporation (mm H2O/s) (+ = to atm)
         qflx_tran_veg           => waterflux_inst%qflx_tran_veg_patch      , & ! Input:  [real(r8) (:)   ]  vegetation transpiration (mm H2O/s) (+ = to atm)
         qflx_evap_tot           => waterflux_inst%qflx_evap_tot_patch      , & ! Output: [real(r8) (:)   ]  qflx_evap_soi + qflx_evap_veg + qflx_tran_veg
         qflx_evap_grnd          => waterflux_inst%qflx_evap_grnd_patch     , & ! Output: [real(r8) (:)   ]  ground surface evaporation rate (mm H2O/s) [+]
         qflx_sub_snow           => waterflux_inst%qflx_sub_snow_patch      , & ! Output: [real(r8) (:)   ]  sublimation rate from snow pack (mm H2O /s) [+]
         qflx_dew_snow           => waterflux_inst%qflx_dew_snow_patch      , & ! Output: [real(r8) (:)   ]  surface dew added to snow pack (mm H2O /s) [+]
         qflx_dew_grnd           => waterflux_inst%qflx_dew_grnd_patch      , & ! Output: [real(r8) (:)   ]  ground surface dew formation (mm H2O /s) [+]
         qflx_ev_snow            => waterflux_inst%qflx_ev_snow_patch       , & ! In/Out: [real(r8) (:)   ]  evaporation flux from snow (mm H2O/s) [+ to atm]
         qflx_ev_soil            => waterflux_inst%qflx_ev_soil_patch       , & ! In/Out: [real(r8) (:)   ]  evaporation flux from soil (mm H2O/s) [+ to atm]
         qflx_ev_h2osfc          => waterflux_inst%qflx_ev_h2osfc_patch     , & ! In/Out: [real(r8) (:)   ]  evaporation flux from soil (mm H2O/s) [+ to atm]
         
         eflx_sh_grnd            => energyflux_inst%eflx_sh_grnd_patch      , & ! Output: [real(r8) (:)   ]  sensible heat flux from ground (W/m**2) [+ to atm]
         eflx_sh_veg             => energyflux_inst%eflx_sh_veg_patch       , & ! Output: [real(r8) (:)   ]  sensible heat flux from leaves (W/m**2) [+ to atm]
         eflx_soil_grnd          => energyflux_inst%eflx_soil_grnd_patch    , & ! Output: [real(r8) (:)   ]  soil heat flux (W/m**2) [+ = into soil] 
         eflx_soil_grnd_u        => energyflux_inst%eflx_soil_grnd_u_patch  , & ! Output: [real(r8) (:)   ]  urban soil heat flux (W/m**2) [+ = into soil]
         eflx_soil_grnd_r        => energyflux_inst%eflx_soil_grnd_r_patch  , & ! Output: [real(r8) (:)   ]  rural soil heat flux (W/m**2) [+ = into soil]
         eflx_sh_tot             => energyflux_inst%eflx_sh_tot_patch       , & ! Output: [real(r8) (:)   ]  total sensible heat flux (W/m**2) [+ to atm]
         eflx_sh_tot_u           => energyflux_inst%eflx_sh_tot_u_patch     , & ! Output: [real(r8) (:)   ]  urban total sensible heat flux (W/m**2) [+ to atm]
         eflx_sh_tot_r           => energyflux_inst%eflx_sh_tot_r_patch     , & ! Output: [real(r8) (:)   ]  rural total sensible heat flux (W/m**2) [+ to atm]
         eflx_lh_tot             => energyflux_inst%eflx_lh_tot_patch       , & ! Output: [real(r8) (:)   ]  total latent heat flux (W/m**2)  [+ to atm]
         eflx_lh_tot_u           => energyflux_inst%eflx_lh_tot_u_patch     , & ! Output: [real(r8) (:)   ]  urban total latent heat flux (W/m**2)  [+ to atm]
         eflx_lh_tot_r           => energyflux_inst%eflx_lh_tot_r_patch     , & ! Output: [real(r8) (:)   ]  rural total latent heat flux (W/m**2)  [+ to atm]
         eflx_lwrad_out          => energyflux_inst%eflx_lwrad_out_patch    , & ! Output: [real(r8) (:)   ]  emitted infrared (longwave) radiation (W/m**2)
         eflx_lwrad_net          => energyflux_inst%eflx_lwrad_net_patch    , & ! Output: [real(r8) (:)   ]  net infrared (longwave) rad (W/m**2) [+ = to atm]
         eflx_lwrad_net_r        => energyflux_inst%eflx_lwrad_net_r_patch  , & ! Output: [real(r8) (:)   ]  rural net infrared (longwave) rad (W/m**2) [+ = to atm]
         eflx_lwrad_out_r        => energyflux_inst%eflx_lwrad_out_r_patch  , & ! Output: [real(r8) (:)   ]  rural emitted infrared (longwave) rad (W/m**2)
         eflx_lwrad_net_u        => energyflux_inst%eflx_lwrad_net_u_patch  , & ! Output: [real(r8) (:)   ]  urban net infrared (longwave) rad (W/m**2) [+ = to atm]
         eflx_lwrad_out_u        => energyflux_inst%eflx_lwrad_out_u_patch  , & ! Output: [real(r8) (:)   ]  urban emitted infrared (longwave) rad (W/m**2)
         eflx_lh_vege            => energyflux_inst%eflx_lh_vege_patch      , & ! Output: [real(r8) (:)   ]  veg evaporation heat flux (W/m**2) [+ to atm]
         eflx_lh_vegt            => energyflux_inst%eflx_lh_vegt_patch      , & ! Output: [real(r8) (:)   ]  veg transpiration heat flux (W/m**2) [+ to atm]
         eflx_lh_grnd            => energyflux_inst%eflx_lh_grnd_patch      , & ! Output: [real(r8) (:)   ]  ground evaporation heat flux (W/m**2) [+ to atm]
         errsoi_col              => energyflux_inst%errsoi_col              , & ! Output: [real(r8) (:)   ]  column-level soil/lake energy conservation error (W/m**2)
         errsoi_patch            => energyflux_inst%errsoi_patch              & ! Output: [real(r8) (:)   ]  patch-level soil/lake energy conservation error (W/m**2)
         )

      ! Get step size

      dtime = get_step_size()

      call t_startf('bgp2_loop_1')
      do fc = 1,num_nolakec
         c = filter_nolakec(fc)
         j = col%snl(c)+1

         ! Calculate difference in soil temperature from last time step, for
         ! flux corrections

         if (col%snl(c) < 0) then
            t_grnd0(c) = frac_sno_eff(c) * tssbef(c,col%snl(c)+1) &
                 + (1 - frac_sno_eff(c) - frac_h2osfc(c)) * tssbef(c,1) &
                 + frac_h2osfc(c) * t_h2osfc_bef(c)
         else
            t_grnd0(c) = (1 - frac_h2osfc(c)) * tssbef(c,1) + frac_h2osfc(c) * t_h2osfc_bef(c)
         endif

         tinc(c) = t_grnd(c) - t_grnd0(c)

         ! Determine ratio of topsoil_evap_tot

         egsmax(c) = (h2osoi_ice(c,j)+h2osoi_liq(c,j)) / dtime

         ! added to trap very small negative soil water,ice

         if (egsmax(c) < 0._r8) then
            egsmax(c) = 0._r8
         end if
      end do

      ! A preliminary pft loop to determine if corrections are required for
      ! excess evaporation from the top soil layer... Includes new logic
      ! to distribute the corrections between patches on the basis of their
      ! evaporative demands.
      ! egirat holds the ratio of demand to availability if demand is
      ! greater than availability, or 1.0 otherwise.
      ! Correct fluxes to present soil temperature

      do fp = 1,num_nolakep
         p = filter_nolakep(fp)
         c = patch%column(p)
         eflx_sh_grnd(p) = eflx_sh_grnd(p) + tinc(c)*cgrnds(p)
         qflx_evap_soi(p) = qflx_evap_soi(p) + tinc(c)*cgrndl(p)

         ! set ev_snow, ev_soil for urban landunits here
         l = patch%landunit(p)
         if (lun%urbpoi(l)) then
            qflx_ev_snow(p) = qflx_evap_soi(p)
            qflx_ev_soil(p) = 0._r8
            qflx_ev_h2osfc(p) = 0._r8
         else
            qflx_ev_snow(p) = qflx_ev_snow(p) + tinc(c)*cgrndl(p)
            qflx_ev_soil(p) = qflx_ev_soil(p) + tinc(c)*cgrndl(p)
            qflx_ev_h2osfc(p) = qflx_ev_h2osfc(p) + tinc(c)*cgrndl(p)
         endif
      end do

      ! Set the column-average qflx_evap_soi as the weighted average over all patches
      ! but only count the patches that are evaporating

      do fc = 1,num_nolakec
         c = filter_nolakec(fc)
         topsoil_evap_tot(c) = 0._r8
         sumwt(c) = 0._r8
      end do

      do pi = 1,max_patch_per_col
         do fc = 1,num_nolakec
            c = filter_nolakec(fc)
            if ( pi <= col%npatches(c) ) then
               p = col%patchi(c) + pi - 1
               if (patch%active(p)) then
                  topsoil_evap_tot(c) = topsoil_evap_tot(c) + qflx_evap_soi(p) * patch%wtcol(p)
               end if
            end if
         end do
      end do
      call t_stopf('bgp2_loop_1')
      call t_startf('bgp2_loop_2')

      ! Calculate ratio for rescaling patch-level fluxes to meet availability

      do fc = 1,num_nolakec
         c = filter_nolakec(fc)
         if (topsoil_evap_tot(c) > egsmax(c)) then
            egirat(c) = (egsmax(c)/topsoil_evap_tot(c))
         else
            egirat(c) = 1.0_r8
         end if
      end do

      do fp = 1,num_nolakep
         p = filter_nolakep(fp)
         c = patch%column(p)
         l = patch%landunit(p)
         g = patch%gridcell(p)
         j = col%snl(c)+1

         ! Correct soil fluxes for possible evaporation in excess of top layer water
         ! excess energy is added to the sensible heat flux from soil

         if (egirat(c) < 1.0_r8) then
            save_qflx_evap_soi = qflx_evap_soi(p)
            qflx_evap_soi(p) = qflx_evap_soi(p) * egirat(c)
            eflx_sh_grnd(p) = eflx_sh_grnd(p) + (save_qflx_evap_soi - qflx_evap_soi(p))*htvp(c)
            qflx_ev_snow(p) = qflx_ev_snow(p) * egirat(c)
            qflx_ev_soil(p) = qflx_ev_soil(p) * egirat(c)
            qflx_ev_h2osfc(p) = qflx_ev_h2osfc(p) * egirat(c)
         end if

         ! Ground heat flux
         
         if (.not. lun%urbpoi(l)) then
            lw_grnd=(frac_sno_eff(c)*tssbef(c,col%snl(c)+1)**4 &
                 +(1._r8-frac_sno_eff(c)-frac_h2osfc(c))*tssbef(c,1)**4 &
                 +frac_h2osfc(c)*t_h2osfc_bef(c)**4)

            eflx_soil_grnd(p) = ((1._r8- frac_sno_eff(c))*sabg_soil(p) + frac_sno_eff(c)*sabg_snow(p)) + dlrad(p) &
                 + (1-frac_veg_nosno(p))*emg(c)*forc_lwrad(c) &
                 - emg(c)*sb*lw_grnd - emg(c)*sb*t_grnd0(c)**3*(4._r8*tinc(c)) &
                 - (eflx_sh_grnd(p)+qflx_evap_soi(p)*htvp(c))

            if (lun%itype(l) == istsoil .or. lun%itype(l) == istcrop) then
               eflx_soil_grnd_r(p) = eflx_soil_grnd(p)
            end if
         else
            ! For all urban columns we use the net longwave radiation (eflx_lwrad_net) since
            ! the term (emg*sb*tssbef(col%snl+1)**4) is not the upward longwave flux because of 
            ! interactions between urban columns.

            eflx_lwrad_del(p) = 4._r8*emg(c)*sb*t_grnd0(c)**3*tinc(c)

            ! Include transpiration term because needed for pervious road
            ! and wasteheat and traffic flux
            eflx_soil_grnd(p) = sabg(p) + dlrad(p) &
                 - eflx_lwrad_net(p) - eflx_lwrad_del(p) &
                 - (eflx_sh_grnd(p) + qflx_evap_soi(p)*htvp(c) + qflx_tran_veg(p)*hvap) &
                 + eflx_wasteheat_patch(p) + eflx_heat_from_ac_patch(p) + eflx_traffic_patch(p)
            eflx_soil_grnd_u(p) = eflx_soil_grnd(p)
         end if

         ! Total fluxes (vegetation + ground)

         eflx_sh_tot(p) = eflx_sh_veg(p) + eflx_sh_grnd(p)
         qflx_evap_tot(p) = qflx_evap_veg(p) + qflx_evap_soi(p)
         eflx_lh_tot(p)= hvap*qflx_evap_veg(p) + htvp(c)*qflx_evap_soi(p)
         if (lun%itype(l) == istsoil .or. lun%itype(l) == istcrop) then
            eflx_lh_tot_r(p)= eflx_lh_tot(p)
            eflx_sh_tot_r(p)= eflx_sh_tot(p)
         else if (lun%urbpoi(l)) then
            eflx_lh_tot_u(p)= eflx_lh_tot(p)
            eflx_sh_tot_u(p)= eflx_sh_tot(p)
         end if

         ! Assign ground evaporation to sublimation from soil ice or to dew
         ! on snow or ground

         qflx_evap_grnd(p) = 0._r8
         qflx_sub_snow(p) = 0._r8
         qflx_dew_snow(p) = 0._r8
         qflx_dew_grnd(p) = 0._r8

         if (qflx_ev_snow(p) >= 0._r8) then
            ! for evaporation partitioning between liquid evap and ice sublimation, 
            ! use the ratio of liquid to (liquid+ice) in the top layer to determine split
            if ((h2osoi_liq(c,j)+h2osoi_ice(c,j)) > 0.) then
               qflx_evap_grnd(p) = max(qflx_ev_snow(p)*(h2osoi_liq(c,j)/(h2osoi_liq(c,j)+h2osoi_ice(c,j))), 0._r8)
            else
               qflx_evap_grnd(p) = 0.
            end if
            qflx_sub_snow(p) = qflx_ev_snow(p) - qflx_evap_grnd(p)
         else
            if (t_grnd(c) < tfrz) then
               qflx_dew_snow(p) = abs(qflx_ev_snow(p))
            else
               qflx_dew_grnd(p) = abs(qflx_ev_snow(p))
            end if
         end if

         ! Variables needed by history tape

         qflx_evap_can(p)  = qflx_evap_veg(p) - qflx_tran_veg(p)
         eflx_lh_vege(p)   = (qflx_evap_veg(p) - qflx_tran_veg(p)) * hvap
         eflx_lh_vegt(p)   = qflx_tran_veg(p) * hvap
         eflx_lh_grnd(p)   = qflx_evap_soi(p) * htvp(c)

      end do
      call t_stopf('bgp2_loop_2')
      call t_startf('bgp2_loop_3')

      ! Soil Energy balance check

      do fp = 1,num_nolakep
         p = filter_nolakep(fp)
         c = patch%column(p)
         errsoi_patch(p) = eflx_soil_grnd(p) - xmf(c) - xmf_h2osfc(c) &
              - frac_h2osfc(c)*(t_h2osfc(c)-t_h2osfc_bef(c)) &
              *(c_h2osfc(c)/dtime)
         errsoi_patch(p) =  errsoi_patch(p)+eflx_h2osfc_to_snow_col(c) 
         ! For urban sunwall, shadewall, and roof columns, the "soil" energy balance check
         ! must include the heat flux from the interior of the building.
         if (col%itype(c)==icol_sunwall .or. col%itype(c)==icol_shadewall .or. col%itype(c)==icol_roof) then
            errsoi_patch(p) = errsoi_patch(p) + eflx_building_heat_errsoi(c) 
         end if
      end do
      do j = -nlevsno+1,nlevgrnd
         do fp = 1,num_nolakep
            p = filter_nolakep(fp)
            c = patch%column(p)

            if ((col%itype(c) /= icol_sunwall .and. col%itype(c) /= icol_shadewall &
                 .and. col%itype(c) /= icol_roof) .or. ( j <= nlevurb)) then
               ! area weight heat absorbed by snow layers
               if (j >= col%snl(c)+1 .and. j < 1) errsoi_patch(p) = errsoi_patch(p) &
                    - frac_sno_eff(c)*(t_soisno(c,j)-tssbef(c,j))/fact(c,j)
               if (j >= 1) errsoi_patch(p) = errsoi_patch(p) &
                    - (t_soisno(c,j)-tssbef(c,j))/fact(c,j)
            end if
         end do
      end do
      call t_stopf('bgp2_loop_3')
      call t_startf('bgp2_loop_4')

      ! Outgoing long-wave radiation from vegetation + ground
      ! For conservation we put the increase of ground longwave to outgoing
      ! For urban patches, ulrad=0 and (1-fracveg_nosno)=1, and eflx_lwrad_out and eflx_lwrad_net 
      ! are calculated in UrbanRadiation. The increase of ground longwave is added directly 
      ! to the outgoing longwave and the net longwave.

      do fp = 1,num_nolakep
         p = filter_nolakep(fp)
         c = patch%column(p)
         l = patch%landunit(p)
         g = patch%gridcell(p)
         j = col%snl(c)+1

         if (.not. lun%urbpoi(l)) then
            lw_grnd=(frac_sno_eff(c)*tssbef(c,col%snl(c)+1)**4 &
                 +(1._r8-frac_sno_eff(c)-frac_h2osfc(c))*tssbef(c,1)**4 &
                 +frac_h2osfc(c)*t_h2osfc_bef(c)**4)

            eflx_lwrad_out(p) = ulrad(p) &
                 + (1-frac_veg_nosno(p))*(1.-emg(c))*forc_lwrad(c) &
                 + (1-frac_veg_nosno(p))*emg(c)*sb*lw_grnd &
                 + 4._r8*emg(c)*sb*t_grnd0(c)**3*tinc(c)


            ! Calculate the skin temperature as a weighted sum of all the surface contributions (surface water table, snow, etc...)
            ! Note: This is the bare ground calculation of skin temperature
            !       The Urban and Vegetation are done in other place.  Urban=Later in this function Veg=CanopyFluxMod
!            t_skin_patch(p) = ((1._r8 - emv(p))*(1-frac_veg_nosno(p)) * sqrt(sqrt(lw_grnd)))  +  emv(p)*t_veg(p)
!            if( frac_veg_nosno(p).eq.0 ) then
!               t_skin_patch(p) = ((1._r8 - emv(p))*(1-frac_veg_nosno(p)) * sqrt(sqrt(lw_grnd)))  +  &
!                                           emv(p) *   frac_veg_nosno(p)  * t_veg(p)
!            end if
             if(frac_veg_nosno(p).eq.0) then
                t_skin_patch(p) = sqrt(sqrt(lw_grnd))
                if (lun%itype(l) == istsoil .or. lun%itype(l) == istcrop) then
                   !-------------------------------------------------------------Where Keer Modified-----------------------------------------------------------
                   t_skin_r_patch(p) = t_skin_patch(p)
                   !-------------------------------------------------------------Where Keer Modified----------------------------------------------------------- 
                end if               
             end if
            eflx_lwrad_net(p) = eflx_lwrad_out(p) - forc_lwrad(c)
            if (lun%itype(l) == istsoil .or. lun%itype(l) == istcrop) then
               eflx_lwrad_net_r(p) = eflx_lwrad_out(p) - forc_lwrad(c)
               eflx_lwrad_out_r(p) = eflx_lwrad_out(p)
            end if
         else
            eflx_lwrad_out(p) = eflx_lwrad_out(p) + eflx_lwrad_del(p)
            eflx_lwrad_net(p) = eflx_lwrad_net(p) + eflx_lwrad_del(p)
            eflx_lwrad_net_u(p) = eflx_lwrad_net_u(p) + eflx_lwrad_del(p)
            eflx_lwrad_out_u(p) = eflx_lwrad_out(p)
         end if
      end do

      ! lake balance for errsoi is not over pft
      ! therefore obtain column-level radiative temperature

      call p2c(bounds, num_nolakec, filter_nolakec, &
           errsoi_patch(bounds%begp:bounds%endp), &
           errsoi_col(bounds%begc:bounds%endc))

      ! Assign column-level t_soisno(snl+1) to t_skin for each urban pft
      do fp = 1, num_urbanp
         p = filter_urbanp(fp)         
         c = patch%column(p)
         
         t_skin_patch(p) = t_soisno(c,col%snl(c)+1)
         !-------------------------------------------------------------Where Keer Modified-----------------------------------------------------------
         t_skin_u_patch(p) = t_skin_patch(p)
         !-------------------------------------------------------------Where Keer Modified-----------------------------------------------------------                
  
      end do

      call t_stopf('bgp2_loop_4')

    end associate 

  end subroutine SoilFluxes

end module SoilFluxesMod

