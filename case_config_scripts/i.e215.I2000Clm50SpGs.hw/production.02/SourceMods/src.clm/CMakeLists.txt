# Note that this is just used for unit testing; hence, we only need to add
# source files that are currently used in unit tests

list(APPEND clm_sources 
  BareGroundFluxesMod.F90
  CanopyFluxesMod.F90
  CMakeLists.txt
  HumanIndexMod.F90
  LakeFluxesMod.F90
  README
  SoilFluxesMod.F90
  TemperatureType.F90
  UrbanFluxesMod.F90
  WaterStateType.F90

  )

sourcelist_to_parent(clm_sources)
