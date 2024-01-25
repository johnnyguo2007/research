#!/bin/bash

# Defining variables
export CTSM_ROOT="/home/jguo/my_cesm_sandbox"
export CASE_NAME="i.e215.I2000Clm50SpGs.hw.001"
export PROJECT_DIR="/home/jguo/projects/cesm/scratch"
export DIR_CASE="$PROJECT_DIR/$CASE_NAME"
export COMPSET="I2000Clm50SpGs"
export RESOLUTION="f09_g17"
# /home/jguo/research/case_config_scripts/i.e215.I2000CPLHISTClm50BgcCropGs._Johnny_BHIST_Spin/user_nl_clm
export USER_NL_CLM_FILE="/home/jguo/research/case_config_scripts/i.e215.I2000Clm50SpGs.hw/user_nl_clm"


export FORCE_RM="T"

# Navigating to the project directory
cd $PROJECT_DIR

# Check if the directory exists and FORCE_RM is set to T or true
if [ -d "$DIR_CASE" ]; then
    if [ "$FORCE_RM" = "T" ] || [ "$FORCE_RM" = "true" ]; then
        echo "Removing existing case directory: $DIR_CASE"
        rm -rf "$DIR_CASE"
    else
        echo "Directory $DIR_CASE already exists. Aborting."
        exit 1
    fi
fi

# Creating a new case
# Note: Added a check to ensure the directory doesn't exist before creating a new case
if ! [ -d "$DIR_CASE" ]; then
    # Keer, you may need to add in project, driver and walltime etc.
    cd $CTSM_ROOT/cime/scripts
    ./create_newcase --case "$DIR_CASE" --compset $COMPSET --res $RESOLUTION --run-unsupported
fi

# Entering the case directory
cd "$DIR_CASE"

# Setting up the case
./case.setup

# Changing XML configurations

#./xmlchange RUN_STARTDATE=0001-01-01,STOP_OPTION=nyears,STOP_N=1,RESUBMIT=20
./xmlchange RUN_STARTDATE=0001-01-01,STOP_OPTION=nmonths,STOP_N=1,RESUBMIT=3
./xmlchange DATM_CLMNCEP_YR_ALIGN=1, DATM_CLMNCEP_YR_START=1985,DATM_CLMNCEP_YR_END=2014

# Querying XML variables
./xmlquery RUN_REFCASE,RUN_REFDIR,RUN_REFDATE,RUN_TYPE,RUN_STARTDATE,CCSM_BGC,CCSM_CO2_PPMV,CLM_NML_USE_CASE,CLM_CO2_TYPE,CLM_BLDNML_OPTS,STOP_OPTION,STOP_N,RESUBMIT,JOB_WALLCLOCK_TIME,CONTINUE_RUN,DATM_CPLHIST_CASE,DATM_CPLHIST_DIR,LND_TUNING_MODE,DATM_CPLHIST_YR_ALIGN,DATM_CPLHIST_YR_END,DATM_CPLHIST_YR_START

# Copying the user_nl_clm file
cp "$USER_NL_CLM_FILE" "$DIR_CASE/"

# Building and submitting the case
./case.build --clean-all && ./case.build && ./case.submit
