#!/bin/bash

# Log file
CONFIG_ROOT="/home/jguo/research/case_config_scripts/i.e215.I2000Clm50SpGs.hw/production.03"
LOGFILE="$CONFIG_ROOT/command_log.txt"

# Function to log and execute commands
log_and_execute() {
    echo "$@" >> "$LOGFILE"  # Log the command to the log file
    "$@"                      # Execute the command
}

# Defining variables
export CTSM_ROOT="/home/jguo/my_cesm_sandbox"
export CASE_NAME="i.e215.I2000Clm50SpGs.hw_production.03"
export PROJECT_DIR="/home/jguo/projects/cesm/scratch"
export DIR_CASE="$PROJECT_DIR/$CASE_NAME"
export COMPSET="I2000Clm50SpGs"
export RESOLUTION="f09_g17"

export USER_NL_CLM_FILE="$CONFIG_ROOT/user_nl_clm"
export MY_SourceMods="$CONFIG_ROOT/SourceMods"

export FORCE_RM="F"

# Navigating to the project directory
log_and_execute cd "$PROJECT_DIR"

# Check if the directory exists and FORCE_RM is set to T or true
if [ -d "$DIR_CASE" ]; then
    if [ "$FORCE_RM" = "T" ] || [ "$FORCE_RM" = "true" ]; then
        log_and_execute echo "Removing existing case directory: $DIR_CASE"
        log_and_execute rm -rf "$DIR_CASE"
    else
        log_and_execute echo "Directory $DIR_CASE already exists. Aborting."
        exit 1
    fi
fi

# Creating a new case
if ! [ -d "$DIR_CASE" ]; then
    log_and_execute cd "$CTSM_ROOT/cime/scripts"
    log_and_execute ./create_newcase --case "$DIR_CASE" --compset $COMPSET --res $RESOLUTION --run-unsupported
fi

# Entering the case directory
log_and_execute cd "$DIR_CASE"

# Changing XML configurations
log_and_execute ./xmlchange RUN_STARTDATE=1985-01-01,STOP_OPTION=nyears,STOP_N=1,RESUBMIT=29
log_and_execute ./xmlchange DATM_CLMNCEP_YR_ALIGN=1985,DATM_CLMNCEP_YR_START=1985,DATM_CLMNCEP_YR_END=2014
log_and_execute ./xmlchange DOUT_S_ROOT="/media/jguo/external_data/simulation_output/i.e215.I2000Clm50SpGs.hw_production.03"


# Setting up the case
log_and_execute ./case.setup

# Querying XML variables
log_and_execute ./xmlquery RUN_REFCASE,RUN_REFDIR,RUN_REFDATE,RUN_TYPE,RUN_STARTDATE,CCSM_BGC,CCSM_CO2_PPMV,CLM_NML_USE_CASE,CLM_CO2_TYPE,CLM_BLDNML_OPTS,STOP_OPTION,STOP_N,RESUBMIT,JOB_WALLCLOCK_TIME,CONTINUE_RUN,DATM_CPLHIST_CASE,DATM_CPLHIST_DIR,LND_TUNING_MODE,DATM_CPLHIST_YR_ALIGN,DATM_CPLHIST_YR_END,DATM_CPLHIST_YR_START,DATM_CLMNCEP_YR_ALIGN,DATM_CLMNCEP_YR_START,DATM_CLMNCEP_YR_END

# Copying the user_nl_clm file
log_and_execute cp "$USER_NL_CLM_FILE" "$DIR_CASE/"

# Copying the SourceMods dir
log_and_execute cp -r "$MY_SourceMods" "$DIR_CASE/"

# Note: The case build and submit commands are commented out. Uncomment and use log_and_execute if needed.
log_and_execute ./case.build --clean-all &&  log_and_execute ./case.build && log_and_execute ./case.submit
