#!/bin/zsh

START="$(date +%s)"

######################################
### Conda Environment Installation ###
######################################

# Uncomment the necessary environment that corresponds with hardware available
# NOTE: GPU version only works with CUDA devices.
cd environments
conda env create -f cpu_environment.yml
#conda env create -f gpu_environment.yml 

##########################
### Dataset Extraction ###
##########################
cd ../protein_dataset
tar -xf 1e14.zip.bz2
tar -xf 1e15.zip.bz2
tar -xf 1e16.zip.bz2


###########################
### NN Model Extraction ###
###########################

cd ../models/gpu1/early_termination
zip -F --quiet gpu1_1e14_early_termination.zip --out gpu1_1e14_early_termination_FULL.zip
zip -F --quiet gpu1_1e15_early_termination.zip --out gpu1_1e15_early_termination_FULL.zip
zip -F --quiet gpu1_1e16_early_termination.zip --out gpu1_1e16_early_termination_FULL.zip
unzip gpu1_1e14_early_termination_FULL.zip
unzip gpu1_1e15_early_termination_FULL.zip
unzip gpu1_1e16_early_termination_FULL.zip


cd ../no_early_termination
zip -F --quiet gpu1_1e14_no_early_termination.zip --out gpu1_1e14_no_early_termination_FULL.zip
zip -F --quiet gpu1_1e15_no_early_termination.zip --out gpu1_1e15_no_early_termination_FULL.zip
zip -F --quiet gpu1_1e16_no_early_termination.zip --out gpu1_1e16_no_early_termination_FULL.zip
unzip gpu1_1e14_no_early_termination_FULL.zip
unzip gpu1_1e15_no_early_termination_FULL.zip
unzip gpu1_1e16_no_early_termination_FULL.zip


cd ../../gpu4/early_termination
zip -F --quiet gpu4_1e14_early_termination.zip --out gpu4_1e14_early_termination_FULL.zip
zip -F --quiet gpu4_1e15_early_termination.zip --out gpu4_1e15_early_termination_FULL.zip
zip -F --quiet gpu4_1e16_early_termination.zip --out gpu4_1e16_early_termination_FULL.zip
unzip gpu4_1e14_early_termination_FULL.zip
unzip gpu4_1e15_early_termination_FULL.zip
unzip gpu4_1e16_early_termination_FULL.zip


DURATION=$[ $(date +%s) - ${START} ]
echo "Setup took ${DURATION} seconds"
