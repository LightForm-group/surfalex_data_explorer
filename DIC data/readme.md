# DIC data for Surfalex material

This folder contains DIC data acquired during tensile test of Surfalex material. The structure of the files is described below.

## Data folder
The raw data files are provided in the 'Data' folder

test 0-1 means tensile test conducted along the rolling direction (1st sample)
test 30-1 means tensile test conducted along the 30 degrees from rolling direction (1st sample)
test 45-1 means tensile test conducted along the 45 degrees from rolling direction (1st sample)
test 60-1 means tensile test conducted along the 60 degrees from rolling direction (1st sample)
test 90-1 means tensile test conducted along the 90 degrees from rolling direction (1st sample)

All the tests which end with number '2' are the repeat tests.

Each of these folders contains three sub-folders.

1. Displacement data: It contains information about X and Y pixel and the corresponding displacement values for the different stages of deformation.

2. Images: It contans images of the sample recorded during different stages of deformation. 

3. Voltage data: It contains the voltage and strain data for different stages of deformation. The voltage can be converted to load and subsequently to stress. 

The voltage can be converted using the following conversion 1V = 0.5 kN.

The cross section area of each sample is provided below. Sample width and thickness are given in 'mm'.

sample width thickness
0-1    5.2    1.52
0-2    5.1    1.49
30-1   5.1    1.49
30-2   5.1    1.52
45-1   5.1    1.48
45-2   5.2    1.53
60-1   5.1    1.45
60-2   5.1    1.52
90-1   5.2    1.52
90-2   5.25   1.53

## Analysis folder

Scripts to analyse thes raw data are provided in the 'Analysis' folder. 

### Analysis of displacement data
The displacement data can be analsed using the Python Jupyter notebook provided in the 'Analysis' folder. Further instructions are in the notebook.

Once the script is excecuted, different types of plots which are shown in the powerpoint file in the 'Analysis' folder can be obtained. 