# HCP_MRI-behavior
Code for predicting individual differences in behavioral variables (e.g., intelligence, personality) from resting-state fMRI functional connectivity, using data from the Young Adult Human Connectome Project. The code depends on the HCP data structure. HCP data (MRI, and behavior/demographics) is available from the HCP website (https://www.humanconnectome.org/study/hcp-young-adult)

Authors: Julien Dubois (jcrdubois@gmail.com) and Paola Galdi (paola.galdi@gmail.com)

The code is provided as is, for documentation purposes. 

The following files are included:

  * *HCP_helpers.py* : contains all helper functions and module imports for resting-state fMRI preprocessing, and prediction of behavior

  * *personality.ipynb* : reproduces analyses in 
  > Dubois, J.\*, Galdi, P.\*, Han, Y., Paul, L.K. and Adolphs, R. Resting-state functional brain connectivity best predicts the personality dimension of openness to experience. In revision, to appear in *Personality Neuroscience*. Preprint: https://www.biorxiv.org/content/early/2017/11/07/215129

  * *intelligence.ipynb* : reproduces analyses in 
  > Dubois, J., Galdi, P., Paul, L.K. and Adolphs, R. A distributed brain network predicts general intelligence from resting-state human neuroimaging data. Under review. Preprint: https://www.biorxiv.org/content/early/2018/01/31/257865


If you use this code, please **cite the most relevant of these two papers**. 

## Prerequisites

In order to run the provided code the following packages are needed:

- scipy==0.19.0
- pandas==0.20.1
- matplotlib==2.0.2
- statsmodels==0.8.0
- numpy==1.12.1
- nibabel==2.2.1
- nilearn==0.4.0
- nipype==1.0.1
- seaborn==0.8.1
- scikit_learn==0.19.1

## Instruction for launching:
1. Customize parameters
	* In the notebook in cell <b>Set Parameters</b>
2. Launch the pipeline
	* In the notebook cells can be executed sequentially (one by one or from the menu Cell->Run All)

## Overview

### Preprocessing Pipeline
Preprocessing is performed executing a sequence of steps or operations, each of which corresponds to a function in the *HCP_helpers.py* file. There are 7 available operations: [Voxel Normalization](#voxel-normalization), [Detrending](#detrending), [Motion Regression](#motion-regression), [Scrubbing](#voxel-normalization), [Tissue Regression](#tissue-regression), [Global Signal Regression](#global-signal-regression) and [Temporal Filtering](#temporal-filtering). The corresponding functions are built using the following template:

` def OperationName(niiImg, flavor, masks, imgInfo):`

* <b>niiImg</b> it is a list of two elements `[data,volData]`. If the file to be processes is a Nifti file, the variable `data` contains Nifti data and the variable `volData` is `None`. If the input file is a Cifti file, the variable `data` contains Cifti data and the variable `volData` contains volumetric data. Image data can be loaded with function `load_img()`.
* <b>flavor</b> is a list containing the Operation parameters (detailed in the [following section](#pipeline-operations)).
* <b>masks</b> is the output of the function `makeTissueMasks()`, a list of four elements corresponding to 1) whole brain mask, 2) white matter mask, 3) cerebrospinal fluid mask and 4) gray matter mask. 
* <b>imgInfo</b> is a list of 6 elements corresponding to 1) no. of rows in a slice, 2) no. of columns in a slice, 3) no. of slices, 4) no. of time points, 5) the affine matrix and 6) repetion time. These data can be retrieved with function `load_img()`.

A preprocessing pipeline can be fully described with the following data structure:

```
[
    ['Operation1',  1, [param1]],
    ['Operation2',  2, [param1, param2, param3]],
    ['Operation3',  3, [param1, param2]],
    ['Operation4',  4, [param1]],
    ['Operation5',  5, [param1, param2]],
    ['Operation6',  6, [param1, param2, param3]],
    ['Operation7',  7, [param]]
]
```
It is a list of structures. Each structure is a list of 3 elements:
1. The operation name.
2. The operation order.
3. The list of parameters for the specific operation.

Operations are executed following the operation order specified by the user. Note that in case of regression operations, a single regression step combining multiple regressors (e.g., tissue regressors and global signal regressor) can be requested by assigning the same order to all regression steps.

### Pipeline Operations

#### Voxel Normalization
* <b>zscore:</b> convert each voxel’s time course to z-score (remove mean, divide by standard deviation).
* <b>demean:</b> substract the mean from each voxel's time series.

Example: `['VoxelNormalization',      1, ['zscore']]`
#### Detrending
* <b>poly:</b> polynomial regressors up to specified order.
	1. Specify polynomial order.
	2. Specify tissue, one of 'WMCSF' or 'GM'.
*<b>legendre:</b> Legendre polynomials up to specified order.
	1. Specify polynomial order 
	2. Specify tissue, one of 'WMCSF' or 'GM'.

Example: `['Detrending',      2, ['poly', 3, 'GM']]`
#### Motion Regression 
Note: R = [X Y Z pitch yaw roll]<br>
Note: if temporal filtering has already been performed, the motion regressors are filtered too. <br>
Note: if scrubbing has been requested, a regressor is added for each volume to be censored; the censoring option performs only scrubbing.
* <b>R dR:</b> translational and rotational realignment parameters (R) and their temporal derivatives are used as explanatory variables in motion regression.
* <b>R dR R^2 dR^2:</b>: realignment parameters (R) with their derivatives (dR), quadratic terms (R^2) and square of derivatives (dR^2) are used in motion regression.
* <b>censoring:</b> for each volume tagged for scrubbing, a unit impulse function  with a value of 1 at that time point and 0 elsewhere is included as a regressor.

Example: `['MotionRegression',      3, ['R dR']]`
#### Scrubbing
Note: this step only flags the volumes to be censored, that are then regressed out in the MotionRegression step.<br>
Note: uncensored segments of data lasting fewer than 5 contiguous volumes, are flagged for removal as well.
* <b>FD+DVARS</b>
	1. Specify a threshold for framewise displacement (FD) in mm.
	2. Specify a threshold <i>t</i> s.t. volumes with a variance of differentiated signal (DVARS) greater than (100 + <i>t</i>)% of the run median DVARS are flagged for removal.
	3. Specify number of adjacent volumes to exclude (optional).
*<b>RMS</b>
	1. Specify threshold for root mean square displacement in mm.
	2. Specify number of adjacent volumes to exclude (optional).

Example: `['Scrubbing',      4, ['FD+DVARS', 0.25, 5]]`
#### Tissue Regression 
*<b>GM:</b> the gray matter signal is added as a regressor.
	1. Specify if regression should be performed on gray matter signal ('GM') or whole brain signal ('wholebrain')
*<b>WMCSF:</b> white matter and cerebrospinal fluid signals are added as regressors.
	1. Specify if regression should be performed on gray matter signal ('GM') or whole brain signal ('wholebrain')
*<b>WMCSF+dt+sq:</b> white matter and cerebrospinal fluid signals with their derivatives and quadratic terms are added as regressors.
	1. Specify if regression should be performed on gray matter signal ('GM') or whole brain signal ('wholebrain')
*CompCor:</b> a PCA-based method (Behzadi et al., 2007) is used to derive N components from CSF and WM signals.
	1. Specify no. of components to compute for specified tissue mask (see following parameter).
	2. Specify if components should be computed using a single mask for white matter and cerebrospinal fluid ('WMCSF') or separatedly for white matter and cerebrospinal fluid ('WM+CSF')
	3. Specify if regression should be performed on gray matter signal ('GM') or whole brain signal ('wholebrain').

Example: `['TissueRegression',        5, ['CompCor', 5, 'WMCSF', 'wholebrain']]`

#### Global Signal Regression
*<b>GS:</b> the global signal is added as a regressor.
*<b>GS+dt+sq:</b> the global signal with its derivative and square term are added as regressors.

Example: `['GlobalSignalRegression',      6, ['GS+dt+sq']]`
#### Temporal Filtering
Note: if scrubbing has been requested, censored volumes are replaced by linear interpolation.
* <b>Butter:</b> Butterworth band pass filtering.
	1. Specify high pass threshold.
	2. Specify low pass threshold. </ol>
* <b>Gaussian:</b> Low pass Gaussian smoothing.
	1. Specify standard deviation.</ol>

Example: `['TemporalFiltering',       7, ['Butter', 0.009, 0.08]]`
