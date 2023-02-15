# HCP_MRI-behavior
Code for predicting individual differences in behavioral variables (e.g., intelligence, personality) from resting-state fMRI functional connectivity, using data from the Young Adult Human Connectome Project. The code depends on the HCP data structure. HCP data (MRI, and behavior/demographics) is available from the HCP website (https://www.humanconnectome.org/study/hcp-young-adult)

A later version of this pipeline, extended to be used with other datasets processed with fmriprep, is available at https://github.com/adolphslab/rsDenoise

Authors: Julien Dubois (jcrdubois@gmail.com) and Paola Galdi (paola.galdi@gmail.com)

The code is provided as is, for documentation purposes. 

The following files are included:

  * *HCP_helpers.py* : contains all helper functions and module imports for resting-state fMRI preprocessing, and prediction of behavior

  * *personality.ipynb* : reproduces analyses in 
  > Dubois, J.\*, Galdi, P.\*, Han, Y., Paul, L.K. and Adolphs, R. Resting-state functional brain connectivity best predicts the personality dimension of openness to experience. *Personality Neuroscience*, in press. Preprint: https://www.biorxiv.org/content/early/2018/03/02/215129

  * *intelligence.ipynb* : reproduces analyses in 
  > Dubois, J., Galdi, P., Paul, L.K. and Adolphs, R. A distributed brain network predicts general intelligence from resting-state human neuroimaging data. *Philosophical Transactions of the Royal Society B*, in revision. Preprint: https://www.biorxiv.org/content/early/2018/01/31/257865

  * *recomputeNEOFFIfactors.ipynb* : standalone notebook which can be used to correct NEOFAC_A [agreeableness factor] (see https://www.mail-archive.com/hcp-users@humanconnectome.org/msg05266.html: it appears that one of the test items was not properly reverse coded to compute the agreeableness factor). Data downloaded on 03/08/2018 was found to still be affected by the bug

If you use this code, please **cite the most relevant of these two papers**. 

## Prerequisites.

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

## Instruction for launching
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

There are three pipelines already specified in the *HCP_helpers.py* file.

```
config.operationDict = {
    'A': [ #Finn et al. 2015
        ['VoxelNormalization',      1, ['zscore']],
        ['Detrending',              2, ['legendre', 3, 'WMCSF']],
        ['TissueRegression',        3, ['WMCSF', 'GM']],
        ['MotionRegression',        4, ['R dR']],
        ['TemporalFiltering',       5, ['Gaussian', 1]],
        ['Detrending',              6, ['legendre', 3 ,'GM']],
        ['GlobalSignalRegression',  7, ['GS']]
        ],
    'B': [ #Satterthwaite et al. 2013 (Ciric7)
        ['VoxelNormalization',      1, ['demean']],
        ['Detrending',              2, ['poly', 2, 'wholebrain']],
        ['TemporalFiltering',       3, ['Butter', 0.01, 0.08]],
        ['MotionRegression',        4, ['R dR R^2 dR^2']],
        ['TissueRegression',        4, ['WMCSF+dt+sq', 'wholebrain']],
        ['GlobalSignalRegression',  4, ['GS+dt+sq']],
        ['Scrubbing',               4, ['RMS', 0.25]]
        ],
    'C': [ #Siegel et al. 2016 (SiegelB)
        ['VoxelNormalization',      1, ['demean']],
        ['Detrending',              2, ['poly', 1, 'wholebrain']],
        ['TissueRegression',        3, ['CompCor', 5, 'WMCSF', 'wholebrain']],
        ['TissueRegression',        3, ['GM', 'wholebrain']], 
        ['GlobalSignalRegression',  3, ['GS']],
        ['MotionRegression',        3, ['censoring']],
        ['Scrubbing',               3, ['FD+DVARS', 0.25, 5]], 
        ['TemporalFiltering',       4, ['Butter', 0.009, 0.08]]
        ],
    }
```

### Pipeline Operations

#### Voxel Normalization
* <b>zscore:</b> convert each voxel’s time course to z-score (remove mean, divide by standard deviation).
* <b>demean:</b> substract the mean from each voxel's time series.
* <b>pcSigCh:</b> convert each voxel’s time course to % signal change (remove mean, divide by mean, multiply by 100)

Example: `['VoxelNormalization',      1, ['zscore']]`
#### Detrending
* <b>poly:</b> polynomial regressors up to specified order.
	1. Specify polynomial order.
	2. Specify tissue, one of 'WMCSF' or 'GM'.
* <b>legendre:</b> Legendre polynomials up to specified order.
	1. Specify polynomial order 
	2. Specify tissue, one of 'WMCSF' or 'GM'.

Example: `['Detrending',      2, ['poly', 3, 'GM']]`
#### Motion Regression 
Note: R = [X Y Z pitch yaw roll]<br>
Note: if temporal filtering has already been performed, the motion regressors are filtered too. <br>
Note: if scrubbing has been requested, a regressor is added for each volume to be censored; the censoring option performs only scrubbing.
* <b>R:</b> translational and rotational realignment parameters (R) are used as explanatory variables in motion regression.
* <b>R dR:</b> translational and rotational realignment parameters (R) and their temporal derivatives (dR) are used as explanatory variables in motion regression.
* <b>R R^2:</b> translational and rotational realignment parameters (R) and their quadratic terms (R^2) are used as explanatory variables in motion regression.
* <b>R dR R^2 dR^2:</b> realignment parameters (R) with their derivatives (dR), quadratic terms (R^2) and square of derivatives (dR^2) are used in motion regression.
* <b>R R^2 R-1 R-1^2:</b> realignment parameters at time t (R) and realignment parameters at time t-1 (R-1) with their square terms are used in motion regression.
* <b>R R^2 R-1 R-1^2 R-2 R-2^2:</b> realignment parameters at time t (R), t-1 (R-1) and t-2 (R-2) with their square terms are used in motion regression.
* <b>censoring:</b> for each volume tagged for scrubbing, a unit impulse function  with a value of 1 at that time point and 0 elsewhere is included as a regressor.
* <b>ICA-AROMA:</b> ICA is used to identify noise components that are used as regressors.

Example: `['MotionRegression',      3, ['R dR']]`
#### Scrubbing
Note: this step only flags the volumes to be censored, that are then regressed out in the MotionRegression step.<br>
Note: uncensored segments of data lasting fewer than 5 contiguous volumes, are flagged for removal as well.
* <b>FD</b>
	1. Specify a threshold for framewise displacement (FD) in mm.
	2. Specify number of adjacent volumes to exclude (optional).
* <b>DVARS</b>
	1. Specify a threshold for variance of differentiated signal (DVARS).
	2. Specify number of adjacent volumes to exclude (optional).
* <b>FD+DVARS</b>
	1. Specify a threshold for framewise displacement (FD) in mm.
	2. Specify a threshold <i>t</i> s.t. volumes with a variance of differentiated signal (DVARS) greater than (100 + <i>t</i>)% of the run median DVARS are flagged for removal (as in Siegel et al, Cerebral Cortex, 2016).
	3. Specify number of adjacent volumes to exclude (optional).
* <b>RMS</b>
	1. Specify threshold for root mean square displacement in mm.
	2. Specify number of adjacent volumes to exclude (optional).

Example: `['Scrubbing',      4, ['FD+DVARS', 0.25, 5]]`
#### Tissue Regression 
* <b>GM:</b> the gray matter signal is added as a regressor.
	1. Specify if regression should be performed on gray matter signal ('GM') or whole brain signal ('wholebrain')
* <b>WM:</b> the white matter signal is added as a regressor.
	1. Specify if regression should be performed on gray matter signal ('GM') or whole brain signal ('wholebrain')
* <b>WMCSF:</b> white matter and cerebrospinal fluid signals are added as regressors.
	1. Specify if regression should be performed on gray matter signal ('GM') or whole brain signal ('wholebrain')
* <b>WMCSF+dt:</b> white matter and cerebrospinal fluid signals with their derivatives		 are added as regressors.
	1. Specify if regression should be performed on gray matter signal ('GM') or whole brain signal ('wholebrain')
* <b>WMCSF+dt+sq:</b> white matter and cerebrospinal fluid signals with their derivatives and quadratic terms are added as regressors.
	1. Specify if regression should be performed on gray matter signal ('GM') or whole brain signal ('wholebrain')
* <b>CompCor:</b> a PCA-based method (Behzadi et al., 2007) is used to derive N components from CSF and WM signals.
	1. Specify no. of components to compute for specified tissue mask (see following parameter).
	2. Specify if components should be computed using a single mask for white matter and cerebrospinal fluid ('WMCSF') or separatedly for white matter and cerebrospinal fluid ('WM+CSF')
	3. Specify if regression should be performed on gray matter signal ('GM') or whole brain signal ('wholebrain').

Example: `['TissueRegression',        5, ['CompCor', 5, 'WMCSF', 'wholebrain']]`

#### Global Signal Regression
* <b>GS:</b> the global signal is added as a regressor.
* <b>GS+dt+sq:</b> the global signal with its derivatives are added as regressors.
* <b>GS+dt+sq:</b> the global signal with its derivatives and square term are added as regressors.

Example: `['GlobalSignalRegression',      6, ['GS+dt+sq']]`
#### Temporal Filtering
Note: if scrubbing has been requested, censored volumes are replaced by linear interpolation.
* <b>Butter:</b> Butterworth band pass filtering.
	1. Specify high pass threshold.
	2. Specify low pass threshold. 
* <b>Gaussian:</b> Low pass Gaussian smoothing.
	1. Specify standard deviation.
* <b>DCT:</b> The discrete cosine transform is used to compute regressors to perform temporal filtering.
	1. Specify high pass threshold.
	2. Specify low pass threshold. 

Example: `['TemporalFiltering',       7, ['Butter', 0.009, 0.08]]`

## Changelog
- 04/24/2018: Fixed indexing bug in the feature selection step (lines 1389-1391 in HCP_helpers.py). Prior to this date the feature selection step was performed on features indexed by range(1,n_edges) instead of range(0,n_edges), always discarding the first (0-indexed) feature. As a consequence, there might be slight discrepancies in analyses run before and after the fix.
- 11/21/2018: Fixed subject selection bug; the code did not properly discard subjects with incomplete RS data. The original code included 11 subjects with less than 4800 timepoints [119732,119833,140420,150423,159946,169747,196952,202820,317332,644246,751550]. The total number of selected subjects for the analyses is now 874 (instead of 885). Thanks to Feilong Ma (@feilong) for pointing this out.
