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
<ol>
<li> Customize parameters
<ul>
<li> In the notebook in cell <b>Set Parameters</b>
</ul>
<li> Launch the pipeline
<ul>
<li> In the notebook cells can be executed sequentially (one by one or from the menu Cell->Run All)
</ul>
</ol>

## Overview

### Preprocessing Pipeline
...

### Pipeline Operations

#### Voxel Normalization
<ul>
  <li> <b>zscore:</b> convert each voxel’s time course to z-score (remove mean, divide by standard deviation).
  <li> <b>demean:</b> substract the mean from each voxel's time series.
</ul>
Example:
```
 ['VoxelNormalization',      1, ['zscore']]
```
#### Detrending
  <ul>
    <li> <b>poly:</b> polynomial regressors up to specified order.
	<ol><li>Specify polynomial order.</li>
	<li>Specify tissue, one of 'WMCSF' or 'GM'.</li> </li></ol>
    <li> <b>legendre:</b> Legendre polynomials up to specified order.
	<ol><li>Specify polynomial order 
	<li>Specify tissue, one of 'WMCSF' or 'GM'.</ol>
  </ul>
Example:
```
 ['Detrending',      2, ['poly', 3, 'GM']]
```
#### MotionRegression 
Note: R = [X Y Z pitch yaw roll]<br>
Note: if temporal filtering has already been performed, the motion regressors are filtered too. <br>
Note: if scrubbing has been requested, a regressor is added for each volume to be censored; the censoring option performs only scrubbing.
  <ul> 
  <li> <b>R dR:</b> translational and rotational realignment parameters (R) and their temporal derivatives are used as explanatory variables in motion regression.
  <li> <b>R dR R^2 dR^2:</b>: realignment parameters (R) with their derivatives (dR), quadratic terms (R^2) and square of derivatives (dR^2) are used in motion regression.
  <li> <b>censoring:</b> for each volume tagged for scrubbing, a unit impulse function  with a value of 1 at that time point and 0 elsewhere is included as a regressor.
  </ul>
Example:
```
 ['MotionRegression',      3, ['R dR']]
```
#### Scrubbing
Note: this step only flags the volumes to be censored, that are then regressed out in the MotionRegression step.<br>
Note: uncensored segments of data lasting fewer than 5 contiguous volumes, are flagged for removal as well.
  <ul>
  <li> <b>FD+DVARS</b>
<ol><li> Specify a threshold for framewise displacement (FD) in mm.
<li> Specify a threshold <i>t</i> s.t. volumes with a variance of differentiated signal (DVARS) greater than (100 + <i>t</i>)% of the run median DVARS are flagged for removal.
<li> Specify number of adjacent volumes to exclude (optional).
</ol>
  <li> <b>RMS</b>
<ol> <li> Specify threshold for root mean square displacement in mm.
<li> Specify number of adjacent volumes to exclude (optional).
</ol>
  </ul>
Example:
```
 ['Scrubbing',      4, ['FD+DVARS', 0.25, 5]]
```
#### Tissue Regression 
  <ul>
  <li> <b>GM:</b> the gray matter signal is added as a regressor.
	<ol><li>Specify if regression should be performed on gray matter signal ('GM') or whole brain signal ('wholebrain').</ol>
  <li> <b>WMCSF:</b> white matter and cerebrospinal fluid signals are added as regressors.
	<ol><li>Specify if regression should be performed on gray matter signal ('GM') or whole brain signal ('wholebrain').</ol>
  <li> <b>WMCSF+dt+sq:</b> white matter and cerebrospinal fluid signals with their derivatives and quadratic terms are added as regressors.
	<ol><li>Specify if regression should be performed on gray matter signal ('GM') or whole brain signal ('wholebrain').</ol>
  <li> <b>CompCor:</b> a PCA-based method (Behzadi et al., 2007) is used to derive N components from CSF and WM signals.
	<ol><li>Specify no. of components to compute for specified tissue mask (see following parameter).
	<li> Specify if components should be computed using a single mask for white matter and cerebrospinal fluid ('WMCSF') or separatedly for white matter and cerebrospinal fluid ('WM+CSF')
	<li>Specify if regression should be performed on gray matter signal ('GM') or whole brain signal ('wholebrain').
	</ol>
  </ul>
Example:
```
['TissueRegression',        5, ['CompCor', 5, 'WMCSF', 'wholebrain']]
```

#### Global Signal Regression
<ul>
<li> <b>GS:</b> the global signal is added as a regressor.
<li> <b>GS+dt+sq:</b> the global signal with its derivative and square term are added as regressors.
</ul>
Example:
```
 ['GlobalSignalRegression',      6, ['GS+dt+sq']]
```
#### Temporal Filtering
Note: if scrubbing has been requested, censored volumes are replaced by linear interpolation.
  <ul>
  <li> <b>Butter:</b> Butterworth band pass filtering.
<ol><li>Specify high pass threshold.
<li> Specify low pass threshold. </ol>
  <li> <b>Gaussian:</b> Low pass Gaussian smoothing.
	<ol><li>Specify standard deviation.</ol>
  </ul>
Example:
```
 ['TemporalFiltering',       7, ['Butter', 0.009, 0.08]]
```







