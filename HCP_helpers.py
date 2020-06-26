import os.path as op
from os import mkdir, makedirs, getcwd, remove, listdir, environ
#----------------------------------
# initialize global variable config
#----------------------------------
class config(object):
    overwrite          = False
    scriptlist         = list()
    joblist            = list()
    queue              = False
    tStamp             = ''
    useFIX             = False
    useMemMap          = False
    steps              = {}
    Flavors            = {}
    sortedOperations   = list()
    maskParcelswithGM  = False
    preWhitening       = False
    maskParcelswithAll = True
    save_voxelwise     = False
    useNative          = False
    parcellationName   = ''
    parcellationFile   = ''
    outDir             = 'rsDenoise'								
    FCDir              = 'FC'
    headradius         = 50 #50mm as in Powers et al. 2012
    interpolation      = 'linear'
    melodicFolder      = op.join('#fMRIrun#_hp2000.ica','filtered_func_data.ica') #the code #fMRIrun# will be replaced
    plotSteps          = False # produce a grayplot after each processing step
    isCifti            = False
    sourceDir          = getcwd()
    fcType             = 'correlation' # one of {"correlation", "partial correlation", "tangent", "covariance", "precision"}
    # these variables are initialized here and used later in the pipeline, do not change
    filtering   = []
    doScrubbing = False


#----------------------------------
# IMPORTS
#----------------------------------
# Force matplotlib to not use any Xwindows backend.
import matplotlib
# core dump with matplotlib 2.0.0; use earlier version, e.g. 1.5.3
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
					
																
import sys
import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy import stats, linalg,signal
import scipy.io as sio
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage.morphology import binary_closing, binary_dilation, binary_erosion, binary_opening, generate_binary_structure
import nipype.interfaces.fsl as fsl
from subprocess import call, check_output, CalledProcessError, Popen, getoutput
import nibabel as nib
import sklearn.model_selection as cross_validation
from sklearn.linear_model import ElasticNetCV
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model,feature_selection,preprocessing
from sklearn.preprocessing import RobustScaler											 
from nilearn.signal import clean
from nilearn import connectome
from sklearn.covariance import MinCovDet,GraphicalLassoCV,LedoitWolf
from past.utils import old_div
import operator
import gzip
import string
import random
import xml.etree.cElementTree as ET
from time import localtime, strftime, sleep, time
import fnmatch
import re
import os
import glob
from statsmodels.nonparametric.smoothers_lowess import lowess
import seaborn as sns
import nistats
from nistats import design_matrix

#----------------------------------
# function to build dinamycally path to input fMRI file
#----------------------------------
def buildpath():
    return op.join(config.DATADIR, config.subject,'MNINonLinear','Results',config.fmriRun)
    
#----------------------------------
# function to build dinamycally output path (BIDS-like) 
#----------------------------------
def outpath():
    if not op.isdir(config.outDir): mkdir(config.outDir)
    outPath = op.join(config.outDir,'denoise_'+config.pipelineName)
    if not op.isdir(outPath): mkdir(outPath)
    outPath = op.join(outPath,config.subject)
    if not op.isdir(outPath): mkdir(outPath)
    if hasattr(config, 'session') and config.session:
        outPath = op.join(outPath,config.session)
        if not op.isdir(outPath): mkdir(outPath)
    outPath = op.join(outPath,config.fmriRun)
    if not op.isdir(outPath): mkdir(outPath)
    return outPath


#----------------------------------
# EVs for task regression
#----------------------------------
# Selected as in Elliot et al. (2018)
def get_EVs(path,task):
    EVs = {}
    if task == 'GAMBLING' : EVs = {
        'win_event' : np.loadtxt(op.join(path,'EVs','win_event.txt'),ndmin=2),
        'loss_event' : np.loadtxt(op.join(path,'EVs','loss_event.txt'),ndmin=2),
        'neut_event' : np.loadtxt(op.join(path,'EVs','neut_event.txt'),ndmin=2),
    }
    if task == 'WM' : EVs = {
        '0bk_body' : np.loadtxt(op.join(path,'EVs','0bk_body.txt'),ndmin=2),
        '0bk_faces' : np.loadtxt(op.join(path,'EVs','0bk_faces.txt'),ndmin=2),
        '0bk_places' : np.loadtxt(op.join(path,'EVs','0bk_places.txt'),ndmin=2),
        '0bk_tools' : np.loadtxt(op.join(path,'EVs','0bk_tools.txt'),ndmin=2),
        '2bk_body' : np.loadtxt(op.join(path,'EVs','2bk_body.txt'),ndmin=2),
        '2bk_faces' : np.loadtxt(op.join(path,'EVs','2bk_faces.txt'),ndmin=2),
        '2bk_places' : np.loadtxt(op.join(path,'EVs','2bk_places.txt'),ndmin=2),
        '2bk_tools' : np.loadtxt(op.join(path,'EVs','2bk_tools.txt'),ndmin=2),
    }
    if task == 'MOTOR' : EVs = {
        'cue' : np.loadtxt(op.join(path,'EVs','cue.txt'),ndmin=2),
        'lf' : np.loadtxt(op.join(path,'EVs','lf.txt'),ndmin=2),
        'rf' : np.loadtxt(op.join(path,'EVs','rf.txt'),ndmin=2),
        'lh' : np.loadtxt(op.join(path,'EVs','lh.txt'),ndmin=2),
        'rh' : np.loadtxt(op.join(path,'EVs','rh.txt'),ndmin=2),
        't' : np.loadtxt(op.join(path,'EVs','t.txt'),ndmin=2),
    }
    if task == 'LANGUAGE' : EVs = {
        'cue' : np.loadtxt(op.join(path,'EVs','cue.txt'),ndmin=2),
        'present_math' : np.loadtxt(op.join(path,'EVs','present_math.txt'),ndmin=2),
        'question_math' : np.loadtxt(op.join(path,'EVs','question_math.txt'),ndmin=2),
        'response_math' : np.loadtxt(op.join(path,'EVs','response_math.txt'),ndmin=2),
        'present_story' : np.loadtxt(op.join(path,'EVs','present_story.txt'),ndmin=2),
        'question_story' : np.loadtxt(op.join(path,'EVs','question_story.txt'),ndmin=2),
        'response_story' : np.loadtxt(op.join(path,'EVs','response_story.txt'),ndmin=2),
    }
    if task == 'SOCIAL' : EVs = {
        'mental' : np.loadtxt(op.join(path,'EVs','mental.txt'),ndmin=2),
        'rnd' : np.loadtxt(op.join(path,'EVs','rnd.txt'),ndmin=2),
    }
    if task == 'RELATIONAL' : EVs = {
        'match' : np.loadtxt(op.join(path,'EVs','match.txt'),ndmin=2),
        'relation' : np.loadtxt(op.join(path,'EVs','relation.txt'),ndmin=2),
        'error' : np.loadtxt(op.join(path,'EVs','error.txt'),ndmin=2), # might be empty
    }
    if task == 'EMOTION' : EVs = {
        'fear' : np.loadtxt(op.join(path,'EVs','fear.txt'),ndmin=2),
        'neut' : np.loadtxt(op.join(path,'EVs','neut.txt'),ndmin=2),
    }
    return EVs

#----------------------------------
# 3 alternate denoising pipelines
# many more can be implemented
#----------------------------------
config.operationDict = {
    'Task': [ #test task regression
        ['TaskRegression',  1, []]
        ],
     'MyConnectome': [
        ['VoxelNormalization',      1, ['demean']],
        ['Detrending',              2, ['poly', 1, 'wholebrain']],
        ['TissueRegression',        3, ['WMCSF', 'wholebrain']],
        ['MotionRegression',        3, ['R R^2 R-1 R-1^2']],
        ['GlobalSignalRegression',  3, ['GS']],
        ['TemporalFiltering',       4, ['Butter', 0.009, 0.08]],
        ['Scrubbing',               5, ['FD', 0.25]]
        ],
    'A': [ #Finn et al. 2015
        ['VoxelNormalization',      1, ['zscore']],
        ['Detrending',              2, ['legendre', 3, 'WMCSF']],
        ['TissueRegression',        3, ['WMCSF', 'GM']],
        ['MotionRegression',        4, ['R dR']],
        ['TemporalFiltering',       5, ['Gaussian', 1]],
        ['Detrending',              6, ['legendre', 3 ,'GM']],
        ['GlobalSignalRegression',  7, ['GS']]
        ],
    'A0': [ #Finn et al. 2015 + Task regression
        ['VoxelNormalization',      1, ['zscore']],
        ['Detrending',              2, ['legendre', 3, 'WMCSF']],
        ['TissueRegression',        3, ['WMCSF', 'GM']],
        ['MotionRegression',        4, ['R dR']],
        ['TemporalFiltering',       5, ['Gaussian', 1]],
        ['Detrending',              6, ['legendre', 3 ,'GM']],
        ['GlobalSignalRegression',  7, ['GS']],
        ['TaskRegression',          8, []],
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
    'B0': [ # same as B, with very small change to force recomputation after bug discovered in polynomial filtering 2/12/2018
        ['VoxelNormalization',      1, ['demean']],
        ['Detrending',              2, ['poly', 2, 'wholebrain']],
        ['TemporalFiltering',       3, ['Butter', 0.01, 0.0801]], 
        ['MotionRegression',        4, ['R dR R^2 dR^2']],
        ['TissueRegression',        4, ['WMCSF+dt+sq', 'wholebrain']],
        ['GlobalSignalRegression',  4, ['GS+dt+sq']],
        ['Scrubbing',               4, ['RMS', 0.25]]
        ],
    'C0': [ # same as C, with very small change to force recomputation after bug discovered in polynomial filtering 2/12/2018
        ['VoxelNormalization',      1, ['demean']],
        ['Detrending',              2, ['poly', 1, 'wholebrain']],
        ['TissueRegression',        3, ['CompCor', 5, 'WMCSF', 'wholebrain']],
        ['TissueRegression',        3, ['GM', 'wholebrain']], 
        ['GlobalSignalRegression',  3, ['GS']],
        ['MotionRegression',        3, ['censoring']],
        ['Scrubbing',               3, ['FD+DVARS', 0.25, 5]], 
        ['TemporalFiltering',       4, ['Butter', 0.009, 0.0801]]
        ]
    }
#----------------------------------


#----------------------------------
# HELPER FUNCTIONS
# several of these functions may not be used 
# for the specific analyses conducted 
# in intelligence.ipynb and personality.ipynb
#----------------------------------

## 
#  @brief Apply filter to regressors 
#  
#  @param  [int] regressors (nTRs,n) array of n regressors to be filtered
#  @param  [int] filtering filtering method, either 'Butter' or 'Gaussian'
#  @param  [int] nTRs number of time points
#  @param  [int] TR repetition time
#  @return [np.array] filtered regressors
#  
def filter_regressors(regressors, filtering, nTRs, TR):
    if len(filtering)==0:
        print('Warning! Missing or wrong filtering flavor. Regressors were not filtered.')
    else:
        if filtering[0] == 'Butter':
            regressors = clean(regressors, detrend=False, standardize=False, 
                                  t_r=TR, high_pass=filtering[1], low_pass=filtering[2])
        elif filtering[0] == 'Gaussian':
            w = signal.gaussian(11,std=filtering[1])
            regressors = signal.lfilter(w,1,regressors, axis=0)  
    return regressors

## 
#  @brief Apply voxel-wise linear regression to input image
#  
#  @param  [numpy.array] data input image
#  @param  [int] nTRs number of time points
#  @param  [float] TR repetition time
#  @param  [numpy.array] regressors (nTRs,n) array of n regressors
#  @param  [bool] preWhitening True if preWhitening should be applied
#  @return [numpy.array] residuals of regression, same dimensions as data
#  	
def regress(data, nTRs, TR, regressors, preWhitening=False):
    print('Starting regression with {} regressors...'.format(regressors.shape[1]))
    if preWhitening:
        W = prewhitening(data, nTRs, TR, regressors)
        data = np.dot(data,W)
        regressors = np.dot(W,regressors)
    X  = np.concatenate((np.ones([nTRs,1]), regressors), axis=1)
    N = data.shape[0]
    start_time = time()
    fit = np.linalg.lstsq(X, data.T, rcond=None)[0]
    fittedvalues = np.dot(X, fit)
    resid = data - fittedvalues.T
    data = resid
    elapsed_time = time() - start_time
    print('Regression completed in {:02d}h{:02d}min{:02d}s'.format(int(np.floor(elapsed_time/3600)),int(np.floor((elapsed_time%3600)/60)),int(np.floor(elapsed_time%60)))) 
    return data
	
## 
#  @brief Create Legendre polynomial regressor
#  
#  @param  [int] order degree of polynomial
#  @param  [int] nTRs number of time points
#  @return [numpy.array] polynomial regressors
#  
def legendre_poly(order, nTRs):
    # ** a) create polynomial regressor **
    x = np.arange(nTRs)
    x = x - x.max()/2
    num_pol = range(order+1)
    y = np.ones((len(num_pol),len(x)))   
    coeff = np.eye(order+1)
    
    for i in num_pol:
        myleg = Legendre(coeff[i])
        y[i,:] = myleg(x) 
        if i>0:
            y[i,:] = y[i,:] - np.mean(y[i,:])
            y[i,:] = y[i,:]/np.max(y[i,:])
    return y
	
## 
#  @brief Load Nifti data 
#  
#  @param  [str] volFile filename of volumetric file to be loaded
#  @param  [numpy.array] maskAll whole brain mask
#  @param  [bool] unzip True if memmap should be used to load data
#  @return [tuple] image data, no. of rows in image, no. if columns in image, no. of slices in image, no. of time points, affine matrix and repetition time
#  
def load_img(volFile,maskAll=None,unzip=config.useMemMap):
    if unzip:
        volFileUnzip = volFile.replace('.gz','') 
        if not op.isfile(volFileUnzip):
            with open(volFile, 'rb') as fFile:
                decompressedFile = gzip.GzipFile(fileobj=fFile)
                with open(volFileUnzip, 'wb') as outfile:
                    outfile.write(decompressedFile.read())
        img = nib.load(volFileUnzip)
    else:
        img = nib.load(volFile)

    try:
        nRows, nCols, nSlices, nTRs = img.header.get_data_shape()
    except:
        nRows, nCols, nSlices = img.header.get_data_shape()
        nTRs = 1
    TR = img.header.structarr['pixdim'][4]

    if unzip:
        data = np.memmap(volFile, dtype=img.header.get_data_dtype(), mode='c', order='F',
            offset=img.dataobj.offset,shape=img.header.get_data_shape())
        if nTRs==1:
            data = data.reshape(nRows*nCols*nSlices, order='F')[maskAll,:]
        else:
            data = data.reshape((nRows*nCols*nSlices,data.shape[3]), order='F')[maskAll,:]
    else:
        if nTRs==1:
            data = np.asarray(img.dataobj).reshape(nRows*nCols*nSlices, order='F')[maskAll]
        else:
            data = np.asarray(img.dataobj).reshape((nRows*nCols*nSlices,nTRs), order='F')[maskAll,:]

    return data, nRows, nCols, nSlices, nTRs, img.affine, TR, img.header
	
## 
#  @brief Create whole brain and tissue masks
#  
#  @param  [bool] overwrite True if existing files should be overwritten
#  @return [tuple] whole brain, white matter, cerebrospinal fluid and gray matter masks
#  
def makeTissueMasks(overwrite=False,precomputed=False):
    if config.isCifti:
        prefix = config.session+'_' if  hasattr(config,'session')  else '' 
        fmriFile = op.join(buildpath(), prefix+config.fmriRun+'.nii.gz')
    else:
        fmriFile = config.fmriFile
    WMmaskFileout = op.join(outpath(), 'WMmask.nii')
    CSFmaskFileout = op.join(outpath(), 'CSFmask.nii')
    GMmaskFileout = op.join(outpath(), 'GMmask.nii')
    
    if not op.isfile(GMmaskFileout) or overwrite:
        # load ribbon.nii.gz and wmparc.nii.gz
        ribbonFilein = op.join(config.DATADIR, config.subject, 'MNINonLinear','ribbon.nii.gz')
        wmparcFilein = op.join(config.DATADIR, config.subject, 'MNINonLinear', 'wmparc.nii.gz')
        # make sure it is resampled to the same space as the functional run
        ribbonFileout = op.join(outpath(), 'ribbon.nii.gz')
        wmparcFileout = op.join(outpath(), 'wmparc.nii.gz')
        # make identity matrix to feed to flirt for resampling
        ribbonMat = op.join(outpath(), 'ribbon_flirt_{}.mat'.format(config.pipelineName))
        wmparcMat = op.join(outpath(), 'wmparc_flirt_{}.mat'.format(config.pipelineName))
        eyeMat = op.join(outpath(), 'eye_{}.mat'.format(config.pipelineName))
        with open(eyeMat,'w') as fid:
            fid.write('1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1')

        
        flirt_ribbon = fsl.FLIRT(in_file=ribbonFilein, out_file=ribbonFileout,
                                reference=fmriFile, apply_xfm=True,
                                in_matrix_file=eyeMat, out_matrix_file=ribbonMat, interp='nearestneighbour')
        flirt_ribbon.run()

        flirt_wmparc = fsl.FLIRT(in_file=wmparcFilein, out_file=wmparcFileout,
                                reference=fmriFile, apply_xfm=True,
                                in_matrix_file=eyeMat, out_matrix_file=wmparcMat, interp='nearestneighbour')

        flirt_wmparc.run()
        
        # load nii (ribbon & wmparc)
        ribbon = np.asarray(nib.load(ribbonFileout).dataobj)
        wmparc = np.asarray(nib.load(wmparcFileout).dataobj)
        
        # white & CSF matter mask
        # indices are from FreeSurferColorLUT.txt
        
        # Left-Cerebral-White-Matter, Right-Cerebral-White-Matter
        ribbonWMstructures = [2, 41]
        # Left-Cerebral-Cortex, Right-Cerebral-Cortex
        ribbonGMstrucures = [3, 42]
        # Cerebellar-White-Matter-Left, Brain-Stem, Cerebellar-White-Matter-Right
        wmparcWMstructures = [7, 16, 46]
        # Left-Cerebellar-Cortex, Right-Cerebellar-Cortex, Thalamus-Left, Caudate-Left
        # Putamen-Left, Pallidum-Left, Hippocampus-Left, Amygdala-Left, Accumbens-Left 
        # Diencephalon-Ventral-Left, Thalamus-Right, Caudate-Right, Putamen-Right
        # Pallidum-Right, Hippocampus-Right, Amygdala-Right, Accumbens-Right
        # Diencephalon-Ventral-Right
        wmparcGMstructures = [8, 47, 10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60]
        # Fornix, CC-Posterior, CC-Mid-Posterior, CC-Central, CC-Mid-Anterior, CC-Anterior
        wmparcCCstructures = [250, 251, 252, 253, 254, 255]
        # Left-Lateral-Ventricle, Left-Inf-Lat-Vent, 3rd-Ventricle, 4th-Ventricle, CSF
        # Left-Choroid-Plexus, Right-Lateral-Ventricle, Right-Inf-Lat-Vent, Right-Choroid-Plexus
        wmparcCSFstructures = [4, 5, 14, 15, 24, 31, 43, 44, 63]
        
        # make masks
        WMmask = np.double(np.logical_and(np.logical_and(np.logical_or(np.logical_or(np.in1d(ribbon, ribbonWMstructures),
                                                                              np.in1d(wmparc, wmparcWMstructures)),
                                                                np.in1d(wmparc, wmparcCCstructures)),
                                                  np.logical_not(np.in1d(wmparc, wmparcCSFstructures))),
                                   np.logical_not(np.in1d(wmparc, wmparcGMstructures))))
        CSFmask = np.double(np.in1d(wmparc, wmparcCSFstructures))
        GMmask = np.double(np.logical_or(np.in1d(ribbon,ribbonGMstrucures),np.in1d(wmparc,wmparcGMstructures)))
        
        # write masks
        ref = nib.load(wmparcFileout)
        img = nib.Nifti1Image(WMmask.reshape(ref.shape).astype('<f4'), ref.affine)
        nib.save(img, WMmaskFileout)
        
        img = nib.Nifti1Image(CSFmask.reshape(ref.shape).astype('<f4'), ref.affine)
        nib.save(img, CSFmaskFileout)
        
        img = nib.Nifti1Image(GMmask.reshape(ref.shape).astype('<f4'), ref.affine)
        nib.save(img, GMmaskFileout)
        
        # delete temporary files
        cmd = 'rm {} {} {}'.format(eyeMat, ribbonMat, wmparcMat)
        call(cmd,shell=True)
        
        
    tmpWM = nib.load(WMmaskFileout)
    nRows, nCols, nSlices = tmpWM.header.get_data_shape()
    maskWM = np.asarray(tmpWM.dataobj).reshape(nRows*nCols*nSlices, order='F') > 0

    tmpCSF = nib.load(CSFmaskFileout)
    maskCSF = np.asarray(tmpCSF.dataobj).reshape(nRows*nCols*nSlices, order='F')  > 0

    tmpGM = nib.load(GMmaskFileout)
    maskGM = np.asarray(tmpGM.dataobj).reshape(nRows*nCols*nSlices, order='F') > 0

    maskAll  = np.logical_or(np.logical_or(maskWM, maskCSF), maskGM)
    maskWM_  = maskWM[maskAll]
    maskCSF_ = maskCSF[maskAll]
    maskGM_  = maskGM[maskAll]

    return maskAll, maskWM_, maskCSF_, maskGM_

def extract_noise_components(niiImg, WMmask, CSFmask, num_components=5, flavor=None):
    """
    Largely based on https://github.com/nipy/nipype/blob/master/examples/
    rsfmri_vol_surface_preprocessing_nipy.py#L261
    Derive components most reflective of physiological noise according to
    aCompCor method (Behzadi 2007)
    Parameters
    ----------
    niiImg: raw data
    num_components: number of components to use for noise decomposition
    extra_regressors: additional regressors to add
    Returns
    -------
    components: n_time_points x regressors
    """
    if flavor == 'WMCSF' or flavor == None:
        niiImgWMCSF = niiImg[np.logical_or(WMmask,CSFmask),:] 

        niiImgWMCSF[np.isnan(np.sum(niiImgWMCSF, axis=1)), :] = 0
        # remove mean and normalize by variance
        # voxel_timecourses.shape == [nvoxels, time]
        X = niiImgWMCSF.T
        stdX = np.std(X, axis=0)
        stdX[stdX == 0] = 1.
        stdX[np.isnan(stdX)] = 1.
        stdX[np.isinf(stdX)] = 1.
        X = (X - np.mean(X, axis=0)) / stdX
        u, _, _ = linalg.svd(X, full_matrices=False)
        components = u[:, :num_components]
    elif flavor == 'WM+CSF':    
        niiImgWM = niiImg[WMmask,:] 
        niiImgWM[np.isnan(np.sum(niiImgWM, axis=1)), :] = 0
        niiImgCSF = niiImg[CSFmask,:] 
        niiImgCSF[np.isnan(np.sum(niiImgCSF, axis=1)), :] = 0
        # remove mean and normalize by variance
        # voxel_timecourses.shape == [nvoxels, time]
        X = niiImgWM.T
        stdX = np.std(X, axis=0)
        stdX[stdX == 0] = 1.
        stdX[np.isnan(stdX)] = 1.
        stdX[np.isinf(stdX)] = 1.
        X = (X - np.mean(X, axis=0)) / stdX
        u, _, _ = linalg.svd(X, full_matrices=False)
        components = u[:, :num_components]
        X = niiImgCSF.T
        stdX = np.std(X, axis=0)
        stdX[stdX == 0] = 1.
        stdX[np.isnan(stdX)] = 1.
        stdX[np.isinf(stdX)] = 1.
        X = (X - np.mean(X, axis=0)) / stdX
        u, _, _ = linalg.svd(X, full_matrices=False)
        components = np.hstack((components, u[:, :num_components]))
    return components
	
## 
#  @brief Create a XML file describing preprocessing steps
#  
#  @param [str] inFile filename of input image data
#  @param [str] dataDir data directory path
#  @param [list] operations pipeline operations
#  @param [float] startTime start time of preprocessing
#  @param [float] endTime end time of preprocessing
#  @param [str] fname filename of XML log file
#  
def conf2XML(inFile, dataDir, operations, startTime, endTime, fname):
    doc = ET.Element("pipeline")
    
    nodeInput = ET.SubElement(doc, "input")
    nodeInFile = ET.SubElement(nodeInput, "inFile")
    nodeInFile.text = inFile
    nodeDataDir = ET.SubElement(nodeInput, "dataDir")
    nodeDataDir.text = dataDir
    
    nodeDate = ET.SubElement(doc, "date")
    nodeDay = ET.SubElement(nodeDate, "day")
    day = strftime("%Y-%m-%d", localtime())
    nodeDay.text = day
    stime = strftime("%H:%M:%S", startTime)
    etime = strftime("%H:%M:%S", endTime)
    nodeStart = ET.SubElement(nodeDate, "timeStart")
    nodeStart.text = stime
    nodeEnd = ET.SubElement(nodeDate, "timeEnd")
    nodeEnd.text = etime
    
    nodeSteps = ET.SubElement(doc, "steps")
    for op in operations:
        if op[1] == 0: continue
        nodeOp = ET.SubElement(nodeSteps, "operation", name=op[0])
        nodeOrder = ET.SubElement(nodeOp, "order")
        nodeOrder.text = str(op[1])
        nodeFlavor = ET.SubElement(nodeOp, "flavor")
        nodeFlavor.text = str(op[2])
    tree = ET.ElementTree(doc)
    tree.write(fname)
	
## 
#  @brief Create string timestamp
#  
#  @return [str] timestamp
#  
def timestamp():
   now          = time()
   loctime      = localtime(now)
   milliseconds = '%03d' % int((now - int(now)) * 1000)
   return strftime('%Y%m%d%H%M%S', loctime) + milliseconds


## 
#  @brief Submit jobs with sge qsub
#      
def fnSubmitToCluster(strScript, strJobFolder, strJobUID, resources='-l mem_free=25G -pe openmp 6', specifyqueue='long.q'):
    # clean up .o and .e
    tmpfname = op.join(strJobFolder,strJobUID)
    try: 
        remove(tmpfname+'.e')
    except OSError: 
        pass
    try: 
        remove(tmpfname+'.o')  
    except OSError: 
        pass    
   
    strCommand = 'qsub {} -cwd -V {} -N {} -e {} -o {} {}'.format(specifyqueue,resources,strJobUID,
                      op.join(strJobFolder,strJobUID+'.e'), op.join(strJobFolder,strJobUID+'.o'), strScript)
    # write down the command to a file in the job folder
    with open(op.join(strJobFolder,strJobUID+'.cmd'),'w+') as hFileID:
        hFileID.write(strCommand+'\n')
    # execute the command
    cmdOut = check_output(strCommand, shell=True)
    return cmdOut.split()[2]    


## 
#  @brief Submit array of jobs with sge qsub (needs to be customized)
#  
def fnSubmitJobArrayFromJobList():
    config.tStamp = timestamp()
    # make directory
    mkdir('tmp{}'.format(config.tStamp))
    # write a temporary file with the list of scripts to execute as an array job
    with open(op.join('tmp{}'.format(config.tStamp),'scriptlist'),'w') as f:
        f.write('\n'.join(config.scriptlist))
    # write the .qsub file
    with open(op.join('tmp{}'.format(config.tStamp),'qsub'),'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#$ -S /bin/bash\n')
        f.write('#$ -t 1-{} -tc 8\n'.format(len(config.scriptlist)))
        f.write('#$ -cwd -V -N tmp{}\n'.format(config.tStamp))
        f.write('#$ -e {}\n'.format(op.join('tmp{}'.format(config.tStamp),'err')))
        f.write('#$ -o {}\n'.format(op.join('tmp{}'.format(config.tStamp),'out')))
        f.write('#$ {}\n'.format(config.sgeopts))
        f.write('SCRIPT=$(awk "NR==$SGE_TASK_ID" {})\n'.format(op.join('tmp{}'.format(config.tStamp),'scriptlist')))
        f.write('bash $SCRIPT\n')
    strCommand = 'qsub {}'.format(op.join('tmp{}'.format(config.tStamp),'qsub'))
    #strCommand = 'ssh csclprd3s1 "cd {};qsub {}"'.format(getcwd(),op.join('tmp{}'.format(config.tStamp),'qsub'))
    # write down the command to a file in the job folder
    with open(op.join('tmp{}'.format(config.tStamp),'cmd'),'w+') as f:
        f.write(strCommand+'\n')
    # execute the command
    cmdOut = check_output(strCommand, shell=True)
    
    config.scriptlist = []
    return cmdOut.split()[2]    

## 
#  @brief Submit jobs with sge qsub
#      
def fnSubmitToCluster(strScript, strJobFolder, strJobUID, resources='-l mem_free=25G -pe openmp 6', specifyqueue='long.q'):
					 
    # clean up .o and .e
    tmpfname = op.join(strJobFolder,strJobUID)
    try: 
        remove(tmpfname+'.e')
    except OSError: 
        pass
    try: 
        remove(tmpfname+'.o')  
    except OSError: 
        pass    
   
    strCommand = 'qsub {} -cwd -V {} -N {} -e {} -o {} {}'.format(specifyqueue,resources,strJobUID,
                      op.join(strJobFolder,strJobUID+'.e'), op.join(strJobFolder,strJobUID+'.o'), strScript)
    # write down the command to a file in the job folder
    with open(op.join(strJobFolder,strJobUID+'.cmd'),'w+') as hFileID:
        hFileID.write(strCommand+'\n')
    # execute the command
    cmdOut = check_output(strCommand, shell=True)
    return cmdOut.split()[2]    

## 
#  @brief Return list of files ordered by edit time
#  
#  @param  [str] path directory of files to be listed
#  @param  [bool] reverseOrder True if files should be sorted by most recent
#  @return [list] file list ordered by edit time
#  
def sorted_ls(path, reverseOrder):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
    return list(sorted(os.listdir(path), key=mtime, reverse=reverseOrder))

## 
#  @brief Check if preprocessing has already been performed
#  
#  @param [str] inFile input image data file
#  @param [list] operations pipeline operations
#  @param [list] params pipeline operation params
#  @param [str] resDir directory path where results are stored
#  @param [bool] useMostRecent True if most recent files should be checked first
#  @return [str] preprocessed image file name if exists, None otherwise
#  	
def checkXML(inFile, operations, params, resDir, isCifti=False, useMostRecent=True):
    fileList = sorted_ls(resDir, useMostRecent)
    for xfile in fileList:
        if fnmatch.fnmatch(op.join(resDir,xfile), op.join(resDir,'????????.xml')):
            tree = ET.parse(op.join(resDir,xfile))
            root = tree.getroot()
            tvalue = op.basename(root[0][0].text) == op.basename(inFile)
            if not tvalue:
                continue
            if len(root[2]) != np.sum([len(ops) for ops in operations.values()]):
                continue
            try:
                if max([int(el[0].text) for el in root[2]]) != len(operations):
                    continue
            except:
               continue
            for el in root[2]:
                try:
                    tvalue = tvalue and (el.attrib['name'] in operations[int(el[0].text)])
                    tvalue = tvalue and (el[1].text in [repr(param) for param in params[int(el[0].text)]])
                except:
                    tvalue = False
            if not tvalue:
                continue
            else:    
                rcode = xfile.replace('.xml','')
                return op.join(resDir,config.fmriRun+'_prepro_'+rcode+config.ext)
    return None
	
## 
#  @brief Extract random identifier from preprocessed image filename
#  
#  @param  [str] mystring preprocessed image filename
#  @return [str] string identifier
#  
def get_rcode(mystring):
    if not config.isCifti:
        return re.search('.*_(........)\.nii.gz', mystring).group(1)
    else:
        return re.search('.*_(........)\.dtseries.nii', mystring).group(1)

def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024*1024)

def rawgencount(filename):
    f = open(filename, 'rb')
    f_gen = _make_gen(f.read)
    return sum( buf.count(b'\n') for buf in f_gen )        

"""
The following functions implement the ICA-AROMA algorithm (Pruim et al. 2015) 
and are adapted from https://github.com/rhr-pruim/ICA-AROMA
"""

def feature_time_series(melmix, mc):
    """ This function extracts the maximum RP correlation feature scores. 
    It determines the maximum robust correlation of each component time-series 
    with a model of 72 realigment parameters.

    Parameters
    ---------------------------------------------------------------------------------
    melmix:     Full path of the melodic_mix text file
    mc:     Full path of the text file containing the realignment parameters
    
    Returns
    ---------------------------------------------------------------------------------
    maxRPcorr:  Array of the maximum RP correlation feature scores for the components of the melodic_mix file"""

    # Read melodic mix file (IC time-series), subsequently define a set of squared time-series
    mix = np.loadtxt(melmix)
    mixsq = np.power(mix,2)

    # Read motion parameter file
    RP6 = np.loadtxt(mc)[:,:6]

    # Determine the derivatives of the RPs (add zeros at time-point zero)
    RP6_der = np.array(RP6[list(range(1,RP6.shape[0])),:] - RP6[list(range(0,RP6.shape[0]-1)),:])
    RP6_der = np.concatenate((np.zeros((1,6)),RP6_der),axis=0)

    # Create an RP-model including the RPs and its derivatives
    RP12 = np.concatenate((RP6,RP6_der),axis=1)

    # Add the squared RP-terms to the model
    RP24 = np.concatenate((RP12,np.power(RP12,2)),axis=1)

    # Derive shifted versions of the RP_model (1 frame for and backwards)
    RP24_1fw = np.concatenate((np.zeros((1,24)),np.array(RP24[list(range(0,RP24.shape[0]-1)),:])),axis=0)
    RP24_1bw = np.concatenate((np.array(RP24[list(range(1,RP24.shape[0])),:]),np.zeros((1,24))),axis=0)

    # Combine the original and shifted mot_pars into a single model
    RP_model = np.concatenate((RP24,RP24_1fw,RP24_1bw),axis=1)

    # Define the column indices of respectively the squared or non-squared terms
    idx_nonsq = np.array(np.concatenate((list(range(0,12)), list(range(24,36)), list(range(48,60))),axis=0))
    idx_sq = np.array(np.concatenate((list(range(12,24)), list(range(36,48)), list(range(60,72))),axis=0))

    # Determine the maximum correlation between RPs and IC time-series
    nSplits=int(1000)
    maxTC = np.zeros((nSplits,mix.shape[1]))
    for i in range(0,nSplits):
        # Get a random set of 90% of the dataset and get associated RP model and IC time-series matrices
        idx = np.array(random.sample(list(range(0,mix.shape[0])),int(round(0.9*mix.shape[0]))))
        RP_model_temp = RP_model[idx,:]
        mix_temp = mix[idx,:]
        mixsq_temp = mixsq[idx,:]

        # Calculate correlation between non-squared RP/IC time-series
        RP_model_nonsq = RP_model_temp[:,idx_nonsq]
        cor_nonsq = np.array(np.zeros((mix_temp.shape[1],RP_model_nonsq.shape[1])))
        for j in range(0,mix_temp.shape[1]):
            for k in range(0,RP_model_nonsq.shape[1]):
                cor_temp = np.corrcoef(mix_temp[:,j],RP_model_nonsq[:,k])
                cor_nonsq[j,k] = cor_temp[0,1]

        # Calculate correlation between squared RP/IC time-series
        RP_model_sq = RP_model_temp[:,idx_sq]
        cor_sq = np.array(np.zeros((mix_temp.shape[1],RP_model_sq.shape[1])))
        for j in range(0,mixsq_temp.shape[1]):
            for k in range(0,RP_model_sq.shape[1]):
                cor_temp = np.corrcoef(mixsq_temp[:,j],RP_model_sq[:,k])
                cor_sq[j,k] = cor_temp[0,1]

        # Combine the squared an non-squared correlation matrices
        corMatrix = np.concatenate((cor_sq,cor_nonsq),axis=1)

        # Get maximum absolute temporal correlation for every IC
        corMatrixAbs = np.abs(corMatrix)
        maxTC[i,:] = corMatrixAbs.max(axis=1)

    # Get the mean maximum correlation over all random splits
    maxRPcorr = maxTC.mean(axis=0)

    # Return the feature score
    return maxRPcorr

def feature_frequency(melFTmix, TR):
    """ 
    Taken from https://github.com/rhr-pruim/ICA-AROMA
    This function extracts the high-frequency content feature scores. 
    It determines the frequency, as fraction of the Nyquist frequency, 
    at which the higher and lower frequencies explain half of the total power between 0.01Hz and Nyquist. 
    
    Parameters
    ---------------------------------------------------------------------------------
    melFTmix:   Full path of the melodic_FTmix text file
    TR:     TR (in seconds) of the fMRI data (float)
    
    Returns
    ---------------------------------------------------------------------------------
    HFC:        Array of the HFC ('High-frequency content') feature scores for the components of the melodic_FTmix file"""

    
    # Determine sample frequency
    Fs = old_div(1,TR)

    # Determine Nyquist-frequency
    Ny = old_div(Fs,2)
        
    # Load melodic_FTmix file
    FT=np.loadtxt(melFTmix)

    # Determine which frequencies are associated with every row in the melodic_FTmix file  (assuming the rows range from 0Hz to Nyquist)
    f = Ny*(np.array(list(range(1,FT.shape[0]+1))))/(FT.shape[0])

    # Only include frequencies higher than 0.01Hz
    fincl = np.squeeze(np.array(np.where( f > 0.01 )))
    FT=FT[fincl,:]
    f=f[fincl]

    # Set frequency range to [0-1]
    f_norm = old_div((f-0.01),(Ny-0.01))

    # For every IC; get the cumulative sum as a fraction of the total sum
    fcumsum_fract = old_div(np.cumsum(FT,axis=0), np.sum(FT,axis=0))

    # Determine the index of the frequency with the fractional cumulative sum closest to 0.5
    idx_cutoff=np.argmin(np.abs(fcumsum_fract-0.5),axis=0)

    # Now get the fractions associated with those indices index, these are the final feature scores
    HFC = f_norm[idx_cutoff]
         
    # Return feature score
    return HFC

def feature_spatial(fslDir, tempDir, aromaDir, melIC):
    """ 
    Taken from https://github.com/rhr-pruim/ICA-AROMA
    This function extracts the spatial feature scores. 
    For each IC it determines the fraction of the mixture modeled thresholded Z-maps 
    respecitvely located within the CSF or at the brain edges, using predefined standardized masks.

    Parameters
    ---------------------------------------------------------------------------------
    fslDir:     Full path of the bin-directory of FSL
    tempDir:    Full path of a directory where temporary files can be stored (called 'temp_IC.nii.gz')
    aromaDir:   Full path of the ICA-AROMA directory, containing the mask-files (mask_edge.nii.gz, mask_csf.nii.gz & mask_out.nii.gz) 
    melIC:      Full path of the nii.gz file containing mixture-modeled threholded (p>0.5) Z-maps, registered to the MNI152 2mm template
    
    Returns
    ---------------------------------------------------------------------------------
    edgeFract:  Array of the edge fraction feature scores for the components of the melIC file
    csfFract:   Array of the CSF fraction feature scores for the components of the melIC file"""

    EDGEmaskFileout = op.join(outpath(), 'EDGEmask.nii')

    if not op.isfile(EDGEmaskFileout):
        WMmaskFileout = op.join(outpath(), 'WMmask.nii')
        CSFmaskFileout = op.join(outpath(), 'CSFmask.nii')
        GMmaskFileout = op.join(outpath(), 'GMmask.nii')
        OUTmaskFileout = op.join(outpath(), 'OUTmask.nii')

        tmpWM = nib.load(WMmaskFileout)
        nRows, nCols, nSlices = tmpWM.header.get_data_shape()
        tmpCSF = nib.load(CSFmaskFileout)
        tmpGM = nib.load(GMmaskFileout)
        maskGM = np.asarray(tmpGM.dataobj).reshape(nRows*nCols*nSlices, order='F') > 0

        GMWMmask = np.logical_or(tmpWM.dataobj,tmpGM.dataobj)
        ALLmask = np.logical_or(GMWMmask, tmpCSF.dataobj)
        ALLclose = binary_closing(ALLmask,structure=generate_binary_structure(3,4))
        OUTmask = binary_erosion(np.logical_not(ALLclose),structure=generate_binary_structure(3,2),border_value=1)
        OUTmask = binary_opening(OUTmask,structure=generate_binary_structure(3,2))
        img = nib.Nifti1Image(OUTmask.astype('<f4'), tmpWM.affine)
        nib.save(img, OUTmaskFileout)

        OUTdil = binary_dilation(OUTmask, structure=generate_binary_structure(3,5),iterations=2)
        GMWMdil = binary_dilation(GMWMmask, structure=generate_binary_structure(3,5))
        CSFdil = binary_dilation(tmpCSF.dataobj, structure=generate_binary_structure(3,5),iterations=2)
        CSFero = binary_erosion(CSFdil, iterations=4)
        EDGEmask = np.logical_or(binary_opening(np.logical_and(CSFdil,GMWMdil)),binary_closing(np.logical_and(GMWMdil,OUTdil)))
        EDGEmask = np.logical_and(EDGEmask, binary_opening(np.logical_not(CSFero)))
        img = nib.Nifti1Image(EDGEmask.astype('<f4'), tmpWM.affine)
        nib.save(img, EDGEmaskFileout)

    # Get the number of ICs
    numICs = int(getoutput('%sfslinfo %s | grep dim4 | head -n1 | awk \'{print $2}\'' % (fslDir, melIC) ))

    # Loop over ICs
    edgeFract=np.zeros(numICs)
    csfFract=np.zeros(numICs)
    for i in range(0,numICs):
        # Define temporary IC-file
        tempIC = op.join(tempDir,'temp_IC.nii.gz')

        # Extract IC from the merged melodic_IC_thr2MNI2mm file
        os.system(' '.join([op.join(fslDir,'fslroi'),
            melIC,
            tempIC,
            str(i),
            '1']))

        # Change to absolute Z-values
        os.system(' '.join([op.join(fslDir,'fslmaths'),
            tempIC,
            '-abs',
            tempIC]))
        
        # Get sum of Z-values within the total Z-map (calculate via the mean and number of non-zero voxels)
        totVox = int(getoutput(' '.join([op.join(fslDir,'fslstats'),
                            tempIC,
                            '-V | awk \'{print $1}\''])))
        
        if not (totVox == 0):
            totMean = float(getoutput(' '.join([op.join(fslDir,'fslstats'),
                            tempIC,
                            '-M'])))
        else:
            print('     - The spatial map of component ' + str(i+1) + ' is empty. Please check!')
            totMean = 0

        totSum = totMean * totVox
        
        # Get sum of Z-values of the voxels located within the CSF (calculate via the mean and number of non-zero voxels)
        csfVox = int(getoutput(' '.join([op.join(fslDir,'fslstats'),
                            tempIC,
                            '-k {}/CSFmask.nii'.format(aromaDir),
                            '-V | awk \'{print $1}\''])))

        if not (csfVox == 0):
            csfMean = float(getoutput(' '.join([op.join(fslDir,'fslstats'),
                            tempIC,
                            '-k {}/CSFmask.nii'.format(aromaDir),
                            '-M'])))
        else:
            csfMean = 0

        csfSum = csfMean * csfVox   

        # Get sum of Z-values of the voxels located within the Edge (calculate via the mean and number of non-zero voxels)
        edgeVox = int(getoutput(' '.join([op.join(fslDir,'fslstats'),
                            tempIC,
                            '-k {}/EDGEmask.nii'.format(aromaDir),
                            '-V | awk \'{print $1}\''])))
        if not (edgeVox == 0):
            edgeMean = float(getoutput(' '.join([op.join(fslDir,'fslstats'),
                            tempIC,
                            '-k {}/EDGEmask.nii'.format(aromaDir),
                            '-M'])))
        else:
            edgeMean = 0
        
        edgeSum = edgeMean * edgeVox

        # Get sum of Z-values of the voxels located outside the brain (calculate via the mean and number of non-zero voxels)
        outVox = int(getoutput(' '.join([op.join(fslDir,'fslstats'),
                            tempIC,
                            '-k {}/OUTmask.nii'.format(aromaDir),
                            '-V | awk \'{print $1}\''])))
        if not (outVox == 0):
            outMean = float(getoutput(' '.join([op.join(fslDir,'fslstats'),
                            tempIC,
                            '-k {}/OUTmask.nii'.format(aromaDir),
                            '-M'])))
        else:
            outMean = 0
        
        outSum = outMean * outVox

        # Determine edge and CSF fraction
        if not (totSum == 0):
            edgeFract[i] = old_div((outSum + edgeSum),(totSum - csfSum))
            csfFract[i] = old_div(csfSum, totSum)
        else:
            edgeFract[i]=0
            csfFract[i]=0

    # Remove the temporary IC-file
    remove(tempIC)

    # Return feature scores
    return edgeFract, csfFract

def classification(outDir, maxRPcorr, edgeFract, HFC, csfFract):
    """ 
    Taken from https://github.com/rhr-pruim/ICA-AROMA
    This function classifies a set of components into motion and non-motion 
    components based on four features; maximum RP correlation, high-frequency content, 
    edge-fraction and CSF-fraction

    Parameters
    ---------------------------------------------------------------------------------
    outDir:     Full path of the output directory
    maxRPcorr:  Array of the 'maximum RP correlation' feature scores of the components
    edgeFract:  Array of the 'edge fraction' feature scores of the components
    HFC:        Array of the 'high-frequency content' feature scores of the components
    csfFract:   Array of the 'CSF fraction' feature scores of the components

    Return
    ---------------------------------------------------------------------------------
    motionICs   Array containing the indices of the components identified as motion components

    Output (within the requested output directory)
    ---------------------------------------------------------------------------------
    classified_motion_ICs.txt   A text file containing the indices of the components identified as motion components """

    # Classify the ICs as motion or non-motion

    # Define criteria needed for classification (thresholds and hyperplane-parameters)
    thr_csf = 0.10
    thr_HFC = 0.35
    hyp = np.array([-19.9751070082159, 9.95127547670627, 24.8333160239175])
    
    # Project edge & maxRPcorr feature scores to new 1D space
    x = np.array([maxRPcorr, edgeFract])
    proj = hyp[0] + np.dot(x.T,hyp[1:])

    # Classify the ICs
    motionICs = np.squeeze(np.array(np.where((proj > 0) + (csfFract > thr_csf) + (HFC > thr_HFC))))

    return motionICs

def denoising(fslDir, inFile, outDir, melmix, denType, denIdx):
    """ 
    Taken from https://github.com/rhr-pruim/ICA-AROMA
    This function classifies the ICs based on the four features; maximum RP correlation, high-frequency content, edge-fraction and CSF-fraction

    Parameters
    ---------------------------------------------------------------------------------
    fslDir:     Full path of the bin-directory of FSL
    inFile:     Full path to the data file (nii.gz) which has to be denoised
    outDir:     Full path of the output directory
    melmix:     Full path of the melodic_mix text file
    denType:    Type of requested denoising ('aggr': aggressive, 'nonaggr': non-aggressive, 'both': both aggressive and non-aggressive 
    denIdx:     Indices of the components that should be regressed out

    Output (within the requested output directory)
    ---------------------------------------------------------------------------------
    denoised_func_data_<denType>.nii.gz:        A nii.gz file of the denoised fMRI data"""

    # Check if denoising is needed (i.e. are there components classified as motion)
    check = len(denIdx) > 0

    if check==1:
        # Put IC indices into a char array
        denIdxStr = np.char.mod('%i',(denIdx+1))

        # Non-aggressive denoising of the data using fsl_regfilt (partial regression), if requested
        if (denType == 'nonaggr') or (denType == 'both'):       
            os.system(' '.join([op.join(fslDir,'fsl_regfilt'),
                '--in=' + inFile,
                '--design=' + melmix,
                '--filter="' + ','.join(denIdxStr) + '"',
                '--out=' + op.join(outDir,'denoised_func_data_nonaggr.nii.gz')]))

        # Aggressive denoising of the data using fsl_regfilt (full regression)
        if (denType == 'aggr') or (denType == 'both'):
            os.system(' '.join([op.join(fslDir,'fsl_regfilt'),
                '--in=' + inFile,
                '--design=' + melmix,
                '--filter="' + ','.join(denIdxStr) + '"',
                '--out=' + op.join(outDir,'denoised_func_data_aggr.nii.gz'),
                '-a']))
    else:
        print("  - None of the components was classified as motion, so no denoising is applied (a symbolic link to the input file will be created).")
        if (denType == 'nonaggr') or (denType == 'both'):
            os.symlink(inFile,op.join(outDir,'denoised_func_data_nonaggr.nii.gz'))
        if (denType == 'aggr') or (denType == 'both'):
            os.symlink(inFile,op.join(outDir,'denoised_func_data_aggr.nii.gz'))


## 
#  @brief Replace censored time point by linear interpolation
#  
#  @param  [numpy.array] data input image data
#  @param  [numpy.array] censored volumes to be censored
#  @param  [float] TR repetition time
#  @param  [int] nTRs number of time points
#  @return [numpy.array]image data with interpolated values in censored time points
#   
def interpolate(data,censored,TR,nTRs,method='linear'):
    N = data.shape[0]
    tpoints = np.setdiff1d(np.arange(nTRs),censored)
    for i in range(N):
        tseries = data[i,:]
        cens_tseries = tseries[tpoints]
        if method=='linear':
            # linear interpolation
            intpts = np.interp(censored,tpoints,cens_tseries)
            tseries[censored] = intpts
            data[i,:] = tseries
        elif method=='power':
            # as in Power et. al 2014 a frequency transform is used to generate data with
            # the same phase and spectral characteristics as the unflagged data
            N = len(tpoints) # no. of time points
            T = (tpoints.max() - tpoints.min())*TR # total time span
            ofac = 8 # oversampling frequency (generally >=4)
            hifac = 1 # highest frequency allowed.  hifac = 1 means 1*nyquist limit

            # compute sampling frequencies
            f = np.arange(1/(T*ofac),hifac*N/(2*T)+1/(T*ofac),1/(T*ofac))
            # angular frequencies and constant offsets
            w = 2*np.pi*f
            w = w[:,np.newaxis]
            t = TR*tpoints[:,np.newaxis].T
            tau = np.arctan2(np.sum(np.sin(2*w*(t+1)),1),np.sum(np.cos(2*w*(t+1)),1))/(2*np.squeeze(w))

            # spectral power sin and cosine terms
            sterm = np.sin(w*(t+1) - (np.squeeze(w)*tau)[:,np.newaxis])
            cterm = np.cos(w*(t+1) - (np.squeeze(w)*tau)[:,np.newaxis])

            mean_ct = cens_tseries.mean()
            D = cens_tseries - mean_ct

            c = np.sum(cterm * D,1) / np.sum(np.power(cterm,2),1)
            s = np.sum(sterm * D,1) / np.sum(np.power(sterm,2),1)

            # The inverse function to re-construct the original time series
            full_tpoints = (np.arange(nTRs)[:,np.newaxis]+1).T*TR
            prod = full_tpoints*w
            sin_t = np.sin(prod)
            cos_t = np.cos(prod)
            sw_p = sin_t*s[:,np.newaxis]
            cw_p = cos_t*c[:,np.newaxis]
            S = np.sum(sw_p,axis=0)
            C = np.sum(cw_p,axis=0)
            H = C + S

            # Normalize the reconstructed spectrum, needed when ofac > 1
            Std_H = np.std(H, ddof=1)
            Std_h = np.std(cens_tseries,ddof=1)
            norm_fac = Std_H/Std_h
            H = H/norm_fac
            H = H + mean_ct

            intpts = H[censored]
            tseries[censored] = intpts
            data[i,:] = tseries
        elif method == 'astropy':
            lombs = LombScargle(tpoints*TR, cens_tseries)
            frequency, power = lombs.autopower(normalization='standard', samples_per_peak=8, nyquist_factor=1, method='fast')
            pwsort = np.argsort(power)
            frequency = frequency[pwsort[-100:]]
            mean_ct = np.mean(cens_tseries)
            for f in np.arange(len(frequency)):
                if f == 0:
                    y_all = lombs.model(censored*TR, frequency[f])
                else:
                    y_all = y_all + lombs.model(censored*TR, frequency[f])
            y_all = y_all - mean_ct*len(frequency)
            Std_y = np.std(y_all, ddof=1)
            Std_h = np.std(cens_tseries,ddof=1)
            norm_fac = Std_y/Std_h
            y_final = y_all/norm_fac
            intpts = y_final + np.mean(cens_tseries)
            tseries[censored] = intpts
            data[i,:] = tseries
        else:
            print("Wrong interpolation method: nothing was done")
            break
    return data

def retrieve_preprocessed(inputFile, operations, outputDir, isCifti):
    if not op.isfile(inputFile):
        print(inputFile, 'missing')
        sys.stdout.flush()
        return None

    sortedOperations = sorted(operations, key=operator.itemgetter(1))
    steps            = {}
    Flavors          = {}
    cstep            = 0

    # If requested, scrubbing is performed first, before any denoising step
    scrub_idx = -1
    curr_idx = -1
    for opr in sortedOperations:
        curr_idx = curr_idx+1
        if opr[0] == 'Scrubbing' and opr[1] != 1 and opr[1] != 0:
            scrub_idx = opr[1]
            break
            
    if scrub_idx != -1:        
        for opr in sortedOperations:  
            if opr[1] != 0:
                opr[1] = opr[1]+1

        sortedOperations[curr_idx][1] = 1
        sortedOperations = sorted(operations, key=operator.itemgetter(1))

    prev_step = 0	
    for opr in sortedOperations:
        if opr[1]==0:
            continue
        else:
            if opr[1]!=prev_step:
                cstep=cstep+1
                steps[cstep] = [opr[0]]
                Flavors[cstep] = [opr[2]]
            else:
                steps[cstep].append(opr[0])
                Flavors[cstep].append(opr[2])
            prev_step = opr[1]                
    precomputed = checkXML(inputFile,steps,Flavors,outputDir,isCifti) 
    return precomputed 																								 

def correlationKernel(X1, X2):
    """(Pre)calculates Gram Matrix K"""

    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            gram_matrix[i, j] = stats.pearsonr(x1, x2)[0]
    return gram_matrix

def dctmtx(N):
    """
    Largely based on http://www.mrc-cbu.cam.ac.uk/wp-content/uploads/2013/01/rsfMRI_GLM.m
    """
    K=N
    n = range(N)
    C = np.zeros((len(n), K),dtype=np.float32)
    C[:,0] = np.ones((len(n)),dtype=np.float32)/np.sqrt(N)
    doublen = [2*x+1 for x in n]
    for k in range(1,K):
        C[:,k] = np.sqrt(2/N)*np.cos([np.pi*x*(k-1)/(2*N) for x in doublen])        
    return C 

def distcorr(X, Y):
    """ Compute the distance correlation function
    
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

# the function parcellate() will save a time series per each voxel if config.save_voxelwise = True
def compute_mFC(tsDir, rstring, overwrite=False):
    parcelData = []
    for iParcel in np.arange(config.nParcels):
        tsFileAll = op.join(tsDir,'parcel{:03d}_{}_all.txt'.format(iParcel+1,rstring))
        if not op.isfile(tsFileAll) or overwrite:
            parcelData.append(np.transpose(data[np.where(allparcels==iParcel+1)[0],:]))
        else:
            parcelData.append(np.loadtxt(tsFileAll,delimiter=','))
        mFC = np.zeros([config.nParcels, config.nParcels])
    for iParcel in np.arange(config.nParcels):
        for jParcel in np.arange(config.nParcels):
            if iParcel>jParcel:
                mFC[iParcel,jParcel] = distcorr(parcelData[iParcel],parcelData[jParcel])
                mFC[jParcel,iParcel] = mFC[iParcel,jParcel]
    return mFC

# Input
#
# 'TS' : original time course encoded in a matrix of size (T=number of time points,k=number of variables/ROIs)
# 'p'  : order of the AR model to be identified
#
# Given these inputs, the AR model of order 'p' reads:
#
# TS(t,:)' = w + \sum_i=1^p A_i * TS(t-i,:)' + e(t),     Eq. (1)
#
# where: w    is an intercept vector of size (T,1). It is supposed to be
#             zero if timecourses are centered.
#        A_i  are matrices of size k*k linking T(t,:) and T(t-i,:).
#        e(t) is an error vector of size (T,1) following a centered multivariate gaussian distribution
#
# Eq. (1) can be written for all t \in [p+1...T]. Concatenating these
# (T-p) equations yields the following matrix form:
#
# Y = B*Z+E,    Eq. (2)
#
# where: Y = [TS(p+1,:)' TS(p+2,:)' ... TS(T,:)'] is a matrix of size (k,T-p)
#
#        B = [w A_1 ... A_p] is a matrix of size (k,k*p+1) that
#        gathers unknown parameters of the model
#
#            |    1             1           ...        1     |
#            | TS(p,:)'     TS(p+1,:)'      ...    TS(T-1,:)'|
#            |TS(p-1,:)'     TS(p,:)'       ...    TS(T-2,:)'|
#        Z = |    .             .          .           .     |
#            |    .             .            .         .     |
#            |    .             .              .       .     |
#            | TS(1,:)'      TS(2,:)'       ...    TS(T-p,:)'|
#
#        is a matrix of size (k*p+1,T-p) that is directly built from the input TS.
#
#        E = [e(p+1) e(p+2) ... e(T)] is a matrix of size (k,T-p)
#        containing the residuals of the multivariate AR model.
#
# Output
#
# 'Y'             Matrix variables directly built from TS (see Eq. (2))
# 'B'             Matrix containing AR model parameters
# 'Z'             Matrix variables directly built from TS (see Eq. (2))
# 'E'             Residuals of the AR model
def ar_mls(TS,p):
    T = TS.shape[0]
    k = TS.shape[1]
    Z = np.zeros([k*p+1,T-p])
    Y = TS[p:,:].T
    Z[0,:] = 1
    for j in range(p):
        Z[j*k+1 : (j+1)*k+1, :] = TS[p-(j+1):T-(j+1), :].T

    B = linalg.lstsq(np.dot(Z,Z.T).T, np.dot(Y,Z.T).T)[0].T
    E   = Y-np.dot(B,Z)
    return [Y,B,Z,E]

def AR1_FC(alltc):
    n_subj = len(alltc)
    n_rois = alltc[0].shape[1]
    fcMats = np.zeros([n_subj, n_rois*n_rois])
    for i in range(n_subj):
        [_,B,_,_] = ar_mls(alltc[i],1)
        fcMats[i,:] = B[:,1:].reshape(-1)
    return fcMats

def within_sub_edge_var(alltc, measure, sample_size = 0.1, n_rep = 10):
    n_subjects = len(alltc)
    n_rois = alltc[0].shape[1]
    var_edges = np.zeros((n_subjects, int(n_rois*(n_rois-1)/2)))
    for i in range(n_subjects):
        n_timepoints = alltc[i].shape[0]
        sample_timepoints = np.array([random.sample(range(n_timepoints),int(np.round(n_timepoints*sample_size))) for i in range(n_rep)])
        sample_tc = [alltc[i][sample_timepoints]]
        FCsamples = measure.fit_transform([alltc[i][sample_timepoints[j,:],:] for j in range(n_rep)])
        var_edges[i,:] = np.var(FCsamples,axis=0,ddof=1)
    return np.mean(var_edges,axis=0)

def between_sub_edge_var(fcMats):
    return np.var(fcMats, axis=0, ddof=1)

def rank_variance_ratio(between, within):
    ratio = between/within    
    sorted_idx = np.argsort(ratio)
    return sorted_idx[::-1]

# ---------------------
# Pipeline Operations
def TaskRegression(niiImg, flavor, masks, imgInfo):
    nRows, nCols, nSlices, nTRs, affine, TR, header = imgInfo
    trials = get_EVs(buildpath(), flavor[0])
    # sometimes an EV is empty
    # need to drop it
    for k in trials.keys():
        if trials[k].shape[1]==1:
            trials.pop(k, None)
    print(trials)
    frame_times = np.arange(nTRs) * TR
    d = {
        'onset' : np.hstack([trials[k][:,0] for k in trials.keys()]),
        'trial_type' : np.hstack([np.tile(k, len(trials[k])) for k in trials.keys()]),
        'duration' : np.hstack([trials[k][:,1] for k in trials.keys()]),
        'modulation' : np.hstack([trials[k][:,2] for k in trials.keys()])
    }
    df = pd.DataFrame(data=d)
    DM = design_matrix.make_first_level_design_matrix(frame_times=frame_times, events=df, 
                    hrf_model='fir', drift_model=None, oversampling=1)
    DM = DM.drop(labels=['constant'],axis=1)
    return np.array(DM)

def MotionRegression(niiImg, flavor, masks, imgInfo):
    # assumes that data is organized as in the HCP
    motionFile = op.join(buildpath(), config.movementRegressorsFile)
    data = np.genfromtxt(motionFile)
    if flavor[0] == 'R':
        X = data[:,:6]
    elif flavor[0] == 'R dR':
        X = data
    elif flavor[0] == 'R R^2':
        data = data[:,:6]
        data_squared = data ** 2
        X = np.concatenate((data, data_squared), axis=1)
    elif flavor[0] == 'R dR R^2 dR^2':
        data_squared = data ** 2
        X = np.concatenate((data, data_squared), axis=1)
    elif flavor[0] == 'R R^2 R-1 R-1^2':
        data = data[:,:6]
        data_roll = np.roll(data, 1, axis=0)
        data_squared = data ** 2
        data_roll[0] = 0
        data_roll_squared = data_roll ** 2
        X = np.concatenate((data, data_squared, data_roll, data_roll_squared), axis=1)
    elif flavor[0] == 'R R^2 R-1 R-1^2 R-2 R-2^2':
        data = data[:,:6]
        data_roll = np.roll(data, 1, axis=0)
        data_squared = data ** 2
        data_roll[0] = 0
        data_roll_squared = data_roll ** 2
        data_roll2 = np.roll(data_roll, 1, axis=0)
        data_roll2[0] = 0
        data_roll2_squared = data_roll2 ** 2
        X = np.concatenate((data, data_squared, data_roll, data_roll_squared, data_roll2, data_roll2_squared), axis=1)
    elif flavor[0] == 'censoring':
        nRows, nCols, nSlices, nTRs, affine, TR, header = imgInfo
        X = np.empty((nTRs, 0))
    elif flavor[0] == 'ICA-AROMA':
        nRows, nCols, nSlices, nTRs, affine, TR, header = imgInfo
        fslDir = op.join(environ["FSLDIR"],'bin','')
        if hasattr(config,'melodicFolder'):
            icaOut = op.join(buildpath(),config.melodicFolder)
        else:
            icaOut = op.join(outpath(), 'icaOut')
            try:
                mkdir(icaOut)
            except OSError:
                pass
            if not op.isfile(op.join(icaOut,'melodic_IC.nii.gz')):
                if config.isCifti:
                    prefix = config.session+'_' if  hasattr(config,'session')  else '' 
                    fmriFile = op.join(buildpath(), prefix+config.fmriRun+'.nii.gz')
                else:
                    fmriFile = config.fmriFile
                os.system(' '.join([os.path.join(fslDir,'melodic'),
                    '--in=' + fmriFile, 
                    '--outdir=' + icaOut, 
                    '--dim=' + str(min(250,np.int(data.shape[0]/2))),
                    '--Oall --nobet ',
                    '--tr=' + str(TR)]))

        melIC_MNI = op.join(icaOut,'melodic_IC.nii.gz')
        mc = op.join(buildpath(), config.movementRegressorsFile)
        melmix = op.join(icaOut,'melodic_mix')
        melFTmix = op.join(icaOut,'melodic_FTmix')
        
        edgeFract, csfFract = feature_spatial(fslDir, icaOut, outpath(), melIC_MNI)
        maxRPcorr = feature_time_series(melmix, mc)
        HFC = feature_frequency(melFTmix, TR)
        motionICs = classification(icaOut, maxRPcorr, edgeFract, HFC, csfFract)
        
        if motionICs.ndim > 0:
            melmix = op.join(icaOut,'melodic_mix')
            if len(flavor)>1:
                denType = flavor[1]
            else:
                denType = 'aggr'
            if denType == 'aggr':
                X = np.loadtxt(melmix)[:,motionICs]
            elif denType == 'nonaggr':  
                # Partial regression
                X = np.loadtxt(melmix)
                # if filtering has already been performed, regressors need to be filtered too
                if len(config.filtering)>0:
                    nRows, nCols, nSlices, nTRs, affine, TR, header = imgInfo
                    X = filter_regressors(X, config.filtering, nTRs, TR)  

                if config.doScrubbing:
                    nRows, nCols, nSlices, nTRs, affine, TR, header = imgInfo
                    toCensor = np.loadtxt(op.join(outpath(), 'Censored_TimePoints_{}.txt'.format(config.pipelineName)), dtype=np.dtype(np.int32))
                    npts = toCensor.size
                    if npts==1:
                        toCensor=np.reshape(toCensor,(npts,))
                    toReg = np.zeros((nTRs, npts),dtype=np.float32)
                    for i in range(npts):
                        toReg[toCensor[i],i] = 1
                    X = np.concatenate((X, toReg), axis=1)
                    
                nRows, nCols, nSlices, nTRs, affine, TR, header = imgInfo
                niiImg[0] = partial_regress(niiImg[0], nTRs, TR, X, motionICs, config.preWhitening)
                if niiImg[1] is not None:
                    niiImg[1] = partial_regress(niiImg[1], nTRs, TR, X, motionICs, config.preWhitening)
                return niiImg[0],niiImg[1]
            else:
                print('Warning! Wrong ICA-AROMA flavor. Using default full regression.')
                X = np.loadtxt(melmix)[:,motionICs]
        else:
            print('ICA-AROMA: None of the components was classified as motion, so no denoising is applied.')
            X = np.empty((nTRs, 0))
    else:
        print('Wrong flavor, using default regressors: R dR')
        X = data
        
    # if filtering has already been performed, regressors need to be filtered too
    if len(config.filtering)>0 and X.size > 0:
        nRows, nCols, nSlices, nTRs, affine, TR, header = imgInfo
        X = filter_regressors(X, config.filtering, nTRs, TR)  
        
    if config.doScrubbing:
        nRows, nCols, nSlices, nTRs, affine, TR, header = imgInfo
        toCensor = np.loadtxt(op.join(outpath(), 'Censored_TimePoints_{}.txt'.format(config.pipelineName)), dtype=np.dtype(np.int32))
        npts = toCensor.size
        if npts==1:
            toCensor=np.reshape(toCensor,(npts,))
        toReg = np.zeros((nTRs, npts),dtype=np.float32)
        for i in range(npts):
            toReg[toCensor[i],i] = 1
        X = np.concatenate((X, toReg), axis=1)
        
    return X

def Scrubbing(niiImg, flavor, masks, imgInfo):
    """
    Largely based on: 
    - https://git.becs.aalto.fi/bml/bramila/blob/master/bramila_dvars.m
    - https://github.com/poldrack/fmriqa/blob/master/compute_fd.py
    """
    thr = flavor[1]
    nRows, nCols, nSlices, nTRs, affine, TR, header = imgInfo

    if flavor[0] == 'DVARS':
        # pcSigCh
        meanImg = np.mean(niiImg[0],axis=1)[:,np.newaxis]
        close0 = np.where(meanImg < 1e5*np.finfo(np.float).eps)[0]
        if close0.shape[0] > 0:
            meanImg[close0,0] = np.max(np.abs(niiImg[0][close0,:]),axis=1)
            niiImg[0][close0,:] = niiImg[0][close0,:] + meanImg[close0,:]
        niiImg2 = 100 * (niiImg[0] - meanImg) / meanImg
        niiImg2[np.where(np.isnan(niiImg2))] = 0
        dt = np.diff(niiImg2, n=1, axis=1)
        dt = np.concatenate((np.zeros((dt.shape[0],1),dtype=np.float32), dt), axis=1)
        score = np.sqrt(np.mean(dt**2,0))        
        censored = np.where(score>thr)
        np.savetxt(op.join(outpath(), '{}_{}.txt'.format(flavor[0],config.pipelineName)), score, delimiter='\n', fmt='%d')
    elif flavor[0] == 'FD':
        motionFile = op.join(buildpath(), config.movementRegressorsFile)
        dmotpars = np.abs(np.genfromtxt(motionFile)[:,6:]) #derivatives
        disp=dmotpars.copy()
        disp[:,3:]=np.pi*config.headradius*2*(disp[:,3:]/360)
        score=np.sum(disp,1)
        censored = np.where(score>thr)
        np.savetxt(op.join(outpath(), '{}_{}.txt'.format(flavor[0],config.pipelineName)), score, delimiter='\n', fmt='%d')
    elif flavor[0] == 'FD+DVARS':
        motionFile = op.join(buildpath(), config.movementRegressorsFile)
        dmotpars = np.abs(np.genfromtxt(motionFile)[:,6:]) #derivatives
        disp=dmotpars.copy()
        disp[:,3:]=np.pi*config.headradius*2*(disp[:,3:]/360)
        score=np.sum(disp,1)
        # pcSigCh
        meanImg = np.mean(niiImg[0],axis=1)[:,np.newaxis]
        close0 = np.where(meanImg < 1e5*np.finfo(np.float).eps)[0]
        if close0.shape[0] > 0:
            meanImg[close0,0] = np.max(np.abs(niiImg[0][close0,:]),axis=1)
            niiImg[0][close0,:] = niiImg[0][close0,:] + meanImg[close0,:]
        niiImg2 = 100 * (niiImg[0] - meanImg) / meanImg
        niiImg2[np.where(np.isnan(niiImg2))] = 0
        dt = np.diff(niiImg2, n=1, axis=1)
        dt = np.concatenate((np.zeros((dt.shape[0],1),dtype=np.float32), dt), axis=1)
        scoreDVARS = np.sqrt(np.mean(dt**2,0)) 
        # as in Siegel et al. 2016
        cleanFD = clean(score[:,np.newaxis], detrend=False, standardize=False, t_r=TR, low_pass=0.3)
        thr2 = flavor[2]
        censDVARS = scoreDVARS > (100+thr2)/100 * np.median(scoreDVARS)
        censored = np.where(np.logical_or(np.ravel(cleanFD)>thr,censDVARS))
        np.savetxt(op.join(outpath(), 'FD_{}.txt'.format(config.pipelineName)), cleanFD, delimiter='\n', fmt='%f')
        np.savetxt(op.join(outpath(), 'DVARS_{}.txt'.format(config.pipelineName)), scoreDVARS, delimiter='\n', fmt='%f')
    elif flavor[0] == 'RMS':
        RelRMSFile = op.join(buildpath(), config.movementRelativeRMSFile)
        score = np.loadtxt(RelRMSFile)
        censored = np.where(score>thr)
        np.savetxt(op.join(outpath(), '{}_{}.txt'.format(flavor[0],config.pipelineName)), score, delimiter='\n', fmt='%d')
    else:
        print('Wrong scrubbing flavor. Nothing was done')
        return niiImg[0],niiImg[1]
    
    if (len(flavor)>3 and flavor[0] == 'FD+DVARS'):
        pad = flavor[3]
        a_minus = [i-k for i in censored[0] for k in range(1, pad+1)]
        a_plus  = [i+k for i in censored[0] for k in range(1, pad+1)]
        censored = np.concatenate((censored[0], a_minus, a_plus))
        censored = np.unique(censored[np.where(np.logical_and(censored>=0, censored<len(score)))])
    elif len(flavor) > 2 and flavor[0] != 'FD+DVARS':
        pad = flavor[2]
        a_minus = [i-k for i in censored[0] for k in range(1, pad+1)]
        a_plus  = [i+k for i in censored[0] for k in range(1, pad+1)]
        censored = np.concatenate((censored[0], a_minus, a_plus))
        censored = np.unique(censored[np.where(np.logical_and(censored>=0, censored<len(score)))])
    censored = np.ravel(censored)
    toAppend = np.array([])
    for i in range(len(censored)):
        if censored[i] > 0 and censored[i] < 5:
            toAppend = np.union1d(toAppend,np.arange(0,censored[i]))
        elif censored[i] > nTRs - 5:
            toAppend = np.union1d(toAppend,np.arange(censored[i]+1,nTRs))
        elif i<len(censored) - 1:
            gap = censored[i+1] - censored[i] 
            if gap > 1 and gap <= 5:
                toAppend = np.union1d(toAppend,np.arange(censored[i]+1,censored[i+1]))
    censored = np.union1d(censored,toAppend)
    censored.sort()
    censored = censored.astype(int)
    
    np.savetxt(op.join(outpath(), 'Censored_TimePoints_{}.txt'.format(config.pipelineName)), censored, delimiter='\n', fmt='%d')
    if len(censored)>0 and len(censored)<nTRs:
        config.doScrubbing = True
    if len(censored) == nTRs:
        print('Warning! All points selected for censoring: scrubbing will not be performed.')

    #even though these haven't changed, they are returned for consistency with other operations
    return niiImg[0],niiImg[1]

def TissueRegression(niiImg, flavor, masks, imgInfo):
    maskAll, maskWM_, maskCSF_, maskGM_ = masks
    nRows, nCols, nSlices, nTRs, affine, TR, header = imgInfo
    
    if config.isCifti:
        volData = niiImg[1]
    else:
        volData = niiImg[0]


    if flavor[0] == 'CompCor':
        X = extract_noise_components(volData, maskWM_, maskCSF_, num_components=flavor[1], flavor=flavor[2])
    elif flavor[0] == 'WMCSF':
        meanWM = np.mean(np.float32(volData[maskWM_,:]),axis=0)
        meanWM = meanWM - np.mean(meanWM)
        meanWM = meanWM/max(meanWM)
        meanCSF = np.mean(np.float32(volData[maskCSF_,:]),axis=0)
        meanCSF = meanCSF - np.mean(meanCSF)
        meanCSF = meanCSF/max(meanCSF)
        X  = np.concatenate((meanWM[:,np.newaxis], meanCSF[:,np.newaxis]), axis=1)
    elif flavor[0] == 'WMCSF+dt':
        meanWM = np.mean(np.float32(volData[maskWM_,:]),axis=0)
        meanWM = meanWM - np.mean(meanWM)
        meanWM = meanWM/max(meanWM)
        meanCSF = np.mean(np.float32(volData[maskCSF_,:]),axis=0)
        meanCSF = meanCSF - np.mean(meanCSF)
        meanCSF = meanCSF/max(meanCSF)
        dtWM=np.zeros(meanWM.shape,dtype=np.float32)
        dtWM[1:] = np.diff(meanWM, n=1)
        dtCSF=np.zeros(meanCSF.shape,dtype=np.float32)
        dtCSF[1:] = np.diff(meanCSF, n=1)
        X  = np.concatenate((meanWM[:,np.newaxis], meanCSF[:,np.newaxis], 
                             dtWM[:,np.newaxis], dtCSF[:,np.newaxis]), axis=1)
    elif flavor[0] == 'WMCSF+dt+sq':
        meanWM = np.mean(np.float32(volData[maskWM_,:]),axis=0)
        meanWM = meanWM - np.mean(meanWM)
        meanWM = meanWM/max(meanWM)
        meanCSF = np.mean(np.float32(volData[maskCSF_,:]),axis=0)
        meanCSF = meanCSF - np.mean(meanCSF)
        meanCSF = meanCSF/max(meanCSF)
        dtWM=np.zeros(meanWM.shape,dtype=np.float32)
        dtWM[1:] = np.diff(meanWM, n=1)
        dtCSF=np.zeros(meanCSF.shape,dtype=np.float32)
        dtCSF[1:] = np.diff(meanCSF, n=1)
        sqmeanWM = meanWM ** 2
        sqmeanCSF = meanCSF ** 2
        sqdtWM = dtWM ** 2
        sqdtCSF = dtCSF ** 2
        X  = np.concatenate((meanWM[:,np.newaxis], meanCSF[:,np.newaxis], 
                             dtWM[:,np.newaxis], dtCSF[:,np.newaxis], 
                             sqmeanWM[:,np.newaxis], sqmeanCSF[:,np.newaxis], 
                             sqdtWM[:,np.newaxis], sqdtCSF[:,np.newaxis]),axis=1)    
    elif flavor[0] == 'GM':
        meanGM = np.mean(np.float32(volData[maskGM_,:]),axis=0)
        meanGM = meanGM - np.mean(meanGM)
        meanGM = meanGM/max(meanGM)
        X = meanGM[:,np.newaxis]
    elif flavor[0] == 'WM':
        meanWM = np.mean(np.float32(volData[maskWM_,:]),axis=0)
        meanWM = meanWM - np.mean(meanWM)
        meanWM = meanWM/max(meanWM)
        X = meanWM[:,np.newaxis]   
    else:
        print('Warning! Wrong tissue regression flavor. Nothing was done')
    
    if flavor[-1] == 'GM':
        if config.isCifti:
            niiImgGM = niiImg[0]
        else:
            niiImgGM = volData[maskGM_,:]
        niiImgGM = regress(niiImgGM, nTRs, TR, X, config.preWhitening)
        if config.isCifti:
            niiImg[0] = niiImgGM
        else:
            volData[maskGM_,:] = niiImgGM
            niiImg[0] = volData    
        return niiImg[0], niiImg[1]

    elif flavor[-1] == 'wholebrain':
        return X
    else:
        print('Warning! Wrong tissue regression flavor. Nothing was done')
        
def Detrending(niiImg, flavor, masks, imgInfo):
    maskAll, maskWM_, maskCSF_, maskGM_ = masks
    nRows, nCols, nSlices, nTRs, affine, TR, header = imgInfo
						 
    
    if config.isCifti:
        volData = niiImg[1]
    else:
        volData = niiImg[0]

    if flavor[2] == 'WMCSF':
        niiImgWMCSF = volData[np.logical_or(maskWM_,maskCSF_),:]
        if flavor[0] == 'legendre':
            y = legendre_poly(flavor[1],nTRs)                
        elif flavor[0] == 'poly':       
            x = np.arange(nTRs)            
            y = np.ones((flavor[1],len(x)))
            for i in range(flavor[1]):
                y[i,:] = (x - (np.max(x)/2)) **(i+1)
                y[i,:] = y[i,:] - np.mean(y[i,:])
                y[i,:] = y[i,:]/np.max(y[i,:]) 
        else:
            print('Warning! Wrong detrend flavor. Nothing was done')
            return niiImg[0],niiImg[1]     
        niiImgWMCSF = regress(niiImgWMCSF, nTRs, TR, y[1:flavor[1],:].T, config.preWhitening)
        volData[np.logical_or(maskWM_,maskCSF_),:] = niiImgWMCSF
    elif flavor[2] == 'GM':
        if config.isCifti:
            niiImgGM = niiImg[0]
        else:
            niiImgGM = volData[maskGM_,:]
        if flavor[0] == 'legendre':
            y = legendre_poly(flavor[1], nTRs)
        elif flavor[0] == 'poly':       
            x = np.arange(nTRs)
            y = np.ones((flavor[1],len(x)))
            for i in range(flavor[1]):
                y[i,:] = (x - (np.max(x)/2)) **(i+1)
                y[i,:] = y[i,:] - np.mean(y[i,:])
                y[i,:] = y[i,:]/np.max(y[i,:])
        else:
            print('Warning! Wrong detrend flavor. Nothing was done')
            return niiImg[0],niiImg[1]     
        niiImgGM = regress(niiImgGM, nTRs, TR, y[1:flavor[1],:].T, config.preWhitening)
        if config.isCifti:
            niiImg[0] = niiImgGM
        else:
            volData[maskGM_,:] = niiImgGM
    elif flavor[2] == 'wholebrain':
        if flavor[0] == 'legendre':
            y = legendre_poly(flavor[1], nTRs)
        elif flavor[0] == 'poly':       
            x = np.arange(nTRs)
            y = np.ones((flavor[1],len(x)))
            for i in range(flavor[1]):
                y[i,:] = (x - (np.max(x)/2)) **(i+1)
                y[i,:] = y[i,:] - np.mean(y[i,:])
                y[i,:] = y[i,:]/np.max(y[i,:])        
        else:
            print('Warning! Wrong detrend flavor. Nothing was done')
        return y.T    
    else:
        print('Warning! Wrong detrend mask. Nothing was done' )

    if config.isCifti:
        niiImg[1] = volData
    else:
        niiImg[0] = volData            
    return niiImg[0],niiImg[1]     
   
   
def TemporalFiltering(niiImg, flavor, masks, imgInfo):
    maskAll, maskWM_, maskCSF_, maskGM_ = masks
    nRows, nCols, nSlices, nTRs, affine, TR, header = imgInfo

    if config.doScrubbing and flavor[0] in ['Butter','Gaussian']:
        censored = np.loadtxt(op.join(outpath(), 'Censored_TimePoints_{}.txt'.format(config.pipelineName)), dtype=np.dtype(np.int32))
        censored = np.atleast_1d(censored)
        if len(censored)<nTRs and len(censored) > 0:
            data = interpolate(niiImg[0],censored,TR,nTRs,method=config.interpolation)     
            if niiImg[1] is not None:
                data2 = interpolate(niiImg[1],censored,TR,nTRs,method=config.interpolation)
        else:
            data = niiImg[0]
            if niiImg[1] is not None:
                data2 = niiImg[1]
    else:
        data = niiImg[0]
        if niiImg[1] is not None:
            data2 = niiImg[1]

    if flavor[0] == 'Butter':
        R = 0.1
        Nr = 50
        x = data.T
        N = x.shape[0]
        NR = min(round(N*R),Nr)
        x1 = np.zeros((NR, x.shape[1]))
        x2 = np.zeros((NR, x.shape[1]))
        for i in range(x.shape[1]):
            x1[:,i] = 2*x[0,i] - np.flipud(x[1:NR+1,i])
            x2[:,i] = 2*x[-1,i] - np.flipud(x[-NR-1:-1,i])
        x = np.vstack([x1,x,x2])
        x = clean(x, detrend=False, standardize=False, 
                              t_r=TR, high_pass=flavor[1], low_pass=flavor[2])
        niiImg[0] = x[NR:-NR,:].T
        if niiImg[1] is not None:
            x = data2.T
            x1 = np.zeros((NR, x.shape[1]))
            x2 = np.zeros((NR, x.shape[1]))
            for i in range(x.shape[2]):
                x1[:,i] = 2*x[0,i] - np.flipud(x[1:NR+1,i])
                x2[:,i] = 2*x[-1,i] - np.flipud(x[-NR-1:-1,i])
            x = np.vstack([x1,x,x2])
            x = clean(data2.T, detrend=False, standardize=False, 
               t_r=TR, high_pass=flavor[1], low_pass=flavor[2])
            niiImg[1] = x[NR:-NR,:].T
    elif flavor[0] == 'Gaussian':
        w = signal.gaussian(11,std=flavor[1])
        niiImg[0] = signal.lfilter(w,1,data)
        if niiImg[1] is not None:
            niiImg[1] = signal.lfilter(w,1,data2)
    elif flavor[0] == 'DCT':
        K = dctmtx(nTRs)
        if len(flavor)>2:
            HPC = 1/flavor[1]
            LPC = 1/flavor[2]
            nHP = int(np.fix(2*(nTRs*TR)/HPC + 1))
            nLP = int(np.fix(2*(nTRs*TR)/LPC + 1))
            K = K[:,np.concatenate((range(2,nHP),range(int(nLP)-1,nTRs)))]
        else:
            HPC = 1/flavor[1]
            nHP = int(np.fix(2*(nTRs*TR)/HPC + 1))
            K = K[:,range(2,nHP)]
        return K
    else:
        print('Warning! Wrong temporal filtering flavor. Nothing was done')
        return niiImg[0],niiImg[1]

    config.filtering = flavor
    return niiImg[0],niiImg[1]    

def GlobalSignalRegression(niiImg, flavor, masks, imgInfo):
    meanAll = np.mean(niiImg[0],axis=0)
    meanAll = meanAll - np.mean(meanAll)
    meanAll = meanAll/max(meanAll)
    if flavor[0] == 'GS':
        return meanAll[:,np.newaxis]
    elif flavor[0] == 'GS+dt':
        dtGS=np.zeros(meanAll.shape,dtype=np.float32)
        dtGS[1:] = np.diff(meanAll, n=1)
        X  = np.concatenate((meanAll[:,np.newaxis], dtGS[:,np.newaxis]), axis=1)
        return X
    elif flavor[0] == 'GS+dt+sq':
        dtGS = np.zeros(meanAll.shape,dtype=np.float32)
        dtGS[1:] = np.diff(meanAll, n=1)
        sqGS = meanAll ** 2
        sqdtGS = dtGS ** 2
        X  = np.concatenate((meanAll[:,np.newaxis], dtGS[:,np.newaxis], sqGS[:,np.newaxis], sqdtGS[:,np.newaxis]), axis=1)
        return X
    else:
        print('Warning! Wrong normalization flavor. Using defalut regressor: GS')
        return meanAll[:,np.newaxis]

def VoxelNormalization(niiImg, flavor, masks, imgInfo):
    if flavor[0] == 'zscore':
        niiImg[0] = stats.zscore(niiImg[0], axis=1, ddof=1)
        if niiImg[1] is not None:
            niiImg[1] = stats.zscore(niiImg[1], axis=1, ddof=1)
    elif flavor[0] == 'pcSigCh':
        meanImg = np.mean(niiImg[0],axis=1)[:,np.newaxis]
        close0 = np.where(meanImg < 1e5*np.finfo(np.float).eps)[0]
        if close0.shape[0] > 0:
            meanImg[close0,0] = np.max(np.abs(niiImg[0][close0,:]),axis=1)
            niiImg[0][close0,:] = niiImg[0][close0,:] + meanImg[close0,:]
        niiImg[0] = 100 * (niiImg[0] - meanImg) / meanImg
        niiImg[0][np.where(np.isnan(niiImg[0]))] = 0
        if niiImg[1] is not None:
            meanImg = np.mean(niiImg[1],axis=1)[:,np.newaxis]
            close0 = np.where(meanImg < 1e5*np.finfo(np.float).eps)[0]
            if close0.shape[0] > 0:
                meanImg[close0,0] = np.max(np.abs(niiImg[1][close0,:]),axis=1)
                niiImg[1][close0,:] = niiImg[1][close0,:] + meanImg[close0,:]
            niiImg[1] = 100 * (niiImg[1] - meanImg) / meanImg
            niiImg[1][np.where(np.isnan(niiImg[1]))] = 0
    elif flavor[0] == 'demean':
        niiImg[0] = niiImg[0] - niiImg[0].mean(1)[:,np.newaxis]
        if niiImg[1] is not None:
            niiImg[1] = niiImg[1] - niiImg[1].mean(1)[:,np.newaxis]
    else:
        print('Warning! Wrong normalization flavor. Nothing was done')
    return niiImg[0],niiImg[1] 

# Struct used to associate functions to operation names
Hooks={
    'TaskRegression'         : TaskRegression,
    'MotionRegression'       : MotionRegression,
    'Scrubbing'              : Scrubbing,
    'TissueRegression'       : TissueRegression,
    'Detrending'             : Detrending,
    'TemporalFiltering'      : TemporalFiltering,  
    'GlobalSignalRegression' : GlobalSignalRegression,  
    'VoxelNormalization'     : VoxelNormalization,
    }

### End of Operations section
# ---------------------------

## 
#  @brief Compute frame displacement
#  
#  @return [np.array] frame displacement score
#  
def computeFD():
    # Frame displacement
    motionFile = op.join(buildpath(), config.movementRegressorsFile)
    dmotpars = np.abs(np.genfromtxt(motionFile)[:,6:]) #derivatives
    disp=dmotpars.copy()
    disp[:,3:]=np.pi*config.headradius*2*(disp[:,3:]/360)
    score=np.sum(disp,1)
    return score
## 
#  @brief Generate gray plot
#  
#  @param [bool] displayPlot True if plot should be displayed
#  @param [bool] overwrite True if existing files should be overwritten
#  
def stepPlot(X,operationName, displayPlot=False,overwrite=False):
    savePlotFile = op.join(outpath(),operationName+'_grayplot.png')
    sys.stdout.flush()
    if not op.isfile(savePlotFile) or overwrite:
        
        if not config.isCifti:
            # load masks
            maskAll, maskWM_, maskCSF_, maskGM_ = makeTissueMasks(False)

        fig = plt.figure(figsize=(15,8))
        ax1 = plt.subplot(111)

        # denoised volume
        if not config.isCifti:
            X = stats.zscore(X, axis=1, ddof=1)
            Xgm  = X[maskGM_,:]
            Xwm  = X[maskWM_,:]
            Xcsf = X[maskCSF_,:]
        else:
            # cifti
            tsvFile = config.fmriFile_dn.replace('.dtseries.nii','.tsv')
            if not op.isfile(tsvFile):
                cmd = 'wb_command -cifti-convert -to-text {} {}'.format(config.fmriFile_dn,tsvFile)
                call(cmd,shell=True)
            Xgm = pd.read_csv(tsvFile,sep='\t',header=None,dtype=np.float32).values
            nTRs = Xgm.shape[1]
            Xgm = stats.zscore(Xgm, axis=1, ddof=1)

        if not config.isCifti:
            im = plt.imshow(np.vstack((Xgm,Xwm,Xcsf)), aspect='auto', interpolation='none', cmap=plt.cm.gray)
        else:
            im = plt.imshow(Xgm, aspect='auto', interpolation='none', cmap=plt.cm.gray)
        im.set_clim(vmin=-3, vmax=3)
        plt.title(operationName)
        plt.ylabel('Voxels')
        if not config.isCifti:
            plt.axhline(y=np.sum(maskGM_), color='r')
            plt.axhline(y=np.sum(maskGM_)+np.sum(maskWM_), color='b')

        # prettify
        fig.subplots_adjust(right=0.9)
        fig.colorbar(im)
        # save figure
        fig.savefig(savePlotFile, bbox_inches='tight',dpi=75)


## 
#  @brief Generate gray plot
#  
#  @param [bool] displayPlot True if plot should be displayed
#  @param [bool] overwrite True if existing files should be overwritten
#  
def makeGrayPlot(displayPlot=False,overwrite=False):
    savePlotFile = config.fmriFile_dn.replace(config.ext,'_grayplot.png')
    if not op.isfile(savePlotFile) or overwrite:
        # FD
        t = time()
        score = computeFD()
        
        if not config.isCifti:
            # load masks
            maskAll, maskWM_, maskCSF_, maskGM_ = makeTissueMasks(False)

            # original volume
            X, nRows, nCols, nSlices, nTRs, affine, TR, header = load_img(config.fmriFile, maskAll)
            X = stats.zscore(X, axis=1, ddof=1)
            Xgm  = X[maskGM_,:]
            Xwm  = X[maskWM_,:]
            Xcsf = X[maskCSF_,:]
        else:
            # cifti
            tsvFile = config.fmriFile.replace('.dtseries.nii','.tsv')
            if not op.isfile(tsvFile):									  
                cmd = 'wb_command -cifti-convert -to-text {} {}'.format(config.fmriFile,tsvFile)
                call(cmd,shell=True)
            Xgm = pd.read_csv(tsvFile,sep='\t',header=None,dtype=np.float32).values
            nTRs = Xgm.shape[1]
            Xgm = stats.zscore(Xgm, axis=1, ddof=1)

        fig = plt.figure(figsize=(15,20))
        ax1 = plt.subplot(311)
        plt.plot(np.arange(nTRs), score)
        plt.title('Subject {}, run {}, denoising {}'.format(config.subject,config.fmriRun,config.pipelineName))
        plt.ylabel('FD (mm)')

        ax2 = plt.subplot(312, sharex=ax1)
        if not config.isCifti:
            im = plt.imshow(np.vstack((Xgm,Xwm,Xcsf)), aspect='auto', interpolation='none', cmap=plt.cm.gray)
        else:
            im = plt.imshow(Xgm, aspect='auto', interpolation='none', cmap=plt.cm.gray)
        im.set_clim(vmin=-3, vmax=3)
        plt.title('Before denoising')
        plt.ylabel('Voxels')
        if not config.isCifti:
            plt.axhline(y=np.sum(maskGM_), color='r')
            plt.axhline(y=np.sum(maskGM_)+np.sum(maskWM_), color='b')

        # denoised volume
        if not config.isCifti:
            X, nRows, nCols, nSlices, nTRs, affine, TR, header = load_img(config.fmriFile_dn, maskAll)
            X = stats.zscore(X, axis=1, ddof=1)
            Xgm  = X[maskGM_,:]
            Xwm  = X[maskWM_,:]
            Xcsf = X[maskCSF_,:]
        else:
            # cifti
            tsvFile = config.fmriFile_dn.replace('.dtseries.nii','.tsv')
            if not op.isfile(tsvFile):
                cmd = 'wb_command -cifti-convert -to-text {} {}'.format(config.fmriFile_dn,tsvFile)
                call(cmd,shell=True)
            Xgm = pd.read_csv(tsvFile,sep='\t',header=None,dtype=np.float32).values
            nTRs = Xgm.shape[1]
            Xgm = stats.zscore(Xgm, axis=1, ddof=1)

        ax3 = plt.subplot(313, sharex=ax1)
        if not config.isCifti:
            im = plt.imshow(np.vstack((Xgm,Xwm,Xcsf)), aspect='auto', interpolation='none', cmap=plt.cm.gray)
        else:
            im = plt.imshow(Xgm, aspect='auto', interpolation='none', cmap=plt.cm.gray)
        im.set_clim(vmin=-3, vmax=3)
        plt.title('After denoising')
        plt.ylabel('Voxels')
        if not config.isCifti:
            plt.axhline(y=np.sum(maskGM_), color='r')
            plt.axhline(y=np.sum(maskGM_)+np.sum(maskWM_), color='b')

        # prettify
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.2, 0.03, 0.4])
        fig.colorbar(im, cax=cbar_ax)
        # save figure
        fig.savefig(savePlotFile, bbox_inches='tight',dpi=75)
        print("makeGrayPlot -- done in {:0.2f}s".format(time()-t))
        sys.stdout.flush()

    else:
        image = mpimg.imread(savePlotFile)
        fig = plt.figure(figsize=(15,20))
        plt.axis("off")
        plt.imshow(image)

    if displayPlot:
        plt.show(fig)
    else:
        plt.close(fig)
## 
#  @brief Apply parcellation (output saved to file)
#  
def parcellate(overwrite=False):
    print("entering parcellate (overwrite={})".format(overwrite))
    # After preprocessing, functional connectivity is computed
    tsDir = op.join(outpath(),config.parcellationName)
    if not op.isdir(tsDir): mkdir(tsDir)
    prefix = config.session+'_' if  hasattr(config,'session')  else ''																  
    tsDir = op.join(tsDir,prefix+config.fmriRun+config.ext)
    if not op.isdir(tsDir): mkdir(tsDir)

    #####################
    # read parcels
    #####################
    if not config.isCifti:
        maskAll, maskWM_, maskCSF_, maskGM_ = makeTissueMasks(False)
        if not config.maskParcelswithAll:     
            maskAll  = np.ones(np.shape(maskAll), dtype=bool)
        allparcels, nRows, nCols, nSlices, nTRs, affine, TR, header = load_img(config.parcellationFile, maskAll)
        if config.maskParcelswithGM:
            allparcels[np.logical_not(maskGM_)] = 0;
    else:
        if not op.isfile(config.parcellationFile.replace('.dlabel.nii','.tsv')):    
            cmd = 'wb_command -cifti-convert -to-text {} {}'.format(config.parcellationFile,
                                                                   config.parcellationFile.replace('.dlabel.nii','.tsv'))
            call(cmd, shell=True)
        allparcels = np.loadtxt(config.parcellationFile.replace('.dlabel.nii','.tsv'))
    
    ####################
    # original data
    ####################
    alltsFile = op.join(tsDir,'allParcels.txt')
    if not op.isfile(alltsFile) or overwrite:
        # read original volume
        if not config.isCifti:
            data, nRows, nCols, nSlices, nTRs, affine, TR, header = load_img(config.fmriFile, maskAll)
        else:
            tsvFile = config.fmriFile.replace('.dtseries.nii','.tsv')
            if not op.isfile(tsvFile):									  
                cmd = 'wb_command -cifti-convert -to-text {} {}'.format(config.fmriFile,tsvFile)
                call(cmd, shell=True)
            data = pd.read_csv(tsvFile,sep='\t',header=None,dtype=np.float32).values
        
        for iParcel in np.arange(config.nParcels):
            tsFile = op.join(tsDir,'parcel{:03d}.txt'.format(iParcel+1))
            if not op.isfile(tsFile) or overwrite:
                np.savetxt(tsFile,np.nanmean(data[np.where(allparcels==iParcel+1)[0],:],axis=0),fmt='%.16f',delimiter='\n')

        # concatenate all ts
        print('Concatenating data')
        cmd = 'paste '+op.join(tsDir,'parcel???.txt')+' > '+alltsFile
        call(cmd, shell=True)

    ####################
    # denoised data
    ####################
    rstring      = get_rcode(config.fmriFile_dn)
    alltsFile    = op.join(tsDir,'allParcels_{}.txt'.format(rstring))
    if (not op.isfile(alltsFile)) or overwrite:
        # read denoised volume
        if not config.isCifti:
            data, nRows, nCols, nSlices, nTRs, affine, TR, header = load_img(config.fmriFile_dn, maskAll)
        else:
            if not op.isfile(config.fmriFile_dn.replace('.dtseries.nii','.tsv')):
                cmd = 'wb_command -cifti-convert -to-text {} {}'.format(config.fmriFile_dn,
                                                                           config.fmriFile_dn.replace('.dtseries.nii','.tsv'))
                call(cmd, shell=True)
            data = pd.read_csv(config.fmriFile_dn.replace('.dtseries.nii','.tsv'),sep='\t',header=None,dtype=np.float32).values
                   
        for iParcel in np.arange(config.nParcels):
            tsFile = op.join(tsDir,'parcel{:03d}_{}.txt'.format(iParcel+1,rstring))
            if not op.isfile(tsFile) or overwrite:
                np.savetxt(tsFile,np.nanmean(data[np.where(allparcels==iParcel+1)[0],:],axis=0),fmt='%.16f',delimiter='\n')
            # save all voxels in mask, with header indicating parcel number
            if config.save_voxelwise:
                tsFileAll = op.join(tsDir,'parcel{:03d}_{}_all.txt'.format(iParcel+1,rstring))
                if not op.isfile(tsFileAll) or overwrite:
                    np.savetxt(tsFileAll,np.transpose(data[np.where(allparcels==iParcel+1)[0],:]),fmt='%.16f',delimiter=',',newline='\n')
        
        # concatenate all ts
        print('Concatenating data')
        cmd = 'paste '+op.join(tsDir,'parcel???_{}.txt'.format(rstring))+' > '+alltsFile
        call(cmd, shell=True)


## 
#  @brief Get FC matrices for list of subjects
#  
#  @param [array-like] subjectList list of subject IDs
#  @param [array-like] runs list of runs 
#  @param [array-like] sessions list of sessions (optional)
#  @param [str] parcellation parcellation name - needed if FCDir is None      
#  @param [list] operations pipeline operations - needed if FCDir is None
#  @param [str] outputDir path to preprocessed data folder (optional, default is outpath())
#  @param [bool] isCifti True if preprocessed data is in cifti format
#  @param [str] fcMatFile full path to output file (default ./fcMats.mat)
#  @param [str] kind type of FC, one of {"correlation", "partial correlation", "tangent", "covariance", "precision"}
#  @param [bool] overwrite True if existing files should be overwritten
#  @param [str] FCDir path to folder containing precomputed timeseries x parcels per subject - if None they are retrieved from each subject's folder
#  @param [bool] mergeSessions True if time series from different sessions should be merged before computing FC, otherwise FC from each session are averaged
#  @param [bool] mergeRuns True if time series from different runs should be merged before computing FC, otherwise FC from each run are averaged (if mergeSessions is True mergeRuns is ignored and everything is concatenated)
#  @param [CovarianceEstimator] cov_estimator is None, default sklearn.covariance.LedoitWolf estimator is used
#  
def getAllFC(subjectList,runs,sessions=None,parcellation=None,operations=None,outputDir=None,isCifti=False,fcMatFile='fcMats.mat',
             kind='correlation',overwrite=True,FCDir=None,mergeSessions=True,mergeRuns=False,cov_estimator=None):
    if (not op.isfile(fcMatFile)) or overwrite:
        if cov_estimator is None:
            cov_estimator=LedoitWolf(assume_centered=False, block_size=1000, store_precision=False)
        measure = connectome.ConnectivityMeasure(
        cov_estimator=cov_estimator,
        kind = kind,
        vectorize=True, 
        discard_diagonal=True)
        if isCifti:
            ext = '.dtseries.nii'
        else:
            ext = '.nii.gz'

        FC_sub = list()
        ts_all = list()
        for subject in subjectList:
            config.subject = str(subject)
            ts_sub = list()
            if sessions:
                ts_ses = list()
                for config.session in sessions:
                    ts_run = list()
                    for config.fmriRun in runs:
                        if FCDir is None: # retrieve data from each subject's folder
                            # retrieve the name of the denoised fMRI file
                            if hasattr(config,'fmriFileTemplate'):
                                inputFile = op.join(buildpath(), config.fmriFileTemplate.replace('#fMRIrun#', config.fmriRun).replace('#fMRIsession#', config.session))
                            else:
                                prefix = config.session+'_'
                                if isCifti:
                                    inputFile = op.join(buildpath(), prefix+config.fmriRun+'_Atlas'+config.suffix+ext)
                                else:
                                    inputFile = op.join(buildpath(), prefix+config.fmriRun+ext)
                            outputPath = outpath() if outputDir is None else outputDir
                            preproFile = retrieve_preprocessed(inputFile, operations, outputPath, isCifti)
                            if preproFile:
                                # retrieve time courses of parcels
                                prefix = config.session+'_'
                                tsDir     = op.join(outputPath,parcellation,prefix+config.fmriRun+ext)
                                rstring   = get_rcode(preproFile)
                                tsFile    = op.join(tsDir,'allParcels_{}.txt'.format(rstring))
                                ts        = np.genfromtxt(tsFile,delimiter="\t")
                            else:
                                continue
                        else: # retrieve data from FCDir
                            tsFile = op.join(FCDir,config.subject+'_'+config.session+'_'+config.fmriRun+'_ts.txt')
                            if op.isfile(tsFile):
                                ts = np.genfromtxt(tsFile,delimiter=",")
                            else:
                                continue
                        # standardize
                        ts -= ts.mean(axis=0)
                        ts /= ts.std(axis=0)
                        ts_sub.append(ts) 
                        ts_run.append(ts)
                    if len(ts_run)>0:
                        ts_ses.append(np.concatenate(ts_run,axis=0))  
                if not mergeSessions and mergeRuns:
                    FC_sub.append(measure.fit_transform(ts_ses)) 
            else:
                mergeSessions = False
                for config.fmriRun in runs:
                    if FCDir is None: # retrieve data from each subject's folder
                        # retrieve the name of the denoised fMRI file
                        if hasattr(config,'fmriFileTemplate'):
                            inputFile = op.join(buildpath(), config.fmriFileTemplate.replace('#fMRIrun#', config.fmriRun).replace('#fMRIsession#', config.session))
                        else:
                            if isCifti:
                                inputFile = op.join(buildpath(), config.fmriRun+'_Atlas'+config.suffix+ext)
                            else:
                                inputFile = op.join(buildpath(), config.fmriRun+ext)
                        outputPath = outpath() if (outputDir is None) else outputDir
                        preproFile = retrieve_preprocessed(inputFile, operations, outputPath, isCifti)
                        if preproFile:
                            # retrieve time courses of parcels
                            tsDir     = op.join(outpath(),config.parcellationName,config.fmriRun+ext)
                            rstring   = get_rcode(preproFile)
                            tsFile    = op.join(tsDir,'allParcels_{}.txt'.format(rstring))
                            ts        = np.genfromtxt(tsFile,delimiter="\t")
                        else:
                            continue
                    else:
                        tsFile = op.join(FCDir,config.subject+'_'+config.fmriRun+'_ts.txt')
                        if op.isfile(tsFile):
                            ts = np.genfromtxt(tsFile,delimiter=",")
                        else:
                            continue
                    # standardize
                    ts -= ts.mean(axis=0)
                    ts /= ts.std(axis=0)
                    ts_sub.append(ts)
            if len(ts_sub)>0:
                ts_all.append(np.concatenate(ts_sub, axis=0))
            if not mergeSessions and not mergeRuns:
               FC_sub.append(measure.fit_transform(ts_sub))

        # compute connectivity matrix
        if mergeSessions or (sessions is None and mergeRuns): 
            fcMats = measure.fit_transform(ts_all)
        else: 
            fcMats = np.vstack([np.mean(el,axis=0) for el in FC_sub])
        # SAVE fcMats
        results      = {}
        results['fcMats'] = fcMats
        results['subjects'] = subjectList
        results['runs'] = np.array(runs)
        if sessions: results['sessions'] = np.array(sessions)
        results['kind'] = kind
        sio.savemat(fcMatFile, results)
        return results
    else:
        results = sio.loadmat(fcMatFile)
        return results
## 
#  @brief Compute functional connectivity matrix (output saved to file)
#  
#  @param [bool] overwrite True if existing files should be overwritten
#  
def computeFC(overwrite=False):
    prefix = config.session+'_' if  hasattr(config,'session')  else ''
    FCDir = config.FCDir if  hasattr(config,'FCDir')  else ''
    if FCDir and not op.isdir(FCDir): makedirs(FCDir, exist_ok=True)
    tsDir = op.join(outpath(),config.parcellationName,prefix+config.fmriRun+config.ext)
    cov_estimator = LedoitWolf(assume_centered=False, block_size=1000, store_precision=False)
    measure = connectome.ConnectivityMeasure(cov_estimator=cov_estimator,kind = config.fcType,vectorize=False)
    ###################
    # original
    ###################
    alltsFile = op.join(tsDir,'allParcels.txt')
    if not op.isfile(alltsFile) or overwrite:
        parcellate(overwrite)
    fcFile     = alltsFile.replace('.txt','_Pearson.txt')
    if not op.isfile(fcFile) or overwrite:
        ts = np.loadtxt(alltsFile)
        # correlation
        corrMat = np.squeeze(measure.fit_transform([ts]))
									 
        # save as .txt
        np.savetxt(fcFile,corrMat,fmt='%.6f',delimiter=',')
    ###################
    # denoised
    ###################
    rstring = get_rcode(config.fmriFile_dn)
    alltsFile = op.join(tsDir,'allParcels_{}.txt'.format(rstring))
    if not op.isfile(alltsFile) or overwrite:
        parcellate(overwrite)
    fcFile    = alltsFile.replace('.txt','_Pearson.txt')
    if not op.isfile(fcFile) or overwrite:
        ts = np.loadtxt(alltsFile)
        # censor time points that need censoring
        if config.doScrubbing:
            censored = np.loadtxt(op.join(outpath(), 'Censored_TimePoints_{}.txt'.format(config.pipelineName)), dtype=np.dtype(np.int32))
            censored = np.atleast_1d(censored)
            tokeep = np.setdiff1d(np.arange(ts.shape[0]),censored)
            ts = ts[tokeep,:]
        # correlation
        corrMat = np.squeeze(measure.fit_transform([ts]))
									 
        # save as .txt
        np.savetxt(fcFile,corrMat,fmt='%.6f',delimiter=',')
        if FCDir:
            np.savetxt(op.join(FCDir,config.subject+'_'+prefix+config.fmriRun+'_ts.txt'),ts,fmt='%.6f',delimiter=',')
	

## 
#  @brief Compute functional connectivity matrices before and after preprocessing and generate FC plot
#  
#  @param  [bool] displayPlot True if plot should be displayed
#  @param  [bool] overwrite True if existing files should be overwritten
#  @return [tuple] functional connectivity matrix before and after denoising
#     
def plotFC(displayPlot=False,overwrite=False):
    print("entering plotFC (overwrite={})".format(overwrite))
    savePlotFile=config.fmriFile_dn.replace(config.ext,'_'+config.parcellationName+'_fcMat.png')

    if not op.isfile(savePlotFile) or overwrite:
        computeFC(overwrite)
    prefix = config.session+'_' if  hasattr(config,'session')  else ''																  
    tsDir      = op.join(outpath(),config.parcellationName,prefix+config.fmriRun+config.ext)
    fcFile     = op.join(tsDir,'allParcels_Pearson.txt')
    fcMat      = np.genfromtxt(fcFile,delimiter=",")
    rstring    = get_rcode(config.fmriFile_dn)
    fcFile_dn  = op.join(tsDir,'allParcels_{}_Pearson.txt'.format(rstring))
    fcMat_dn   = np.genfromtxt(fcFile_dn,delimiter=",")
    
    # if not op.isfile(savePlotFile) or overwrite:
    fig = plt.figure(figsize=(20,7.5))
    ####################
    # original, Pearson
    ####################
    ax1 = plt.subplot(121)
    im = plt.imshow(fcMat, aspect='auto', interpolation='none', cmap=plt.cm.jet)
    im.set_clim(vmin=-.5, vmax=.5)
    plt.axis('equal')  
    plt.axis('off')  
    plt.title('Subject {}, run {}, Pearson'.format(config.subject,config.fmriRun))
    plt.xlabel('parcel #')
    plt.ylabel('parcel #')
    ####################
    # denoised, Pearson
    ####################
    ax2 = plt.subplot(122)#, sharey=ax1)
    im = plt.imshow(fcMat_dn, aspect='auto', interpolation='none', cmap=plt.cm.jet)
    im.set_clim(vmin=-.5, vmax=.5)
    plt.axis('equal')  
    plt.axis('off')  
    plt.title('{}'.format(config.pipelineName))
    plt.xlabel('parcel #')
    plt.ylabel('parcel #')
    ####################
    # prettify
    ####################
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.25, 0.03, 0.5])
    fig.colorbar(im, cax=cbar_ax)
    # save figure
    fig.savefig(savePlotFile, bbox_inches='tight')
    # else:
    #     image = mpimg.imread(savePlotFile)
    #     fig = plt.figure(figsize=(20,7.5))
    #     plt.axis("off")
    #     plt.imshow(image)
    
    if displayPlot:
        plt.show(fig)
    else:
        plt.close(fig)

    return fcMat,fcMat_dn

## 
#  @brief Generate confound vector according to code word
#  
#  @param  [pandas.DataFrame] df data dictionary
#  @param  [int] confound code word identifying list of counfounds (one of 'gender', 'age', 'handedness', 'age^2', 'gender*age', 'gender*age^2', 'brainsize', 'motion', 'recon')
#  @param  [int] session session identifier (one of 'REST1', 'REST2', 'REST12')
#  @return [array_like] vector of confounds
#  
def defConVec(df,confound,session):
    if confound == 'gender':
        conVec = df['Gender']
    elif confound == 'age':
        conVec = df['Age_in_Yrs']
    elif confound == 'handedness':
        conVec = df['Handedness']
    elif confound == 'age^2':
        conVec = np.square(df['Age_in_Yrs'])
    elif confound == 'gender*age':
        conVec = np.multiply(df['Gender'],df['Age_in_Yrs'])
    elif confound == 'gender*age^2':
        conVec = np.multiply(df['Gender'],np.square(df['Age_in_Yrs']))
    elif confound == 'brainsize':
        conVec = df['FS_BrainSeg_Vol']
    elif confound == 'motion':
        if session in ['REST1','REST2','EMOTION','GAMBLING','LANGUAGE','MOTOR','RELATIONAL','SOCIAL','WM']:
            conVec = df['FDsum_'+session]
        elif session == 'REST12':
            conVec = 1./2.*(df['FDsum_REST1'] + df['FDsum_REST2'])
        elif session == 'TASK':
            conVec = 1./7.*(df['FDsum_EMOTION'] + df['FDsum_GAMBLING'] + df['FDsum_LANGUAGE'] + df['FDsum_MOTOR'] + df['FDsum_RELATIONAL'] + df['FDsum_SOCIAL'] + df['FDsum_WM'])
        elif session == 'TASK+REST':
            conVec = 1./9.*(df['FDsum_EMOTION'] + df['FDsum_GAMBLING'] + df['FDsum_LANGUAGE'] + df['FDsum_MOTOR'] + df['FDsum_RELATIONAL'] + df['FDsum_SOCIAL'] + df['FDsum_WM'] + df['FDsum_REST1'] + df['FDsum_REST2'])
    elif confound == 'recon':
        conVec = df['fMRI_3T_ReconVrs']
    elif confound == 'PMAT24_A_CR':
        conVec = df['PMAT24_A_CR']
    return conVec

## 
#  @brief Compute prediction of subject measure (output saved to file)
#  
#  @param     [str] fcMatFile filename of functional connectivity matrices file
#  @param     [str] dataFile filename of subject measures data frame
#  @param     [int] test_index index of the subject whose score will be predicted
#  @param     [float] filterThr threshold for p-value to select edges correlated with subject measure
#  @param     [str] keepEdgeFile name of file containing a mask to select a subset of edge (optional)
#  @iPerm     [array_like] vector of permutation indices, if [0] no permutation test is run
#  @SM        [str] subject measure name
#  @session   [str] session identifier (one of 'REST1', 'REST2, 'REST12')
#  @model     [str] regression model type ('Finn', 'elnet' or 'krr')
#  @outDir    [str] output directory
#  @confounds [list] confound vector
#  
#  @details The edges of FC matrix are used to build a linear regression model to predict the subject measure. Finn model uses a first degree 
#  polynomial to fit the subject measure as a function of the sum of edge weights. Elastic net builds a multivariate model using the edges of 
#  the FC matrix as features. In both models, only edges correlated with the subject measure on training data are selected, and counfounds are
#  regressed out from the subject measure. If requested, a permutation test is also run.
#  
def runPredictionJD(fcMatFile, dataFile, test_index, filterThr=0.01, keepEdgeFile='', iPerm=[0], SM='PMAT24_A_CR', session='REST12', decon='decon', fctype='Pearson', model='Finn',outDir='',confounds=['gender','age','age^2','gender*age','gender*age^2','brainsize','motion','recon']):
    data         = sio.loadmat(fcMatFile)
    edges        = data['fcMats_'+fctype]

    if len(keepEdgeFile)>0:
        keepEdges = np.loadtxt(keepEdgeFile).astype(bool)
        edges     = edges[:,keepEdges]

    n_subs       = edges.shape[0]
    n_edges      = edges.shape[1]

    df           = pd.read_csv(dataFile)
    score        = np.array(np.ravel(df[SM]))

    train_index = np.setdiff1d(np.arange(n_subs),test_index)
    
    # REMOVE CONFOUNDS
    conMat = None
    if len(confounds)>0:
        for confound in confounds:
            conVec = defConVec(df,confound,session)
            # add to conMat
            if conMat is None:
                conMat = np.array(np.ravel(conVec))
            else:
                print(confound,conMat.shape,conVec.shape)
                conMat = np.vstack((conMat,conVec))
        # if only one confound, transform to matrix
        if len(confounds)==1:
            conMat = conMat[:,np.newaxis]
        else:
            conMat = conMat.T

        corrBef = []
        for i in range(len(confounds)):
            corrBef.append(stats.pearsonr(conMat[:,i].T,score)[0])
        print('maximum corr before decon: ',max(corrBef))

        regr        = linear_model.LinearRegression()
        regr.fit(conMat[train_index,:], score[train_index])
        fittedvalues = regr.predict(conMat)
        score        = score - np.ravel(fittedvalues)
        print(score.shape)

        corrAft = []
        for i in range(len(confounds)):
            corrAft.append(stats.pearsonr(conMat[:,i].T,score)[0])
        print('maximum corr after decon: ',max(corrAft))

    # keep a copy of score
    score_ = np.copy(score)

    for thisPerm in iPerm: 
        print("=  perm{:04d}  ==========".format(thisPerm))
        print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
        print("=========================")
        
        score = np.copy(score_)
        # REORDER SCORE!
        if thisPerm > 0:
            # read permutation indices
            permInds = np.loadtxt(op.join(outDir,'..','permInds.txt'),dtype=np.int16)
            score    = score[permInds[thisPerm-1,:]]

        if not op.isdir(op.join(outDir,'{:04d}'.format(thisPerm))):
            mkdir(op.join(outDir,'{:04d}'.format(thisPerm)))


        outFile = op.join(outDir,'{:04d}'.format(thisPerm),'{}.mat'.format(
            '_'.join(['%s' % test_sub for test_sub in df['Subject'][test_index]])))
        print(outFile)

        if op.isfile(outFile) and not config.overwrite:
            continue
     
        # compute univariate correlation between each edge and the Subject Measure
        pears  = [stats.pearsonr(np.squeeze(edges[train_index,j]),score[train_index]) for j in range(0,n_edges)]
        pearsR = [pears[j][0] for j in range(0,n_edges)]
        
        idx_filtered     = np.array([idx for idx in range(0,n_edges) if pears[idx][1]<filterThr])
        idx_filtered_pos = np.array([idx for idx in range(0,n_edges) if pears[idx][1]<filterThr and pears[idx][0]>0])
        idx_filtered_neg = np.array([idx for idx in range(0,n_edges) if pears[idx][1]<filterThr and pears[idx][0]<0])
            
        if model=='Finn':
            print(model)
            lr  = linear_model.LinearRegression()
            # select edges (positively and negatively) correlated with score with threshold filterThr
            filtered_pos = edges[np.ix_(train_index,idx_filtered_pos)]
            filtered_neg = edges[np.ix_(train_index,idx_filtered_neg)]
            # compute network statistic for each subject in training
            strength_pos = filtered_pos.sum(axis=1)
            strength_neg = filtered_neg.sum(axis=1)
            strength_posneg = strength_pos - strength_neg
            # compute network statistic for test subjects
            str_pos_test = edges[np.ix_(test_index,idx_filtered_pos)].sum(axis=1)
            str_neg_test = edges[np.ix_(test_index,idx_filtered_neg)].sum(axis=1)
            str_posneg_test = str_pos_test - str_neg_test
            # regression
            lr_posneg           = lr.fit(np.stack((strength_pos,strength_neg),axis=1),score[train_index])
            predictions_posneg  = lr_posneg.predict(np.stack((str_pos_test,str_neg_test),axis=1))
            lr_pos_neg          = lr.fit(strength_posneg.reshape(-1,1),score[train_index])
            predictions_pos_neg = lr_posneg.predict(str_posneg_test.reshape(-1,1))
            lr_pos              = lr.fit(strength_pos.reshape(-1,1),score[train_index])
            predictions_pos     = lr_pos.predict(str_pos_test.reshape(-1,1))
            lr_neg              = lr.fit(strength_neg.reshape(-1,1),score[train_index])
            predictions_neg     = lr_neg.predict(str_neg_test.reshape(-1,1))
            results = {'score':score[test_index],'pred_posneg':predictions_posneg,'pred_pos_neg':predictions_pos_neg, 'pred_pos':predictions_pos, 'pred_neg':predictions_neg,'idx_filtered_pos':idx_filtered_pos, 'idx_filtered_neg':idx_filtered_neg}
            print('saving results')
            sio.savemat(outFile,results)
        
        elif model=='elnet':
            print(model)
            X_train, X_test, y_train, y_test = edges[np.ix_(train_index,idx_filtered)], edges[np.ix_(test_index,idx_filtered)], score[train_index], score[test_index]
            rbX            = RobustScaler()
            X_train        = rbX.fit_transform(X_train)
            # equalize distribution of score for cv folds
            n_bins_cv      = 4
            hist_cv, bin_limits_cv = np.histogram(y_train, n_bins_cv)
            bins_cv        = np.digitize(y_train, bin_limits_cv[:-1])
            # set up nested cross validation 
            nCV_gridsearch = 3
            cv             = cross_validation.StratifiedKFold(n_splits=nCV_gridsearch)       
            elnet          = ElasticNetCV(l1_ratio=[.01],n_alphas=50,cv=cv.split(X_train, bins_cv),max_iter=1500,tol=0.001)
            # TRAIN
            start_time     = time()
            elnet.fit(X_train,y_train)
            elapsed_time   = time() - start_time
            print("Trained ELNET in {0:02d}h:{1:02d}min:{2:02d}s".format(int(elapsed_time//3600),int((elapsed_time%3600)//60),int(elapsed_time%60)))
            # PREDICT
            X_test         = rbX.transform(X_test)
            if len(X_test.shape) == 1:
                X_test     = X_test.reshape(1, -1)
            prediction     = elnet.predict(X_test)
            results        = {'score':y_test,'pred':prediction, 'coef':elnet.coef_, 'alpha':elnet.alpha_, 'l1_ratio':elnet.l1_ratio_, 'idx_filtered':idx_filtered}
            print('saving results')
            sio.savemat(outFile,results)        
        elif model=='krr':
            X_train, X_test, y_train, y_test = edges[np.ix_(train_index,idx_filtered)], edges[np.ix_(test_index,idx_filtered)], score[train_index], score[test_index]
            rbX            = RobustScaler()
            X_train        = rbX.fit_transform(X_train)
            # equalize distribution of score for cv folds
            n_bins_cv      = 4
            hist_cv, bin_limits_cv = np.histogram(y_train, n_bins_cv)
            bins_cv        = np.digitize(y_train, bin_limits_cv[:-1])
            # Fit KernelRidge with parameter selection based on stratified k-fold cross validation
            nCV_gridsearch = 3
            param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3]}

            kr = GridSearchCV(KernelRidge(kernel='precomputed'), cv=cross_validation.StratifiedKFold(n_splits=nCV_gridsearch).split(X_train, bins_cv), param_grid=param_grid)
            
            # TRAIN
            start_time     = time()
            kr.fit(correlationKernel(X_train,X_train),y_train)
            elapsed_time   = time() - start_time
            print("Trained KRR in {0:02d}h:{1:02d}min:{2:02d}s".format(int(elapsed_time//3600),int((elapsed_time%3600)//60),int(elapsed_time%60)))   
            # PREDICT
            X_test         = rbX.transform(X_test)
            if len(X_test.shape) == 1:
                X_test     = X_test.reshape(1, -1)
            prediction     = kr.predict(correlationKernel(X_test,X_train))
            results        = {
                            'test_index':test_index,
                            'score':y_test,
                            'pred':prediction,
                            'idx_filtered':idx_filtered
                            }
            sio.savemat(outFile,results)
        sys.stdout.flush()
    
## 
#  @brief Run predictions for all subjects in parallel using sge qsub
#  
#  @param     [str] fcMatFile filename of functional connectivity matrices file
#  @param     [str] dataFile filename of subject measures data frame
#  @SM        [str] subject measure name
#  @iPerm     [array_like] vector of permutation indices, if [0] no permutation test is run
#  @confounds [list] confound vector
#  @param     [bool] launchSubproc if False, prediction are computed sequentially instead of being submitted to a queue for parallel computation 
#  @session   [str] session identifier (one of 'REST1', 'REST2, 'REST12')
#  @model     [str] regression model type (either 'Finn' or 'elnet')
#  @outDir    [str] output directory
#  @param     [float] filterThr threshold for p-value to select edges correlated with subject measure
#  @param     [str] keepEdgeFile name of file containing a mask to select a subset of edge (optional)
#  
#  @details Predictions for all subjects are run using a leave-family-out cross validation scheme.
#  
def runPredictionParJD(fcMatFile, dataFile, SM='PMAT24_A_CR', iPerm=[0], confounds=['gender','age','age^2','gender*age','gender*age^2','brainsize','motion','recon'], launchSubproc=False, session='REST12',decon='decon',fctype='Pearson',model='Finn', outDir = '', filterThr=0.01, keepEdgeFile=''):
    data = sio.loadmat(fcMatFile)
    df   = pd.read_csv(dataFile)
    # leave one family out
    iCV = 0
    config.scriptlist = []
    for el in np.unique(df['Family_ID']):
        test_index    = list(df.ix[df['Family_ID']==el].index)
        test_subjects = list(df.ix[df['Family_ID']==el]['Subject'])
        jPerm = list()
        for thisPerm in iPerm:
            outFile = op.join(outDir,'{:04d}'.format(thisPerm),'{}.mat'.format(
                '_'.join(['%s' % test_sub for test_sub in test_subjects])))
            if not op.isfile(outFile) or config.overwrite:
                jPerm.append(thisPerm)
        if len(jPerm)==0:
            iCV = iCV + 1 
            continue
        jobDir = op.join(outDir, 'jobs')
        if not op.isdir(jobDir): 
            mkdir(jobDir)
        jobName = 'f{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(el,config.pipelineName,config.parcellationName,SM, model,config.release,session,decon,fctype)
        # make a script
        thispythonfn  = '<< END\nimport sys\nsys.path.insert(0,"{}")\n'.format(config.sourceDir)
        thispythonfn += 'from HCP_helpers import *\n'
        thispythonfn += 'logFid                  = open("{}","a+")\n'.format(op.join(jobDir,jobName+'.log'))
        thispythonfn += 'sys.stdout              = logFid\n'
        thispythonfn += 'sys.stderr              = logFid\n'
        # print date and time stamp
        thispythonfn += 'print("=========================")\n'
        thispythonfn += 'print(strftime("%Y-%m-%d %H:%M:%S", localtime()))\n'
        thispythonfn += 'print("=========================")\n'
        thispythonfn += 'config.DATADIR          = "{}"\n'.format(config.DATADIR)
        thispythonfn += 'config.outDir          = "{}"\n'.format(config.outDir)
        thispythonfn += 'config.pipelineName     = "{}"\n'.format(config.pipelineName)
        thispythonfn += 'config.parcellationName = "{}"\n'.format(config.parcellationName)
        #        thispythonfn += 'config.outScore         = "{}"\n'.format(config.outScore)
        thispythonfn += 'config.release          = "{}"\n'.format(config.release)
        thispythonfn += 'config.behavFile        = "{}"\n'.format(config.behavFile)
        thispythonfn += 'config.overwrite        = {}\n'.format(config.overwrite)
        thispythonfn += 'print("=========================")\n'
        thispythonfn += 'print("runPredictionJD(\'{}\',\'{}\'))"\n'.format(fcMatFile, dataFile)
        thispythonfn += 'print("=========================")\n'
        thispythonfn += 'print("=========================")\n'
        str1 =  '['+','.join(['%s' % test_ind for test_ind in test_index])+']'
        str2 =  '['+','.join(['"%s"' % el for el in confounds])+']'
        str3 =  '['+','.join(['%s' % el for el in jPerm])+']'
        thispythonfn += 'runPredictionJD("{}","{}",'.format(fcMatFile, dataFile)
        thispythonfn += ' {}, filterThr={}, keepEdgeFile="{}", SM="{}"'.format(str1,filterThr, keepEdgeFile, SM)
        thispythonfn += ', session="{}", decon="{}", fctype="{}", model="{}", outDir="{}", confounds={},iPerm={})\n'.format(session, decon, fctype, model, outDir, str2,str3)
        thispythonfn += 'logFid.close()\n'
        thispythonfn += 'END'
        thisScript=op.join(jobDir,jobName+'.sh')
        while True:
            if op.isfile(thisScript) and (not config.overwrite):
                thisScript=thisScript.replace('.sh','+.sh') # use fsl feat convention
            else:
                break
        with open(thisScript,'w') as fidw:
            fidw.write('#!/bin/bash\n')
            fidw.write('python {}\n'.format(thispythonfn))
        cmd='chmod 774 '+thisScript
        call(cmd,shell=True)
        #this is a "hack" to make sure the .sh script exists before it is called... 
        while not op.isfile(thisScript):
            sleep(.05)
        if config.queue:
            # call to fnSubmitToCluster
            config.scriptlist.append(thisScript)
            sys.stdout.flush()
        elif launchSubproc:
            sys.stdout.flush()
            process = Popen(thisScript,shell=True)
            config.joblist.append(process)
        else:
            runPredictionJD(fcMatFile,dataFile,test_index,filterThr=filterThr,keepEdgeFile=keepEdgeFile,SM=SM, session=session, decon=decon, fctype=fctype, model=model, outDir=outDir, confounds=confounds,iPerm=jPerm)
        iCV = iCV +1

    if len(config.scriptlist)>0:
        # launch array job
        JobID = fnSubmitJobArrayFromJobList()
        config.joblist.append(JobID.split(b'.')[0])
    
## 
#  @brief Run preprocessing pipeline (output saved to file)
#  
def runPipeline():

    Flavors = config.Flavors
    steps   = config.steps
    sortedOperations = config.sortedOperations
    
    timeStart = localtime()
    print('Step 0 : Building WM, CSF and GM masks...')
    masks = makeTissueMasks(overwrite=False)
    maskAll, maskWM_, maskCSF_, maskGM_ = masks    

    if config.isCifti:
        # volume
        prefix = config.session+'_' if  hasattr(config,'session')  else ''
        volFile = op.join(buildpath(), prefix+config.fmriRun+'.nii.gz')
        print('Loading [volume] data in memory... {}'.format(volFile))
        volData, nRows, nCols, nSlices, nTRs, affine, TR, header = load_img(volFile, maskAll) 
        # cifti
        print('Loading [cifti] data in memory... {}'.format(config.fmriFile.replace('.dtseries.nii','.tsv')))
        if not op.isfile(config.fmriFile.replace('.dtseries.nii','.tsv')):
            cmd = 'wb_command -cifti-convert -to-text {} {}'.format(config.fmriFile,config.fmriFile.replace('.dtseries.nii','.tsv'))
            call(cmd,shell=True)
        data = pd.read_csv(config.fmriFile.replace('.dtseries.nii','.tsv'),sep='\t',header=None,dtype=np.float32).values
    else:
        volFile = config.fmriFile
        print('Loading [volume] data in memory... {}'.format(config.fmriFile))
        data, nRows, nCols, nSlices, nTRs, affine, TR, header = load_img(volFile, maskAll) 
        volData = None
       
    nsteps = len(steps)
    for i in range(1,nsteps+1):
        step = steps[i]
        print('Step '+str(i)+' '+str(step))
        if len(step) == 1:
            # Atomic operations
            if ('Regression' in step[0]) or ('TemporalFiltering' in step[0] and 'DCT' in Flavors[i][0]) or ('wholebrain' in Flavors[i][0]):
                if ((step[0]=='TissueRegression' and 'GM' in Flavors[i][0] and 'wholebrain' not in Flavors[i][0]) or
                   (step[0]=='MotionRegression' and 'nonaggr' in Flavors[i][0])): 
                    #regression constrained to GM
                    data, volData = Hooks[step[0]]([data,volData], Flavors[i][0], masks, [nRows, nCols, nSlices, nTRs, affine, TR, header])
                else:
                    r0 = Hooks[step[0]]([data,volData], Flavors[i][0], masks, [nRows, nCols, nSlices, nTRs, affine, TR, header])
                    data = regress(data, nTRs, TR, r0, config.preWhitening)
            else:
                data, volData = Hooks[step[0]]([data,volData], Flavors[i][0], masks, [nRows, nCols, nSlices, nTRs, affine, TR, header])
        else:
            # When multiple regression steps have the same order, all the regressors are combined
            # and a single regression is performed (other operations are executed in order)
            r = np.empty((nTRs, 0))
            for j in range(len(step)):
                opr = step[j]
                if ('Regression' in opr) or ('TemporalFiltering' in opr and 'DCT' in Flavors[i][j]) or ('wholebrain' in Flavors[i][j]):
                    if ((opr=='TissueRegression' and 'GM' in Flavors[i][j] and 'wholebrain' not in Flavors[i][j]) or
                       (opr=='MotionRegression' and 'nonaggr' in Flavors[i][j])): 
                        #regression constrained to GM
                        data, volData = Hooks[opr]([data,volData], Flavors[i][j], masks, [nRows, nCols, nSlices, nTRs, affine, TR, header])
                    else:    
                        r0 = Hooks[opr]([data,volData], Flavors[i][j], masks, [nRows, nCols, nSlices, nTRs, affine, TR, header])
                        r = np.append(r, r0, axis=1)
                else:
                    data, volData = Hooks[opr]([data,volData], Flavors[i][j], masks, [nRows, nCols, nSlices, nTRs, affine, TR, header])
            if r.shape[1] > 0:
                data = regress(data, nTRs, TR, r, config.preWhitening)    
        data[np.isnan(data)] = 0
        if config.isCifti:
            volData[np.isnan(volData)] = 0
        if config.plotSteps:
            stepPlot(data, str(step))


    print('Done! Copy the resulting file...')
    rstring = ''.join(random.SystemRandom().choice(string.ascii_lowercase +string.ascii_uppercase + string.digits) for _ in range(8))
    outDir  = outpath()
    prefix = config.session+'_' if  hasattr(config,'session')  else ''															  
    outFile = prefix+config.fmriRun+'_prepro_'+rstring
    if config.isCifti:
        # write to text file
        np.savetxt(op.join(outDir,outFile+'.tsv'),data, delimiter='\t', fmt='%.6f')
        # need to convert back to cifti
        cmd = 'wb_command -cifti-convert -from-text {} {} {}'.format(op.join(outDir,outFile+'.tsv'),
                                                                     config.fmriFile,
                                                                     op.join(outDir,outFile+'.dtseries.nii'))
        call(cmd,shell=True)
    else:
        niiImg = np.zeros((nRows*nCols*nSlices, nTRs),dtype=np.float32)
        niiImg[maskAll,:] = data
        nib.save(nib.Nifti1Image(niiImg.reshape((nRows, nCols, nSlices, nTRs), order='F').astype('<f4'), affine, header=header),op.join(outDir,outFile+'.nii.gz'))

    timeEnd = localtime()  

    outXML = rstring+'.xml'
    conf2XML(config.fmriFile, config.DATADIR, sortedOperations, timeStart, timeEnd, op.join(outpath(),outXML))

    print('Preprocessing complete. ')
    config.fmriFile_dn = op.join(outDir,outFile+config.ext)

    return

## 
#  @brief Run preprocessing pipeline on all subjects in parallel using sge qsub
#  
#  @param [bool] launchSubproc if False, prediction are computed sequentially instead of being submitted to a queue for parallel computation
#  @param [bool] overwriteFC True if existing FC matrix files should be overwritten 
#  @param [bool] cleanup True if old files should be removed
#  
def runPipelinePar(launchSubproc=False,overwriteFC=False,cleanup=True,do_makeGrayPlot=False,do_plotFC=False):
    if config.queue: 
        priority=-100
    config.suffix = '_hp2000_clean' if config.useFIX else '' 
    if config.isCifti:
        config.ext = '.dtseries.nii'
    else:
        config.ext = '.nii.gz'

    if config.overwrite:
        overwriteFC = True

    if hasattr(config,'fmriFileTemplate'):
        config.fmriFile = op.join(buildpath(), config.fmriFileTemplate.replace('#fMRIrun#', config.fmriRun).replace('#suffix#',config.suffix))
    else:
        prefix = config.session+'_' if  hasattr(config,'session')  else ''																  
        if config.isCifti:
            config.fmriFile = op.join(buildpath(), prefix+config.fmriRun+'_Atlas'+config.suffix+'.dtseries.nii')
        else:
            config.fmriFile = op.join(buildpath(), prefix+config.fmriRun+config.suffix+'.nii.gz')
    
    if not op.isfile(config.fmriFile):
        print(config.fmriFile, 'missing')
        sys.stdout.flush()
        return False

    config.sortedOperations = sorted(config.Operations, key=operator.itemgetter(1))
    config.steps            = {}
    config.Flavors          = {}
    cstep                   = 0

    # If requested, scrubbing is performed first, before any denoising step
    scrub_idx = -1
    curr_idx = -1
    for opr in config.sortedOperations:
        curr_idx = curr_idx+1
        if opr[0] == 'Scrubbing' and opr[1] != 1 and opr[1] != 0:
            scrub_idx = opr[1]
            break
            
    if scrub_idx != -1:        
        for opr in config.sortedOperations:  
            if opr[1] != 0:
                opr[1] = opr[1]+1

        config.sortedOperations[curr_idx][1] = 1
        config.sortedOperations = sorted(config.Operations, key=operator.itemgetter(1))

    prev_step = 0	
    for opr in config.sortedOperations:
        if opr[1]==0:
            continue
        else:
            if opr[1]!=prev_step:
                cstep=cstep+1
                config.steps[cstep] = [opr[0]]
                config.Flavors[cstep] = [opr[2]]
            else:
                config.steps[cstep].append(opr[0])
                config.Flavors[cstep].append(opr[2])
            prev_step = opr[1]                
    precomputed = checkXML(config.fmriFile,config.steps,config.Flavors,outpath(),config.isCifti) 

    if precomputed and not config.overwrite:
								  
								  
        config.fmriFile_dn = precomputed
																				 
								  
																										  
							
					   
							
        if (not do_plotFC) and (not do_makeGrayPlot):
            return True
    else:
        if precomputed:
            try:
                remove(precomputed)
                remove(precomputed.replace(config.ext,'_grayplot.png'))
                remove(precomputed.replace(config.ext,'_'+config.parcellationName+'_fcMat.png'))
            except OSError:
                pass
            try:
                remove(op.join(outpath(),get_rcode(precomputed)+'.xml'))
            except OSError:
                pass
							  
							  

    if config.queue or launchSubproc:
        jobDir = op.join(config.outDir,'jobs')
        if not op.isdir(jobDir): 
            mkdir(jobDir)
        jobName = 's{}_{}_{}_cifti{}_{}'.format(config.subject,config.fmriRun,config.pipelineName,config.isCifti,timestamp())

        # make a script
        thispythonfn  = '<< END\nimport sys\nsys.path.insert(0,"{}")\n'.format(config.sourceDir)
        thispythonfn += 'from HCP_helpers import *\n'
        thispythonfn += 'logFid                  = open("{}","a+",1)\n'.format(op.join(jobDir,jobName+'.log'))
        thispythonfn += 'sys.stdout              = logFid\n'
        thispythonfn += 'sys.stderr              = logFid\n'
        # print date and time stamp
        thispythonfn += 'print("=========================")\n'
        thispythonfn += 'print(strftime("%Y-%m-%d %H:%M:%S", localtime()))\n'
        thispythonfn += 'print("=========================")\n'
        thispythonfn += 'config.subject          = "{}"\n'.format(config.subject)
        thispythonfn += 'config.DATADIR          = "{}"\n'.format(config.DATADIR)
        thispythonfn += 'config.outDir          = "{}"\n'.format(config.outDir)
        thispythonfn += 'config.fmriRun          = "{}"\n'.format(config.fmriRun)
        thispythonfn += 'config.useFIX           = {}\n'.format(config.useFIX)
        thispythonfn += 'config.useNative        = {}\n'.format(config.useNative)
        thispythonfn += 'config.pipelineName     = "{}"\n'.format(config.pipelineName)
        thispythonfn += 'config.overwrite        = {}\n'.format(config.overwrite)
        thispythonfn += 'overwriteFC             = {}\n'.format(overwriteFC)
        thispythonfn += 'config.queue            = {}\n'.format(config.queue)
        thispythonfn += 'config.preWhitening     = {}\n'.format(config.preWhitening)
        thispythonfn += 'config.isCifti          = {}\n'.format(config.isCifti)
        thispythonfn += 'config.Operations       = {}\n'.format(config.Operations)
        thispythonfn += 'config.suffix           = "{}"\n'.format(config.suffix)
        thispythonfn += 'config.ext              = "{}"\n'.format(config.ext)
        thispythonfn += 'config.fmriFile         = "{}"\n'.format(config.fmriFile)
        thispythonfn += 'config.Flavors          = {}\n'.format(config.Flavors)
        thispythonfn += 'config.steps            = {}\n'.format(config.steps)
        thispythonfn += 'config.sortedOperations = {}\n'.format(config.sortedOperations)
        thispythonfn += 'config.parcellationName = "{}"\n'.format(config.parcellationName)
        thispythonfn += 'config.parcellationFile = "{}"\n'.format(config.parcellationFile)
        thispythonfn += 'config.nParcels         = {}\n'.format(config.nParcels)
        if hasattr(config, 'melodicFolder'): 
            thispythonfn += 'config.melodicFolder    = "{}"\n'.format(config.melodicFolder.replace('#fMRIrun#', config.fmriRun))
        if hasattr(config, 'session'): 
            thispythonfn += 'config.session    = "{}"\n'.format(config.session)
        if hasattr(config, 'FCDir'): 
            thispythonfn += 'config.FCDir    = "{}"\n'.format(config.FCDir)
        thispythonfn += 'config.movementRegressorsFile      = "{}"\n'.format(config.movementRegressorsFile)
        thispythonfn += 'config.movementRelativeRMSFile         = "{}"\n'.format(config.movementRelativeRMSFile)
        if precomputed and not config.overwrite:
            thispythonfn += 'config.fmriFile_dn = "{}"\n'.format(precomputed)
        else:
            thispythonfn += 'runPipeline()\n'
        if do_makeGrayPlot:
            thispythonfn += 'makeGrayPlot(overwrite=config.overwrite)\n'
        if do_plotFC:
            thispythonfn += 'plotFC(overwrite=overwriteFC)\n'
        if cleanup:
            if config.useMemMap:
                thispythonfn += 'try:\n    remove(config.fmriFile.replace(".gz",""))\nexcept OSError:\n    pass\n'
                thispythonfn += 'try:\n    remove(config.fmriFile_dn.replace(".gz",""))\nexcept OSError:\n    pass\n'
            if config.isCifti:
                thispythonfn += 'for f in glob.glob(config.fmriFile_dn.replace(".dtseries.nii","*.tsv")): os.remove(f)\n'
                thispythonfn += 'for f in glob.glob(config.fmriFile.replace(".dtseries.nii","*.tsv")): os.remove(f)\n'
        thispythonfn += 'logFid.close()\n'
        thispythonfn += 'END'

        # prepare a script
        thisScript=op.join(jobDir,jobName+'.sh')
            	
        with open(thisScript,'w') as fidw:
            fidw.write('#!/bin/bash\n')
            fidw.write('echo ${FSLSUBALREADYRUN}\n')
            fidw.write('python {}\n'.format(thispythonfn))
        cmd='chmod 774 '+thisScript
        call(cmd,shell=True)

        #this is a "hack" to make sure the .sh script exists before it is called... 
        while not op.isfile(thisScript):
            sleep(.05)
    
        if config.queue:
            # call to fnSubmitToCluster
            # JobID = fnSubmitToCluster(thisScript, jobDir, jobName, '-p {} {}'.format(priority,config.sgeopts))
            # config.joblist.append(JobID)
            config.scriptlist.append(thisScript)
            sys.stdout.flush()
        elif launchSubproc:
            sys.stdout.flush()
            process = Popen(thisScript,shell=True)
            config.joblist.append(process)
            print('submitted {}'.format(jobName))
    
    else:
    
        if precomputed and not config.overwrite:
            config.fmriFile_dn = precomputed
        else:
            if hasattr(config, 'melodicFolder'): 
                config.melodicFolder = config.melodicFolder.replace('#fMRIrun#', config.fmriRun)
            runPipeline()
            if hasattr(config, 'melodicFolder'): 
                config.melodicFolder = config.melodicFolder.replace(config.fmriRun,'#fMRIrun#')

        if do_makeGrayPlot:
            makeGrayPlot(overwrite=config.overwrite)

        if do_plotFC:
            plotFC(overwrite=overwriteFC)

        if cleanup:
            if config.useMemMap:
                try: 
                    remove(config.fmriFile.replace(".gz",""))
                except OSError:
                    pass
                try:
                    remove(config.fmriFile_dn.replace(".gz",""))
                except OSError:
                    pass
            if config.isCifti:
                for f in glob.glob(config.fmriFile_dn.replace(".dtseries.nii","*.tsv")):
                    try:
                        remove(f)
                    except OSError:
                        pass
                for f in glob.glob(config.fmriFile.replace(".dtseries.nii","*.tsv")):
                    try:
                        remove(f)
                    except OSError:
                        pass

    return True


def checkProgress(pause=60,verbose=False):
    if len(config.joblist) != 0:
        while True:
            nleft = len(config.joblist)
            for i in range(nleft):
                if config.queue:
                    #myCmd = "ssh csclprd3s1 ""qstat | grep ' {} '""".format(config.joblist[i])
                    myCmd = "qstat | grep ' {} '".format(config.joblist[i])
                    isEmpty = False
                    try:
                        cmdOut = check_output(myCmd, shell=True)
                    except CalledProcessError as e:
                        isEmpty = True
                    finally:
                        if isEmpty:
                            nleft = nleft-1
                else:
                    returnCode = config.joblist[i].poll()
                    if returnCode is not None:
                        nleft = nleft-1
            if nleft == 0:
                break
            else:
                if verbose:
                    print('Waiting for {} jobs to complete...'.format(nleft))
            sleep(pause)
    if verbose:
        print('All done!!')
    return True

# Compute Cronbach's Alpha
def CronbachAlpha(itemScores):
    itemVars   = itemScores.var(axis=1, ddof=1)
    totScores  = itemScores.sum(axis=0)
    nItems     = itemScores.shape[0]
    return nItems / (nItems-1.) * (1 - itemVars.sum() / totScores.var(ddof=1))

def factor_analysis(X,s=2):
    # translated from Matlab code FA.m
    # by Malec L., Skacel F., Trujillo-Ortiz A.(2007). 
    # FA: Factor analysis by principal factoring. A MATLAB file. 
    # [WWW document]. URL http://www.mathworks.com/matlabcentral/fileexchange/
    # loadFile.do?objectId=14115
    #
    # Inputs: 
    #   X is the data matrix (columns as variables, X must have more
    #     than one row and more than one column).
    #
    #   s is the number of extracted factors (if this parameter is
    #     not included, number is chosen by principal component 
    #     criterion with eigenvalues greater than or equal to one).
    # Results: 
    #   B is the matrix of factor loadings (unrotated), last column
    #     of matrix B indicates an extraction estimate of 
    #     communalities.
    #
    #   L is the varimax rotated loadings matrix.
    #
    #   var describes the variability proportions, last item is
    #     the cumulative variance proportion.
    #
    #   fac is the matrix of factors.
    #   
    #   E is the matrix of specific variances.  
    m, n = X.shape
    # normalize data
    X = preprocessing.scale(X)
    # covariance matrix
    R = np.cov(X, rowvar=False)
    # maximize variance of original variables
    a = linalg.svd(R, full_matrices=False, compute_uv=False)
    # communality estimation by coefficients of multiple correlatio
    c = np.ones((n,)) - 1 / np.diag(linalg.solve(R,np.diag(np.ones((n,)))))
    g = np.array([])
    for i in range(75):
        U, D, Vh = linalg.svd(R - np.diag(np.diag(R)) + np.diag(c), full_matrices=False, compute_uv=True)
        V = Vh.T
        D = np.diag(D)
        N = np.dot(V, np.sqrt(D[:,:s]))
        p = c
        c = np.sum(N**2,1)
        g = np.union1d(g, np.where(c>1)[0])
        if g: c[g] = 1
        if np.max(np.abs(c-p))<0.001:
            break
    print('Factorial number of iterations:', i+1)
    # evaluation of factor loadings and communalities estimation
    B = np.hstack((N,c[:,np.newaxis]))
    # normalization of factor loadings
    h = np.sqrt(c)
    N = N / h[:,np.newaxis]
    L = N
    z = 0
    # iteration cycle maximizing variance of individual columns
    for l in range(35):
        A, S, M = linalg.svd(np.dot(N.T, n * L**3 - np.dot(L, np.diag(np.sum(L**2, axis=0)))), full_matrices=False, compute_uv=True)
        L = np.dot(np.dot(N,A), M)
        b = z
        z = np.sum(S)
        if np.abs(z -b) < 0.00001:
            break
    print('Rotational number of iterations:',l+1)
    # unnormalization of factor loadings
    L = L * h[:,np.newaxis]
    # factors computation by regression and variance proportions
    t = sum(L**2)/n
    var = np.hstack((t, sum(t)))
    fac = np.dot(np.dot(X,linalg.solve(R,np.diag(np.ones((n,))))), L)
    # evaluation of given factor model variance specific matrix
    r = np.diag(R) - np.sum(L**2,axis=1)
    E = R - np.dot(L, L.T) - np.diag(r)

    return B,L,var,fac,E
    
def partialcorr_via_linreg(X):
    # standardize
    X -= X.mean(axis=0)
    X /= X.std(axis=0) 
    p = X.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx    = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(X[:, idx], X[:, j])[0]
            beta_j = linalg.lstsq(X[:, idx], X[:, i])[0]
            res_j  = X[:, j] - X[:, idx].dot(beta_i)
            res_i  = X[:, i] - X[:, idx].dot(beta_j)
            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
    return P_corr

def partialcorr_via_inverse_L1reg(X):
    # standardize
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    model = GraphLassoCV(verbose=True, assume_centered = True) #alphas=np.linspace(.1,1.,19).tolist(), 
    model.fit(X)
    return -cov2corr( model.precision_ ),model 

def partialcorr_via_inverse(X):
    # standardize
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    # correlation
    emp_corr = np.dot(X.T, X) / X.shape[0]
    return -cov2corr(linalg.inv(emp_corr)), emp_corr

def cov2corr( A ):
    """
    covariance matrix to correlation matrix.
    """
    d = np.sqrt(A.diagonal())
    A = ((A.T/d).T)/d
    return A

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total: 
        print()

def vcorrcoef(X,y):
    # correlate columns of X with vector y
    # demean X
    X -= X.mean(axis=0)
    X /= np.sqrt(np.sum(np.square(X),axis=0))
    # demean y
    y -= y.mean()
    y /= np.sqrt(np.sum(np.square(y)))
    # compute correlation between columns of X and y
    r = np.sum(X*y[:,np.newaxis],axis=0)
    return r
