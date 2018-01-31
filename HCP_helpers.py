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
    prewhitening       = False
    maskParcelswithAll = True
    save_voxelwise     = False
    # these variables are initialized here and used later in the pipeline, do not change
    filtering   = []
    doScrubbing = False


#----------------------------------
# IMPORTS
#----------------------------------
from __future__ import division
# Force matplotlib to not use any Xwindows backend.
import matplotlib
# core dump with matplotlib 2.0.0; use earlier version, e.g. 1.5.3
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os.path as op
from os import mkdir, makedirs, getcwd, remove, listdir, environ
import sys
import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy import stats, linalg,signal
import scipy.io as sio
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage.morphology import binary_closing, binary_dilation, binary_erosion, binary_opening, generate_binary_structure
import nipype.interfaces.fsl as fsl
from subprocess import call, check_output, CalledProcessError, Popen
import nibabel as nib
import sklearn.model_selection as cross_validation
from sklearn.linear_model import ElasticNetCV
from sklearn import linear_model,feature_selection,preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.covariance import MinCovDet,GraphLassoCV
from nilearn.signal import clean
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


#----------------------------------
# path following directory structure of HCP data
#----------------------------------
def buildpath():
    return op.join(config.DATADIR, config.subject,'MNINonLinear','Results',config.fmriRun)


#----------------------------------
# 3 alternate denoising pipelines
# many more can be implemented
#----------------------------------
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
        ]
    }
#----------------------------------


#----------------------------------
# HELPER FUNCTIONS
# several of these functions may not be used 
# for the specific analyses conducted 
# in intelligence.ipynb and personality.ipynb
#----------------------------------
def filter_regressors(regressors, filtering, nTRs, TR):
    if len(filtering)==0:
        print 'Error! Missing or wrong filtering flavor. Regressors were not filtered.'
    else:
        if filtering[0] == 'Butter':
            regressors = clean(regressors, detrend=False, standardize=False, 
                                  t_r=TR, high_pass=filtering[1], low_pass=filtering[2])
        elif filtering[0] == 'Gaussian':
            w = signal.gaussian(11,std=filtering[1])
            regressors = signal.lfilter(w,1,regressors, axis=0)  
    return regressors

def regress(data, nTRs, TR, regressors, preWhitening=False):
    print 'Starting regression with {} regressors...'.format(regressors.shape[1])
    if preWhitening:
        W = prewhitening(data, nTRs, TR, regressors)
        data = np.dot(data,W)
        regressors = np.dot(W,regressors)
    X  = np.concatenate((np.ones([nTRs,1]), regressors), axis=1)
    N = data.shape[0]
    start_time = time()
    fit = np.linalg.lstsq(X, data.T)[0]
    fittedvalues = np.dot(X, fit)
    resid = data - fittedvalues.T
    data = resid
    elapsed_time = time() - start_time
    print 'Regression completed in {:02d}h{:02d}min{:02d}s'.format(int(np.floor(elapsed_time/3600)),int(np.floor((elapsed_time%3600)/60)),int(np.floor(elapsed_time%60))) 
    return data

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

    return data, nRows, nCols, nSlices, nTRs, img.affine, TR

def makeTissueMasks(overwrite=False):
    fmriFile = op.join(buildpath(), config.fmriRun+config.suffix+'.nii.gz')
    WMmaskFileout = op.join(buildpath(), 'WMmask.nii')
    CSFmaskFileout = op.join(buildpath(), 'CSFmask.nii')
    GMmaskFileout = op.join(buildpath(), 'GMmask.nii')
    if not op.isfile(GMmaskFileout) or overwrite:
        # load ribbon.nii.gz and wmparc.nii.gz
        ribbonFilein = op.join(config.DATADIR, config.subject, 'MNINonLinear','ribbon.nii.gz')
        wmparcFilein = op.join(config.DATADIR, config.subject, 'MNINonLinear', 'wmparc.nii.gz')
        # make sure it is resampled to the same space as the functional run
        ribbonFileout = op.join(buildpath(), 'ribbon.nii.gz')
        wmparcFileout = op.join(buildpath(), 'wmparc.nii.gz')
        # make identity matrix to feed to flirt for resampling
        ribbonMat = op.join(buildpath(), 'ribbon_flirt_{}.mat'.format(config.pipelineName))
        wmparcMat = op.join(buildpath(), 'wmparc_flirt_{}.mat'.format(config.pipelineName))
        eyeMat = op.join(buildpath(), 'eye_{}.mat'.format(config.pipelineName))
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

def timestamp():
   now          = time()
   loctime      = localtime(now)
   milliseconds = '%03d' % int((now - int(now)) * 1000)
   return strftime('%Y%m%d%H%M%S', loctime) + milliseconds

def fnSubmitJobArrayFromJobList():
    config.tStamp = timestamp()
    # make directory
    mkdir('tmp{}'.format(config.tStamp))
    # write a temporary file with the list of scripts to execute as an array job
    with open(op.join('tmp{}'.format(config.tStamp),'scriptlist'),'w') as f:
        f.write('\n'.join(config.scriptlist))
    # write the .qsub file
    with open(op.join('tmp{}'.format(config.tStamp),'qsub'),'w') as f:
        f.write('#!/bin/sh\n')
        f.write('#$ -S /bin/bash\n')
        f.write('#$ -t 1-{}\n'.format(len(config.scriptlist)))
        f.write('#$ -cwd -V -N tmp{}\n'.format(config.tStamp))
        f.write('#$ -e {}\n'.format(op.join('tmp{}'.format(config.tStamp),'err')))
        f.write('#$ -o {}\n'.format(op.join('tmp{}'.format(config.tStamp),'out')))
        f.write('#$ {}\n'.format(config.sgeopts))
        f.write('SCRIPT=$(awk "NR==$SGE_TASK_ID" {})\n'.format(op.join('tmp{}'.format(config.tStamp),'scriptlist')))
        f.write('bash $SCRIPT\n')
    #strCommand = 'cd {};qsub {}'.format(getcwd(),op.join('tmp{}'.format(config.tStamp),'qsub'))
    strCommand = 'ssh csclprd3s1 "cd {};qsub {}"'.format(getcwd(),op.join('tmp{}'.format(config.tStamp),'qsub'))
    # write down the command to a file in the job folder
    with open(op.join('tmp{}'.format(config.tStamp),'cmd'),'w+') as f:
        f.write(strCommand+'\n')
    # execute the command
    cmdOut = check_output(strCommand, shell=True)
    config.scriptlist = []
    return cmdOut.split()[2]    
    
def fnSubmitToCluster(strScript, strJobFolder, strJobUID, resources):
    specifyqueue = ''
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

def sorted_ls(path, reverseOrder):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
    return list(sorted(os.listdir(path), key=mtime, reverse=reverseOrder))
	
def checkXML(inFile, operations, params, resDir, useMostRecent=True):
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
        else:
            print "Wrong interpolation method: nothing was done"
            break
    return data


### Operations
def MotionRegression(niiImg, flavor, masks, imgInfo):
    # assumes that data is organized as in the HCP
    motionFile = op.join(buildpath(), config.movementRegressorsFile)
    data = np.genfromtxt(motionFile)
    if flavor[0] == 'R dR':
        X = data
    elif flavor[0] == 'R dR R^2 dR^2':
        data_squared = data ** 2
        X = np.concatenate((data, data_squared), axis=1)
    elif flavor[0] == 'censoring':
        nRows, nCols, nSlices, nTRs, affine, TR = imgInfo
        X = np.empty((nTRs, 0))
    else:
        print 'Wrong flavor, using default regressors: R dR'
        X = data
        
    # if filtering has already been performed, regressors need to be filtered too
    if len(config.filtering)>0 and X.size > 0:
        nRows, nCols, nSlices, nTRs, affine, TR = imgInfo
        X = filter_regressors(X, config.filtering, nTRs, TR)  
        
    if config.doScrubbing:
        nRows, nCols, nSlices, nTRs, affine, TR = imgInfo
        toCensor = np.loadtxt(op.join(buildpath(), 'Censored_TimePoints_{}.txt'.format(config.pipelineName)), dtype=np.dtype(np.int32))
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
    nRows, nCols, nSlices, nTRs, affine, TR = imgInfo

    if flavor[0] == 'FD+DVARS':
        motionFile = op.join(buildpath(), config.movementRegressorsFile)
        dmotpars = np.abs(np.genfromtxt(motionFile)[:,6:]) #derivatives
        headradius=50 #50mm as in Powers et al. 2012
        disp=dmotpars.copy()
        disp[:,3:]=np.pi*headradius*2*(disp[:,3:]/360)
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
    elif flavor[0] == 'RMS':
        RelRMSFile = op.join(buildpath(), config.movementRelativeRMSFile)
        score = np.loadtxt(RelRMSFile)
    else:
        print 'Wrong scrubbing flavor. Nothing was done'
        return niiImg[0],niiImg[1]
    
    if flavor[0] == 'FD+DVARS':
        nRows, nCols, nSlices, nTRs, affine, TR = imgInfo
        # as in Siegel et al. 2016
        cleanFD = clean(score[:,np.newaxis], detrend=False, standardize=False, t_r=TR, low_pass=0.3)
        thr2 = flavor[2]
        censDVARS = scoreDVARS > 1.05 * np.median(scoreDVARS)
        censored = np.where(np.logical_or(np.ravel(cleanFD)>thr,censDVARS))
        np.savetxt(op.join(buildpath(), 'FD_{}.txt'.format(config.pipelineName)), cleanFD, delimiter='\n', fmt='%d')
        np.savetxt(op.join(buildpath(), 'DVARS_{}.txt'.format(config.pipelineName)), scoreDVARS, delimiter='\n', fmt='%d')
    else:
        censored = np.where(score>thr)
        np.savetxt(op.join(buildpath(), '{}_{}.txt'.format(flavor[0],config.pipelineName)), score, delimiter='\n', fmt='%d')
    
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
        elif censored[i] > 1200 - 5:
            toAppend = np.union1d(toAppend,np.arange(censored[i]+1,1200))
        elif i<len(censored) - 1:
            gap = censored[i+1] - censored[i] 
            if gap > 1 and gap <= 5:
                toAppend = np.union1d(toAppend,np.arange(censored[i]+1,censored[i+1]))
    censored = np.union1d(censored,toAppend)
    censored.sort()
    censored = censored.astype(int)
    
    np.savetxt(op.join(buildpath(), 'Censored_TimePoints_{}.txt'.format(config.pipelineName)), censored, delimiter='\n', fmt='%d')
    if len(censored)>0 and len(censored)<nTRs:
        config.doScrubbing = True
    if len(censored) == nTRs:
        print 'Warning! All points selected for censoring: scrubbing will not be performed.'

    #even though these haven't changed, they are returned for consistency with other operations
    return niiImg[0],niiImg[1]

def TissueRegression(niiImg, flavor, masks, imgInfo):
    maskAll, maskWM_, maskCSF_, maskGM_ = masks
    nRows, nCols, nSlices, nTRs, affine, TR = imgInfo
    
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
    else:
        print 'Warning! Wrong tissue regression flavor. Nothing was done'
    
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
        print 'Warning! Wrong tissue regression flavor. Nothing was done'
        
    

def Detrending(niiImg, flavor, masks, imgInfo):
    maskAll, maskWM_, maskCSF_, maskGM_ = masks
    nRows, nCols, nSlices, nTRs, affine, TR = imgInfo
    nPoly = flavor[1] + 1
    
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
            y = np.ones((nPoly,len(x)))
            for i in range(nPoly):
                y[i,:] = (x - (np.max(x)/2)) **(i+1)
                y[i,:] = y[i,:] - np.mean(y[i,:])
                y[i,:] = y[i,:]/np.max(y[i,:]) 
        else:
            print 'Warning! Wrong detrend flavor. Nothing was done'
        niiImgWMCSF = regress(niiImgWMCSF, nTRs, TR, y[1:nPoly,:].T, config.preWhitening)
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
            y = np.ones((nPoly,len(x)))
            for i in range(nPoly):
                y[i,:] = (x - (np.max(x)/2)) **(i+1)
                y[i,:] = y[i,:] - np.mean(y[i,:])
                y[i,:] = y[i,:]/np.max(y[i,:])
        niiImgGM = regress(niiImgGM, nTRs, TR, y[1:nPoly,:].T, config.preWhitening)
        if config.isCifti:
            niiImg[0] = niiImgGM
        else:
            volData[maskGM_,:] = niiImgGM
    elif flavor[2] == 'wholebrain':
        if flavor[0] == 'legendre':
            y = legendre_poly(flavor[1], nTRs)
        elif flavor[0] == 'poly':       
            x = np.arange(nTRs)
            y = np.ones((nPoly,len(x)))
            for i in range(nPoly):
                y[i,:] = (x - (np.max(x)/2)) **(i+1)
                y[i,:] = y[i,:] - np.mean(y[i,:])
                y[i,:] = y[i,:]/np.max(y[i,:])        
        else:
            print 'Warning! Wrong detrend flavor. Nothing was done'
        return y[1:nPoly,:].T    
    else:
        print 'Warning! Wrong detrend mask. Nothing was done' 

    if config.isCifti:
        niiImg[1] = volData
    else:
        niiImg[0] = volData            
    return niiImg[0],niiImg[1]     

   
def TemporalFiltering(niiImg, flavor, masks, imgInfo):
    maskAll, maskWM_, maskCSF_, maskGM_ = masks
    nRows, nCols, nSlices, nTRs, affine, TR = imgInfo

    if config.doScrubbing:
        censored = np.loadtxt(op.join(buildpath(), 'Censored_TimePoints_{}.txt'.format(config.pipelineName)), dtype=np.dtype(np.int32))
        censored = np.atleast_1d(censored)
        if len(censored)<nTRs and len(censored) > 0:
            data = interpolate(niiImg[0],censored,TR,nTRs,method='linear')     
            if niiImg[1] is not None:
                data2 = interpolate(niiImg[1],censored,TR,nTRs,method='linear')
        else:
            data = niiImg[0]
            if niiImg[1] is not None:
                data2 = niiImg[1]
    else:
        data = niiImg[0]
        if niiImg[1] is not None:
            data2 = niiImg[1]

    if flavor[0] == 'Butter':
        niiImg[0] = clean(data.T, detrend=False, standardize=False, 
                              t_r=TR, high_pass=flavor[1], low_pass=flavor[2]).T
        if niiImg[1] is not None:
            niiImg[1] = clean(data2.T, detrend=False, standardize=False, 
               t_r=TR, high_pass=flavor[1], low_pass=flavor[2]).T
            
    elif flavor[0] == 'Gaussian':
        w = signal.gaussian(11,std=flavor[1])
        niiImg[0] = signal.lfilter(w,1,data)
        if niiImg[1] is not None:
            niiImg[1] = signal.lfilter(w,1,data2)
    else:
        print 'Warning! Wrong temporal filtering flavor. Nothing was done'    
        return niiImg[0],niiImg[1]

    config.filtering = flavor
    return niiImg[0],niiImg[1]    

def GlobalSignalRegression(niiImg, flavor, masks, imgInfo):
    meanAll = np.mean(niiImg[0],axis=0)
    meanAll = meanAll - np.mean(meanAll)
    meanAll = meanAll/max(meanAll)
    if flavor[0] == 'GS':
        return meanAll[:,np.newaxis]
    elif flavor[0] == 'GS+dt+sq':
        dtGS = np.zeros(meanAll.shape,dtype=np.float32)
        dtGS[1:] = np.diff(meanAll, n=1)
        sqGS = meanAll ** 2
        sqdtGS = dtGS ** 2
        X  = np.concatenate((meanAll[:,np.newaxis], dtGS[:,np.newaxis], sqGS[:,np.newaxis], sqdtGS[:,np.newaxis]), axis=1)
        return X
    else:
        print 'Warning! Wrong normalization flavor. Using defalut regressor: GS'
        return meanAll[:,np.newaxis]

def VoxelNormalization(niiImg, flavor, masks, imgInfo):
    if flavor[0] == 'zscore':
        niiImg[0] = stats.zscore(niiImg[0], axis=1, ddof=1)
        if niiImg[1] is not None:
            niiImg[1] = stats.zscore(niiImg[1], axis=1, ddof=1)
    elif flavor[0] == 'demean':
        niiImg[0] = niiImg[0] - niiImg[0].mean(1)[:,np.newaxis]
        if niiImg[1] is not None:
            niiImg[1] = niiImg[1] - niiImg[1].mean(1)[:,np.newaxis]
    else:
        print 'Warning! Wrong normalization flavor. Nothing was done'
    return niiImg[0],niiImg[1] 


Hooks={
    'MotionRegression'       : MotionRegression,
    'Scrubbing'              : Scrubbing,
    'TissueRegression'       : TissueRegression,
    'Detrending'             : Detrending,
    'TemporalFiltering'      : TemporalFiltering,  
    'GlobalSignalRegression' : GlobalSignalRegression,  
    'VoxelNormalization'     : VoxelNormalization,
    }


def computeFD():
    # Frame displacement
    motionFile = op.join(buildpath(), config.movementRegressorsFile)
    dmotpars = np.abs(np.genfromtxt(motionFile)[:,6:]) #derivatives
    headradius=50 #50mm as in Powers et al. 2012
    disp=dmotpars.copy()
    disp[:,3:]=np.pi*headradius*2*(disp[:,3:]/360)
    score=np.sum(disp,1)
    return score

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
            X, nRows, nCols, nSlices, nTRs, affine, TR = load_img(config.fmriFile, maskAll)
            X = stats.zscore(X, axis=1, ddof=1)
            Xgm  = X[maskGM_,:]
            Xwm  = X[maskWM_,:]
            Xcsf = X[maskCSF_,:]
        else:
            # cifti
            if not op.isfile(config.fmriFile.replace('.dtseries.nii','.tsv')):
                cmd = 'wb_command -cifti-convert -to-text {} {}'.format(config.fmriFile,config.fmriFile.replace('.dtseries.nii','.tsv'))
                call(cmd,shell=True)
            Xgm = pd.read_csv(config.fmriFile.replace('.dtseries.nii','.tsv'),sep='\t',header=None,dtype=np.float32).values
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
            X, nRows, nCols, nSlices, nTRs, affine, TR = load_img(config.fmriFile_dn, maskAll)
            X = stats.zscore(X, axis=1, ddof=1)
            Xgm  = X[maskGM_,:]
            Xwm  = X[maskWM_,:]
            Xcsf = X[maskCSF_,:]
        else:
            # cifti
            if not op.isfile(config.fmriFile_dn.replace('.dtseries.nii','.tsv')):
                cmd = 'wb_command -cifti-convert -to-text {} {}'.format(config.fmriFile_dn,config.fmriFile_dn.replace('.dtseries.nii','.tsv'))
                call(cmd,shell=True)
            Xgm = pd.read_csv(config.fmriFile_dn.replace('.dtseries.nii','.tsv'),sep='\t',header=None,dtype=np.float32).values
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
        print "makeGrayPlot -- done in {:0.2f}s".format(time()-t)
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

def parcellate(overwrite=False):
    print "entering parcellate (overwrite={})".format(overwrite)
    # After preprocessing, functional connectivity is computed
    tsDir = op.join(buildpath(),config.parcellationName)
    if not op.isdir(tsDir): mkdir(tsDir)
    tsDir = op.join(tsDir,config.fmriRun+config.ext)
    if not op.isdir(tsDir): mkdir(tsDir)

    #####################
    # read parcels
    #####################
    if not config.isCifti:
        maskAll, maskWM_, maskCSF_, maskGM_ = makeTissueMasks(False)
        if not config.maskParcelswithAll:     
            maskAll  = np.ones(np.shape(maskAll), dtype=bool)
        allparcels, nRows, nCols, nSlices, nTRs, affine, TR = load_img(config.parcellationFile, maskAll)
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
            data, nRows, nCols, nSlices, nTRs, affine, TR = load_img(config.fmriFile, maskAll)
        else:
            if not op.isfile(config.fmriFile.replace('.dtseries.nii','.tsv')):
                cmd = 'wb_command -cifti-convert -to-text {} {}'.format(config.fmriFile,
                                                                           config.fmriFile.replace('.dtseries.nii','.tsv'))
                call(cmd, shell=True)
            data = pd.read_csv(config.fmriFile.replace('.dtseries.nii','.tsv'),sep='\t',header=None,dtype=np.float32).values
        
        for iParcel in np.arange(config.nParcels):
            tsFile = op.join(tsDir,'parcel{:03d}.txt'.format(iParcel+1))
            if not op.isfile(tsFile) or overwrite:
                np.savetxt(tsFile,np.nanmean(data[np.where(allparcels==iParcel+1)[0],:],axis=0),fmt='%.16f',delimiter='\n')

        # concatenate all ts
        print 'Concatenating data'
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
            data, nRows, nCols, nSlices, nTRs, affine, TR = load_img(config.fmriFile_dn, maskAll)
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
        print 'Concatenating data'
        cmd = 'paste '+op.join(tsDir,'parcel???_{}.txt'.format(rstring))+' > '+alltsFile
        call(cmd, shell=True)

def computeFC(overwrite=False):
    print "entering computeFC (overwrite={})".format(overwrite)
    tsDir = op.join(buildpath(),config.parcellationName,config.fmriRun+config.ext)
    ###################
    # original
    ###################
    alltsFile = op.join(tsDir,'allParcels.txt')
    if not op.isfile(alltsFile):
        parcellate(overwrite)
    fcFile     = alltsFile.replace('.txt','_Pearson.txt')
    if not op.isfile(fcFile) or overwrite:
        ts = np.loadtxt(alltsFile)
        # correlation
        corrMat = np.corrcoef(ts,rowvar=0)
        # np.fill_diagonal(corrMat,1)
        # save as .txt
        np.savetxt(fcFile,corrMat,fmt='%.6f',delimiter=',')
    ###################
    # denoised
    ###################
    rstring = get_rcode(config.fmriFile_dn)
    alltsFile = op.join(tsDir,'allParcels_{}.txt'.format(rstring))
    if not op.isfile(alltsFile):
        parcellate(overwrite)
    fcFile    = alltsFile.replace('.txt','_Pearson.txt')
    if not op.isfile(fcFile) or overwrite:
        ts = np.loadtxt(alltsFile)
        # censor time points that need censoring
        if config.doScrubbing:
            censored = np.loadtxt(op.join(buildpath(), 'Censored_TimePoints_{}.txt'.format(config.pipelineName)), dtype=np.dtype(np.int32))
            censored = np.atleast_1d(censored)
            tokeep = np.setdiff1d(np.arange(ts.shape[0]),censored)
            ts = ts[tokeep,:]
        # correlation
        corrMat = np.corrcoef(ts,rowvar=0)
        # np.fill_diagonal(corrMat,1)
        # save as .txt
        np.savetxt(fcFile,corrMat,fmt='%.6f',delimiter=',')
      
def plotFC(displayPlot=False,overwrite=False):
    print "entering plotFC (overwrite={})".format(overwrite)
    savePlotFile=config.fmriFile_dn.replace(config.ext,'_'+config.parcellationName+'_fcMat.png')

    if not op.isfile(savePlotFile) or overwrite:
        computeFC(overwrite)
    tsDir      = op.join(buildpath(),config.parcellationName,config.fmriRun+config.ext)
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

def runPredictionJD(fcMatFile, dataFile, test_index, filterThr=0.01, keepEdgeFile='', iPerm=[0], SM='PMAT24_A_CR', session='REST12', decon='decon', fctype='Pearson', model='Finn',outDir='',confounds=['gender','age','age^2','gender*age','gender*age^2','brainsize','motion','recon'],):
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
            if confound == 'gender':
                conVec = df['Gender']
            elif confound == 'age':
                conVec = df['Age_in_Yrs']
            elif confound == 'age^2':
                conVec = np.square(df['Age_in_Yrs'])
            elif confound == 'gender*age':
                conVec = np.multiply(df['Gender'],df['Age_in_Yrs'])
            elif confound == 'gender*age^2':
                conVec = np.multiply(df['Gender'],np.square(df['Age_in_Yrs']))
            elif confound == 'brainsize':
                conVec = df['FS_BrainSeg_Vol']
            elif confound == 'motion':
                if session == 'REST12':
                    conVec = .5*(df['FDsum_REST1']+df['FDsum_REST2']) 
                else:
                    conVec = df['FDsum_'+session]
            elif confound == 'recon':
                conVec = df['fMRI_3T_ReconVrs']
            elif confound == 'PMAT24_A_CR':
                conVec = df['PMAT24_A_CR']
            # add to conMat
            if conMat is None:
                conMat = np.array(np.ravel(conVec))
            else:
                print confound,conMat.shape,conVec.shape
                conMat = np.vstack((conMat,conVec))
        # if only one confound, transform to matrix
        if len(confounds)==1:
            conMat = conMat[:,np.newaxis]
        else:
            conMat = conMat.T

        corrBef = []
        for i in range(len(confounds)):
            corrBef.append(stats.pearsonr(conMat[:,i].T,score)[0])
        print 'maximum corr before decon: ',max(corrBef)

        regr        = linear_model.LinearRegression()
        regr.fit(conMat[train_index,:], score[train_index])
        fittedvalues = regr.predict(conMat)
        score        = score - np.ravel(fittedvalues)
        print score.shape

        corrAft = []
        for i in range(len(confounds)):
            corrAft.append(stats.pearsonr(conMat[:,i].T,score)[0])
        print 'maximum corr after decon: ',max(corrAft)

    # keep a copy of score
    score_ = np.copy(score)

    for thisPerm in iPerm: 
        print "=  perm{:04d}  ==========".format(thisPerm)
        print strftime("%Y-%m-%d %H:%M:%S", localtime())
        print "========================="
        
        score = np.copy(score_)
        # REORDER SCORE!
        if thisPerm > 0:
            # read permutation indices
            permInds = np.loadtxt(op.join(outDir,'..','permInds.txt'),dtype=np.int16)
            score    = score[permInds[thisPerm-1,:]]

        outFile = op.join(outDir,'{:04d}'.format(thisPerm),'{}.mat'.format(
            '_'.join(['%s' % test_sub for test_sub in df['Subject'][test_index]])))
        print outFile

        if op.isfile(outFile) and not config.overwrite:
            continue
     
        # compute univariate correlation between each edge and the Subject Measure
        pears  = [stats.pearsonr(np.squeeze(edges[train_index,j]),score[train_index]) for j in range(0,n_edges)]
        pearsR = [pears[j][0] for j in range(0,n_edges)]
        
        idx_filtered     = np.array([idx for idx in range(1,n_edges) if pears[idx][1]<filterThr])
        idx_filtered_pos = np.array([idx for idx in range(1,n_edges) if pears[idx][1]<filterThr and pears[idx][0]>0])
        idx_filtered_neg = np.array([idx for idx in range(1,n_edges) if pears[idx][1]<filterThr and pears[idx][0]<0])
            
        if model=='Finn':
            print model
            lr  = linear_model.LinearRegression()
            # select edges (positively and negatively) correlated with score with threshold filterThr
            filtered_pos = edges[np.ix_(train_index,idx_filtered_pos)]
            filtered_neg = edges[np.ix_(train_index,idx_filtered_neg)]
            # compute network statistic for each subject in training
            strength_pos = filtered_pos.sum(axis=1)
            strength_neg = filtered_neg.sum(axis=1)
            # compute network statistic for test subjects
            str_pos_test = edges[np.ix_(test_index,idx_filtered_pos)].sum(axis=1)
            str_neg_test = edges[np.ix_(test_index,idx_filtered_neg)].sum(axis=1)
            # regression
            print strength_pos.reshape(-1,1).shape
            print strength_neg.reshape(-1,1).shape
            print np.stack((strength_pos,strength_neg),axis=1).shape
            print np.stack((str_pos_test,str_neg_test),axis=1).shape
            lr_posneg           = lr.fit(np.stack((strength_pos,strength_neg),axis=1),score[train_index])
            predictions_posneg  = lr_posneg.predict(np.stack((str_pos_test,str_neg_test),axis=1))
            lr_pos              = lr.fit(strength_pos.reshape(-1,1),score[train_index])
            predictions_pos     = lr_pos.predict(str_pos_test.reshape(-1,1))
            lr_neg              = lr.fit(strength_neg.reshape(-1,1),score[train_index])
            predictions_neg     = lr_neg.predict(str_neg_test.reshape(-1,1))
            results = {'score':score[test_index],'pred_posneg':predictions_posneg, 'pred_pos':predictions_pos, 'pred_neg':predictions_neg,'idx_filtered_pos':idx_filtered_pos, 'idx_filtered_neg':idx_filtered_neg}
            print 'saving results'
            sio.savemat(outFile,results)
        
        elif model=='elnet':
            print model
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
            print "Trained ELNET in {0:02d}h:{1:02d}min:{2:02d}s".format(int(elapsed_time//3600),int((elapsed_time%3600)//60),int(elapsed_time%60))   
            # PREDICT
            X_test         = rbX.transform(X_test)
            if len(X_test.shape) == 1:
                X_test     = X_test.reshape(1, -1)
            prediction     = elnet.predict(X_test)
            results        = {'score':y_test,'pred':prediction, 'coef':elnet.coef_, 'alpha':elnet.alpha_, 'l1_ratio':elnet.l1_ratio_, 'idx_filtered':idx_filtered}
            print 'saving results'
            sio.savemat(outFile,results)        
        sys.stdout.flush()
    
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
        thispythonfn  = '<< END\nimport sys\nsys.path.insert(0,"{}")\n'.format(getcwd())
        thispythonfn += 'from helpers_clean import *\n'
        thispythonfn += 'logFid                  = open("{}","a+")\n'.format(op.join(jobDir,jobName+'.log'))
        thispythonfn += 'sys.stdout              = logFid\n'
        thispythonfn += 'sys.stderr              = logFid\n'
        # print date and time stamp
        thispythonfn += 'print "========================="\n'
        thispythonfn += 'print strftime("%Y-%m-%d %H:%M:%S", localtime())\n'
        thispythonfn += 'print "========================="\n'
        thispythonfn += 'config.DATADIR          = "{}"\n'.format(config.DATADIR)
        thispythonfn += 'config.pipelineName     = "{}"\n'.format(config.pipelineName)
        thispythonfn += 'config.parcellationName = "{}"\n'.format(config.parcellationName)
        #        thispythonfn += 'config.outScore         = "{}"\n'.format(config.outScore)
        thispythonfn += 'config.release          = "{}"\n'.format(config.release)
        thispythonfn += 'config.behavFile        = "{}"\n'.format(config.behavFile)
        thispythonfn += 'config.overwrite        = {}\n'.format(config.overwrite)
        thispythonfn += 'print "========================="\n'
        thispythonfn += 'print "runPredictionJD(\'{}\',\'{}\')"\n'.format(fcMatFile, dataFile)
        thispythonfn += 'print "========================="\n'
        thispythonfn += 'print "========================="\n'
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
        config.joblist.append(JobID.split('.')[0])

#@profile
def runPipeline():

    Flavors = config.Flavors
    steps   = config.steps
    sortedOperations = config.sortedOperations
    
    timeStart = localtime()
    print 'Step 0 : Building WM, CSF and GM masks...'
    masks = makeTissueMasks(overwrite=False)
    maskAll, maskWM_, maskCSF_, maskGM_ = masks    

    if config.isCifti:
        # volume
        volFile = op.join(buildpath(), config.fmriRun+'.nii.gz')
        print 'Loading [volume] data in memory... {}'.format(volFile)
        volData, nRows, nCols, nSlices, nTRs, affine, TR = load_img(volFile, maskAll) 
        # cifti
        print 'Loading [cifti] data in memory... {}'.format(config.fmriFile.replace('.dtseries.nii','.tsv'))
        if not op.isfile(config.fmriFile.replace('.dtseries.nii','.tsv')):
            cmd = 'wb_command -cifti-convert -to-text {} {}'.format(config.fmriFile,config.fmriFile.replace('.dtseries.nii','.tsv'))
            call(cmd,shell=True)
        data = pd.read_csv(config.fmriFile.replace('.dtseries.nii','.tsv'),sep='\t',header=None,dtype=np.float32).values
    else:
        volFile = config.fmriFile
        print 'Loading [volume] data in memory... {}'.format(config.fmriFile)
        data, nRows, nCols, nSlices, nTRs, affine, TR = load_img(volFile, maskAll) 
        volData = None
       
    nsteps = len(steps)
    for i in range(1,nsteps+1):
        step = steps[i]
        print 'Step '+str(i)+' '+str(step)
        if len(step) == 1:
            # Atomic operations
            if 'Regression' in step[0] or ('wholebrain' in Flavors[i][0]):
                if ((step[0]=='TissueRegression' and 'GM' in Flavors[i][0] and 'wholebrain' not in Flavors[i][0]) or
                   (step[0]=='MotionRegression' and 'nonaggr' in Flavors[i][0])): 
                    #regression constrained to GM
                    data, volData = Hooks[step[0]]([data,volData], Flavors[i][0], masks, [nRows, nCols, nSlices, nTRs, affine, TR])
                else:
                    r0 = Hooks[step[0]]([data,volData], Flavors[i][0], masks, [nRows, nCols, nSlices, nTRs, affine, TR])
                    data = regress(data, nTRs, TR, r0, config.preWhitening)
            else:
                data, volData = Hooks[step[0]]([data,volData], Flavors[i][0], masks, [nRows, nCols, nSlices, nTRs, affine, TR])
        else:
            # When multiple regression steps have the same order, all the regressors are combined
            # and a single regression is performed (other operations are executed in order)
            r = np.empty((nTRs, 0))
            for j in range(len(step)):
                opr = step[j]
                if 'Regression' in opr or ('wholebrain' in Flavors[i][j]):
                    if ((opr=='TissueRegression' and 'GM' in Flavors[i][j] and 'wholebrain' not in Flavors[i][j]) or
                       (opr=='MotionRegression' and 'nonaggr' in Flavors[i][j])): 
                        #regression constrained to GM
                        data, volData = Hooks[opr]([data,volData], Flavors[i][j], masks, [nRows, nCols, nSlices, nTRs, affine, TR])
                    else:    
                        r0 = Hooks[opr]([data,volData], Flavors[i][j], masks, [nRows, nCols, nSlices, nTRs, affine, TR])
                        r = np.append(r, r0, axis=1)
                else:
                    data, volData = Hooks[opr]([data,volData], Flavors[i][j], masks, [nRows, nCols, nSlices, nTRs, affine, TR])
            if r.shape[1] > 0:
                data = regress(data, nTRs, TR, r, config.preWhitening)    
        data[np.isnan(data)] = 0
        if config.isCifti:
            volData[np.isnan(volData)] = 0


    print 'Done! Copy the resulting file...'
    rstring = ''.join(random.SystemRandom().choice(string.ascii_lowercase +string.ascii_uppercase + string.digits) for _ in range(8))
    outDir  = buildpath()
    outFile = config.fmriRun+'_prepro_'+rstring
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
        nib.save(nib.Nifti1Image(niiImg.reshape((nRows, nCols, nSlices, nTRs), order='F').astype('<f4'), affine),op.join(outDir,outFile+'.nii.gz'))

    timeEnd = localtime()  

    outXML = rstring+'.xml'
    conf2XML(config.fmriFile, config.DATADIR, sortedOperations, timeStart, timeEnd, op.join(buildpath(),outXML))

    print 'Preprocessing complete. '
    config.fmriFile_dn = op.join(outDir,outFile+config.ext)

    return

def runPipelinePar(launchSubproc=False,overwriteFC=False):
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
        if config.isCifti:
            config.fmriFile = op.join(buildpath(), config.fmriRun+'_Atlas'+config.suffix+'.dtseries.nii')
        else:
            config.fmriFile = op.join(buildpath(), config.fmriRun+config.suffix+'.nii.gz')
    
    if not op.isfile(config.fmriFile):
        print config.subject, 'missing'
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
    precomputed = checkXML(config.fmriFile,config.steps,config.Flavors,buildpath()) 

    if precomputed and not config.overwrite:
        do_makeGrayPlot    = False
        do_plotFC          = False
        config.fmriFile_dn = precomputed
        if not op.isfile(config.fmriFile_dn.replace(config.ext,'_grayplot.png')):
            do_makeGrayPlot = True
        if not op.isfile(config.fmriFile_dn.replace(config.ext,'_'+config.parcellationName+'_fcMat.png')):
            do_plotFC = True
        if overwriteFC:
            do_plotFC = True
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
                remove(op.join(buildpath(),get_rcode(precomputed)+'.xml'))
            except OSError:
                pass
        do_makeGrayPlot = True
        do_plotFC       = True

    if config.queue or launchSubproc:
        jobDir = op.join(buildpath(),'jobs')
        if not op.isdir(jobDir): 
            mkdir(jobDir)
        jobName = 's{}_{}_{}_cifti{}_{}'.format(config.subject,config.fmriRun,config.pipelineName,config.isCifti,timestamp())

        # make a script
        thispythonfn  = '<< END\nimport sys\nsys.path.insert(0,"{}")\n'.format(getcwd())
        thispythonfn += 'from helpers_clean import *\n'
        thispythonfn += 'logFid                  = open("{}","a+",1)\n'.format(op.join(jobDir,jobName+'.log'))
        thispythonfn += 'sys.stdout              = logFid\n'
        thispythonfn += 'sys.stderr              = logFid\n'
        # print date and time stamp
        thispythonfn += 'print "========================="\n'
        thispythonfn += 'print strftime("%Y-%m-%d %H:%M:%S", localtime())\n'
        thispythonfn += 'print "========================="\n'
        thispythonfn += 'config.subject          = "{}"\n'.format(config.subject)
        thispythonfn += 'config.DATADIR          = "{}"\n'.format(config.DATADIR)
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
        if config.useMemMap:
            thispythonfn += 'try:\n    remove(config.fmriFile.replace(".gz",""))\nexcept OSError:\n    pass\n'
            thispythonfn += 'try:\n    remove(config.fmriFile_dn.replace(".gz",""))\nexcept OSError:\n    pass\n'
        if config.isCifti:
            thispythonfn += 'for f in glob.glob(config.fmriFile.replace("_Atlas","").replace(".dtseries.nii","*.tsv")): os.remove(f)\n'
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
            print 'submitted {}'.format(jobName)
    
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
            for f in glob.glob(config.fmriFile.replace('_Atlas','').replace(".dtseries.nii","*.tsv")):
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
                    print 'Waiting for {} jobs to complete...'.format(nleft)
            sleep(pause)
    if verbose:
        print 'All done!!' 
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
    print 'Factorial number of iterations:', i+1
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
    print 'Rotational number of iterations:',l+1
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