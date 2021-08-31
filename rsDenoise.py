import sys, argparse
from fmriprep_helpers import *



### Set parameters #################################################

# config is a global variable used by several functions



# Where does the data live?

config.DATADIR = '/storage/gablab001/data/abide/fsseg/Caltech_derivatives/'

config.sourceDir = os.getcwd() # or replace with path to source code



# Processing options

config.preprocessing = 'fmriprep' 

config.interpolation = 'linear' # 'linear' or 'astropy' 'power'



# Other options

config.queue = False

config.sgeopts = '-l mem_free=25G -pe openmp 6 -q long.q'

config.overwrite = False



# interpolate over non-contiguous voxels

config.n_contiguous = 1 # 1 does not interpolate over timepoint e.g 5 



# Define fMRI runs

fmriRuns = ['task-rest_run-1']

#####################################################################





def main():

    parser = create_parser()

    args = parser.parse_args()

    config.DATADIR = args.datadir

    config.outDir = args.output

    config.isCifti = args.cifti

    config.isGifti = args.gifti

    config.space = args.space

    config.smoothing = args.smoothing

    config.parcellationName = args.parcellationName

    config.nParcels = args.nParcels

    vFC = args.vFC

    subjects = np.loadtxt(args.input,dtype=str,ndmin=1)

    print(subjects)

    config.pipelineName            = args.pipeline 

    config.Operations              = config.operationDict[config.pipelineName]

    if args.seedFolder is not None: # Compute seed FC

        if config.isCifti or config.isGifti:

            iSurf = 0

            for config.surface in args.surface:

                config.parcellationFile = args.parcellationFile[iSurf]

                for config.subject in subjects:

                    sessions = [fpath for fpath in os.listdir(op.join(config.DATADIR,'fmriprep',config.subject)) if fpath.startswith('ses-')]

                    if len(sessions) > 0:

                        for config.session in sessions:

                            for config.fmriRun in fmriRuns:

                                print('Processing:',config.subject, config.session, config.fmriRun)

                                seeds = [op.join(args.seedFolder,config.space,fpath) for fpath in os.listdir(op.join(args.seedFolder,config.space))]

                                for seedFile in seeds:

                                    print('seed:',seedFile)

                                    if args.FCdir is None:

                                        config.FCDir = op.join(config.outDir,config.pipelineName+'_{}_{}_seedFC'.format(config.space,op.splitext(op.basename(seedFile))[0]))

                                    else:

                                        config.FCDir = args.FCdir

                                    runPipelinePar(do_makeGrayPlot=True,do_computeFC=True,seed=seedFile,vFC=vFC)

                    else:

                        if hasattr(config,'session'): delattr(config,'session')

                        for config.fmriRun in fmriRuns:

                            print('Processing:',config.subject, config.fmriRun)

                            seeds = [op.join(args.seedFolder,config.space,fpath) for fpath in os.listdir(op.join(args.seedFolder,config.space))]

                            for seedFile in seeds:

                                print('seed:',seedFile)

                                if args.FCdir is None:

                                    config.FCDir = op.join(config.outDir,config.pipelineName+'_{}_{}_seedFC'.format(config.space,op.splitext(op.basename(seedFile))[0])) 

                                else:

                                    config.FCDir = args.FCdir                           

                                runPipelinePar(do_makeGrayPlot=True,do_computeFC=True,seed=seedFile,vFC=vFC)

                iSurf = iSurf + 1

        else:

            config.parcellationFile = args.parcellationFile[0]

            for config.subject in subjects:

                print(config.subject)

                sessions = [fpath for fpath in os.listdir(op.join(config.DATADIR,'fmriprep',config.subject)) if fpath.startswith('ses-')]

                if len(sessions) > 0:

                    for config.session in sessions:

                        for config.fmriRun in fmriRuns:

                            print('Processing:',config.subject, config.session, config.fmriRun)

                            seeds = [op.join(args.seedFolder,config.space,fpath) for fpath in os.listdir(op.join(args.seedFolder,config.space))]

                            for seedFile in seeds:

                                print('seed:',seedFile)

                                if args.FCdir is None:

                                    config.FCDir = op.join(config.outDir,config.pipelineName+'_{}_{}_seedFC'.format(config.space,op.splitext(op.basename(seedFile))[0]))

                                else:

                                    config.FCDir = args.FCdir

                                runPipelinePar(do_makeGrayPlot=True,do_computeFC=True,seed=seedFile,vFC=vFC)

                else:

                    if hasattr(config,'session'): delattr(config,'session')

                    for config.fmriRun in fmriRuns:

                        print('Processing:',config.subject, config.fmriRun)

                        seeds = [op.join(args.seedFolder,config.space,fpath) for fpath in os.listdir(op.join(args.seedFolder,config.space))]

                        for seedFile in seeds:

                            print('seed:',seedFile)

                            if args.FCdir is None:

                                config.FCDir = op.join(config.outDir,config.pipelineName+'_{}_{}_seedFC'.format(config.space,op.splitext(op.basename(seedFile))[0]))   

                            else:

                                config.FCDir = args.FCdir                           

                            runPipelinePar(do_makeGrayPlot=True,do_computeFC=True,seed=seedFile,vFC=vFC)

    else: # compute either parcel-to-parcel or voxel/vertex-wise whole brain FC

        if config.isCifti or config.isGifti:

            if args.FCdir is None:

                config.FCDir = op.join(config.outDir,config.pipelineName+'_surf_FC')

            else:

                config.FCDir = args.FCdir 

            iSurf = 0

            for config.surface in args.surface: 

                config.parcellationFile = args.parcellationFile[iSurf]

                for config.subject in subjects:

                    sessions = [fpath for fpath in os.listdir(op.join(config.DATADIR,'fmriprep',config.subject)) if fpath.startswith('ses-')]

                    if len(sessions) > 0:

                        for config.session in sessions:

                            for config.fmriRun in fmriRuns:

                                print('Processing:',config.subject, config.session, config.fmriRun)

                                runPipelinePar(do_makeGrayPlot=True,do_computeFC=True,seed=None,vFC=vFC)

                    else:

                        if hasattr(config,'session'): delattr(config,'session')

                        for config.fmriRun in fmriRuns:

                            print('Processing:',config.subject, config.fmriRun)

                            runPipelinePar(do_makeGrayPlot=True,do_computeFC=True,seed=None,vFC=vFC)

                iSurf = iSurf + 1

        else:

            config.parcellationFile = args.parcellationFile[0]

            if args.FCdir is None:

                config.FCDir = op.join(config.outDir,config.pipelineName+'_vol_FC')

            else:

                config.FCDir = args.FCdir 

            for config.subject in subjects:

                sessions = [fpath for fpath in os.listdir(op.join(config.DATADIR,'fmriprep',config.subject)) if fpath.startswith('ses-')]

                if len(sessions) > 0:

                    for config.session in sessions:

                        for config.fmriRun in fmriRuns:

                            print('Processing:',config.subject, config.session, config.fmriRun)

                            runPipelinePar(do_makeGrayPlot=True,do_computeFC=True,seed=None,vFC=vFC)

                else:

                    if hasattr(config,'session'): delattr(config,'session')

                    for config.fmriRun in fmriRuns:

                        print('Processing:',config.subject, config.fmriRun)

                        runPipelinePar(do_makeGrayPlot=True,do_computeFC=True,seed=None,vFC=vFC)        





def create_parser():

    parser = argparse.ArgumentParser(description='Launch rsDenoise pipeline.')

    parser.add_argument('-space', '--space', metavar='SPACE', type=str,

                            default='MNI152NLin6Asym_res-2', help="""Space for volumetric data or volumetric seed.""")

    parser.add_argument('-surf', '--surface', metavar='SURFACE', type=str, nargs='+',

                        default=['fsaverage6_hemi-L','fsaverage6_hemi-R'], help="""Space for surface data. More than one space can be specified.""")

    parser.add_argument('-parcelName', '--parcellationName', metavar='PARCELLATION_NAME', type=str,

                        default=None, help="""Parcellation name, used in output file names. Only required for parcel-wise FC. 

                        To be specified together with -parcelFile and -parcelName""")

    parser.add_argument('-parcelFile', '--parcellationFile', metavar='PARCELLATION_FILE', type=str, nargs='+',

                        default=[None], help="""Path(s) to parcellation file. More than one can be specified, but they need

                        to match order and number of surfaces. Only required for parcel-wise FC. To be specified together with -parcelFile and -parcelName""")

    parser.add_argument('-n', '--nParcels', metavar='N_PARCELS', type=int,

                        default=None, help="""Number of parcels in parcellation. Only required for parcel-wise FC. To be specified together with -parcelFile and -parcelName""")

    parser.add_argument('-vFC', '--vFC', action='store_true', default=False,

                        help="""Compute vertex- or voxel-wise connectivity""")

    parser.add_argument('-seed', '--seedFolder', metavar='FILE', type=str,

                        help="""Folder where seeds are stored. Only required for seed FC.""")

    parser.add_argument('-FCdir', '--FCdir', metavar='FOLDER', type=str,

                        default=None, help="""Folder where to store FC outputs for all subjects.""")

    parser.add_argument('-gifti', '--gifti', action='store_true', default=False,

                        help="""Process gifti surface data.""") 

    parser.add_argument('-cifti', '--cifti', action='store_true', default=False,

                        help="""Process cifti volumetric data.""")    

    parser.add_argument('-smooth', '--smoothing', metavar='FWHM', type=float,

                        default=None, help="""To request smoothing, specify FWHM in mm.""")                     



    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('-data', '--datadir', metavar='FOLDER', type=str,

                               help="""Folder where subject data is be stored.""", required=True)

    requiredNamed.add_argument('-i', '--input', metavar='FILE', type=str,

                               help="""File containing list of subject IDs to process.""", required=True)

    requiredNamed.add_argument('-pipe', '--pipeline', metavar='NAME', type=str,

                               help="""Name of denoising pipeline to apply.""", required=True)

    requiredNamed.add_argument('-o', '--output', metavar='FOLDER', type=str,

                               help="""Folder where outputs will be stored.""", required=True)



    return parser

    

if __name__ == "__main__":

    main()

