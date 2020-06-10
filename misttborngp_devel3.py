#MISTTBORN: the MCMC Interface for Synthesis of Transits, Tomography, Bart, and Others of a Relevant Nature
#Written by Marshall C. Johnson. 
#The horus Doppler tomographic modeling code is also written by Marshall C. Johnson.
#The batman transit modeling code is written by Laura Kreidberg.
#The emcee affine-invariant MCMC, the Bart RV and transit modeling, and the George Gaussian process regression codes are written by Daniel Foreman-Mackey

import numpy as np
import math
import emcee
#from readcol import readcol
from astropy.io import ascii
import argparse
import sys
import os
import multiprocessing
from multiprocessing import Pool
os.environ["OMP_NUM_THREADS"] = "1"

parser=argparse.ArgumentParser()
parser.add_argument("infile", type=str, help="name of the input file")
parser.add_argument("-p", "--photometry", action="store_true", help="perform photometric analysis")
parser.add_argument("-r", "--rvs", action="store_true", help="perform radial velocity analysis")
parser.add_argument("-t", "--tomography", action="store_true", help="perform Doppler tomographic analysis")
parser.add_argument("-l", "--line", action="store_true", help="fit a rotationally broadened model to a single spectral line")
parser.add_argument("-v", "--verbose", action="store_true", help="print a short message every MCMC step")
parser.add_argument("-g", "--gp", action="store_true", help="enable Gaussian process regression")
parser.add_argument("-b", "--binary", action="store_true", help="fit a binary star rather than an exoplanet: two sets of RVs, primary and secondary eclipses")
parser.add_argument("-m", "--rm", action="store_true", help="fit a RV Rossiter-McLaughlin curve")
parser.add_argument("--startnew", action="store_true", help="start a new chain regardless of whether the given output files already exist")
parser.add_argument("--plotbest", action="store_true", help="plot the best-fit model from the input chain file. Will not run a full chain.")
parser.add_argument("--plotstep", action="store_true", help="plot the current model every step. Very slow, mostly useful for debugging. This will only work for 1 thread.")
parser.add_argument("--ploterrors", action="store_true", help="include error bars on the plot.")
parser.add_argument("--plotresids", action="store_true", help="include residuals for the plots.")
parser.add_argument("--bestprob", action="store_true", help="Plot the values for the best-fit model rather than the posterior median.")
parser.add_argument("--time", action="store_true", help="calculate and print the elapsed time for each model call")
parser.add_argument("--getprob", action="store_true", help="print the contributions to lnprob from each dataset and priors")
parser.add_argument("--fullcurve", action="store_true", help="make a model lightcurve that will cover the full transit; call only with --plotbest. WARNING: doesn't totally work right, use at own risk.")
parser.add_argument("--skyline", action="store_true", help="include a sky line in some or all of the tomographic data set")
parser.add_argument("--ttvs", action="store_true", help="account for TTVs in the photometric fit")
parser.add_argument("--dilution", action="store_true", help="account for dilution due to another star in the aperture")
parser.add_argument("--pt", action="store_true", help="Use emcee's parallel tempered ensemble sampler")
args=parser.parse_args()



infile=args.infile

#if args.time:
import time as timemod
thewholestart=timemod.time()

if args.photometry or args.rm: 
    try:
        import batman
    except ImportError:
        print('batman does not appear to be installed correctly.')
        print('you can install it with "pip install batman-package"')
        print('exiting now')
        sys.exit()
        
    if args.photometry: print('burning tin to perform photometry')

if args.rvs or (args.tomography and args.binary):
    try:
        import radvel
    except ImportError:
        print('radvel does not appear to be installed correctly.')
        print('you can install it with "pip install radvel"')
        #print 'exiting now'
        print('Defaulting to the assumption of circular orbits for RVs.')
        #sys.exit()
        
    print('burning pewter to perform RV analysis')

if args.binary:
    print('burning steel to analyze a stellar binary')

if args.tomography or args.line:
    try:
        import horus
    except ImportError:
        print('horus does not appear to be installed correctly.')
        print('you can install it from [URL TBD]')
        print('exiting now')
        sys.exit()
        
    if args.tomography: print('burning bronze to perform Doppler tomography')
    if args.line: print('burning iron to analyze a single line')

if args.rm:
    print('burning copper to perform RV Rossiter-McLaughlin analysis')



def inreader(infile):
    temp=ascii.read(infile)
    names, values = temp[0][:],temp[1][:]
    outstruc = dict(list(zip(names, values)))
    outstruc['index']=names
    outstruc['invals']=values
    return outstruc

struc1=inreader(infile)

index=np.array(struc1['index'])
invals=np.array(struc1['invals'])

if args.gp:

    if not any('gppackflag' in s for s in index):
        struc1['gppackflag'], index, invals = 'george', np.append(index,'gppackflag'), np.append(invals, 'george')

        
    if struc1['gppackflag'] == 'celerite':
        try:
            import celerite as gppack
        except ImportError:
            print('celerite does not appear to be installed correctly.')
            print('you can install it with "pip install celerite"')
            print('(it also requires the Eigen package)')
            print('exiting now')
            sys.exit()

        if struc1['gpmodtypep'] == 'Haywood14QP':
            from celeriteHaywood14QP import CustomTerm as celeritekernel
        elif struc1['gpmodtypep'] == 'SHO':
            from celerite import terms
            from celerite.terms import SHOTerm as celeritekernel            
        elif (struc1['gpmodtypep'] == 'SHOMixture') | (struc1['gpmodtypep'] == 'SHOMixture_sys'):
            from celerite import terms
            from mixterm import MixtureOfSHOsTerm as celeritekernel


    else:
        try:
            import george as gppack
        except ImportError:
            print('george does not appear to be installed correctly.')
            print('you can install it with "pip install george"')
            print('(it also requires the Eigen package)')
            print('exiting now')
            sys.exit()


    print('burning electrum to perform Gaussian process regression')

#Get the parameters for the chains
nplanets=np.int64(struc1['nplanets'])
nwalkers=np.int64(struc1['nwalkers'])
nsteps=np.int64(struc1['nsteps'])
nthreads=np.int64(struc1['nthreads'])
sysname=struc1['sysname'] #not actually used later in the code, just for my own sanity

if args.plotstep or args.plotbest:
    #import matplotlib as mpl
    #mpl.use('Agg')
    import matplotlib.pyplot as pl
    if args.plotstep:
        pl.ion()
        pl.figure(1)
    nthreads=1


#else:
#    args.plotstep = False

#get the general input and output filenames
chainfile=struc1['chainfile']
probfile=struc1['probfile']
accpfile=struc1['accpfile']
#read in the perturbations, if any
if any('perturbfile' in s for s in index): 
    perturbfile=struc1['perturbfile']
    perturbstruc=inreader(perturbfile)
    perturbindex=np.array(perturbstruc['index'])
    perturbinvals=np.array(perturbstruc['invals'])
#read in the priors, if any
if any('priorfile' in s for s in index): 
    priorfile=struc1['priorfile']
    priorstruc=inreader(priorfile)
    priorindex=np.array(priorstruc['index'])
    priorinvals=np.array(priorstruc['invals'])
else:
    priorstruc={'none':'none'}

#These will be needed for any system
if args.photometry or args.tomography or args.rvs or args.rm:
    Per=np.array(invals[[i for i, s in enumerate(index) if 'Per' in s]], dtype=np.float)
    epoch=np.array(invals[[i for i, s in enumerate(index) if 'epoch' in s]], dtype=np.float)

#get the eccentricity if it exists, otherwise fix to zero and don't fit
if any('ecc' in s for s in index):
    ecc=np.array(invals[[i for i, s in enumerate(index) if 'ecc' in s]], dtype=np.float)
    omega=np.array(invals[[i for i, s in enumerate(index) if 'omega' in s]], dtype=np.float)
    fitecc=True
else:
    ecc=np.zeros(nplanets)
    omega=np.zeros(nplanets)+90.
    fitecc=False



omega*=np.pi/180.0 #degrees to radians

#check to see if eccentricity standard--default is sqrt(e) sin or cos omega
if any('ewflag' in s for s in index): 
    ewflag=struc1['ewflag']
else:
    ewflag='sesinw'
#implement the standards
if ewflag == 'sesinw':
    eccpar=np.sqrt(ecc)*np.sin(omega)
    omegapar=np.sqrt(ecc)*np.cos(omega)
    enames=['sesinw','secosw']
    for i in range(0,nplanets):
        struc1['sesinw'+str(i+1)], struc1['secosw'+str(i+1)]=eccpar[i],omegapar[i]
if ewflag == 'ecsinw':
    eccpar=ecc*np.sin(omega)
    omegapar=ecc*np.cos(omega)
    enames=['ecsinw','eccosw']
    for i in range(0,nplanets):
        struc1['ecsinw'+str(i+1)], struc1['eccosw'+str(i+1)]=eccpar[i],omegapar[i]


if ewflag == 'eomega':
    eccpar=ecc
    omegapar=omega
    enames=['ecc','omega']



#parameters needed for photometry
if args.photometry:
    photfile=np.array(invals[[i for i, s in enumerate(index) if 'photfile' in s]], dtype=str)
    pndatasets=len(photfile)
    if any('g1p' in s for s in index): g1p=np.array(invals[[i for i, s in enumerate(index) if 'g1p' in s]], dtype=np.float)
    if any('g2p' in s for s in index): g2p=np.array(invals[[i for i, s in enumerate(index) if 'g2p' in s]], dtype=np.float)
    if any('q1p' in s for s in index): q1p=np.array(invals[[i for i, s in enumerate(index) if 'q1p' in s]], dtype=np.float)
    if any('q2p' in s for s in index): q2p=np.array(invals[[i for i, s in enumerate(index) if 'q2p' in s]], dtype=np.float)
    if any('filternumber' in s for s in index): 
        filternumber=np.array(invals[[i for i, s in enumerate(index) if 'filternumber' in s]], dtype=np.int)
    else:
        filternumber=np.ones(pndatasets,dtype=np.int)
    pnfilters=np.max(filternumber)
    struc1['pnfilters']=pnfilters
    struc1['pndatasets']=pndatasets

    if any('photlcflag' in s for s in index): 
        photlcflag=struc1['photlcflag']
    else:
        photlcflag='q'
    if photlcflag == 'q':
        try:
            q1p, q2p
        except NameError:
            q1p=(g1p+g2p)**2
            q2p=g1p/(2.0*(g1p+g2p))
            for i in range(0,pnfilters):
                index=np.append(index, ['q1p'+str(i+1), 'q2p'+str(i+1)],axis=0)
                invals=np.append(invals, [q1p[i], q2p[i]],axis=0)
                struc1['q1p'+str(i+1)], struc1['q2p'+str(i+1)] = q1p[i], q2p[i]
                if (any('g1p'+str(i+1) in s for s in priorindex)) & (any('g2p'+str(i+1) in s for s in priorindex)):
                    priorindex=np.append(priorindex,['q1p'+str(i+1),'q2p'+str(i+1)],axis=0)
                    sq1p=np.sqrt(2.0*(priorstruc['g1p'+str(i+1)]**2+priorstruc['g2p'+str(i+1)]**2))
                    sq2p=g1p[i]/(2.0*(g1p[i]+g2p[i]))*np.sqrt(priorstruc['g1p'+str(i+1)]**2/g1p[i]**2+(priorstruc['g1p'+str(i+1)]**2+priorstruc['g2p'+str(i+1)]**2)/(g1p[i]+g2p[i])**2)
                    priorinvals=np.append(priorinvals,[sq1p,sq2p],axis=0)
                    priorstruc['q1p'+str(i+1)], priorstruc['q2p'+str(i+1)] = sq1p, sq2p
                    priorstruc['index']=np.append(priorstruc['index'], ['q1p'+str(i+1),'q2p'+str(i+1)],axis=0)
                    priorstruc['invals']=np.append(priorstruc['invals'], [sq1p,sq2p],axis=0)
                
    #now read in the data
    for i in range(0, pndatasets):
        #test=np.loadtxt(photfile[i])#ptime1,pflux1,perror1,pexptime1=np.loadtxt(photfile[i])#,dtype={'formats':(np.float,np.float,np.float,np.int)})
        test=ascii.read(photfile[i])
        ptime1,pflux1,perror1,pexptime1=test[0][:],test[1][:],test[2][:],test[3][:]
        goods=np.where((ptime1 != -1.) & (pflux1 != -1.))
        ptime1,pflux1,perror1,pexptime1=ptime1[goods],pflux1[goods],perror1[goods],pexptime1[goods]
    #check to see if using Kepler cadence and, if so, correct to exposure times
        if any('cadenceflag'+str(i+1) in s for s in index):
            pexptime1=np.array(pexptime1, dtype=float)
            longcad=np.where(pexptime1 == 1)
            shortcad=np.where(pexptime1 == 0)
            if struc1['cadenceflag'+str(i+1)] == 'kepler':
                pexptime1[longcad], pexptime1[shortcad] = 30., 1.
                pexptime1=pexptime1/(60.*24.)
            if struc1['cadenceflag'+str(i+1)] == 'tess':
                pexptime1[longcad], pexptime1[shortcad] = 30., 2.
                pexptime1=pexptime1/(60.*24.)
            if struc1['cadenceflag'+str(i+1)] == 'corot':
                pexptime1[longcad], pexptime1[shortcad] = 512., 32.
                pexptime1=pexptime1/(60.*60.*24.)
        if any('expunit'+str(i+1) in s for s in index):
            if (struc1['expunit'+str(i+1)] == 's') or (struc1['expunit'+str(i+1)] == 'seconds'): pexptime1=pexptime1/(60.*60.*24.)
            if (struc1['expunit'+str(i+1)] == 'm') or (struc1['expunit'+str(i+1)] == 'minutes'): pexptime1=pexptime1/(60.*24.)
            if (struc1['expunit'+str(i+1)] == 'h') or (struc1['expunit'+str(i+1)] == 'hours'): pexptime1=pexptime1/(24.)
            if (struc1['expunit'+str(i+1)] == 'd') or (struc1['expunit'+str(i+1)] == 'days'): pexptime1=pexptime1/(1.)
        else:
            struc1['expunit'+str(i+1)] = 'days'

        if i == 0:
            ptime, pflux, perror, pexptime = ptime1,pflux1,perror1,pexptime1
            pfilter=np.ones(len(ptime))
            pdataset=np.ones(len(ptime))
            if args.gp:
                if any('gppuse'+str(i+1) in s for s in index):
                    gppuse=np.zeros(len(ptime))+np.float(struc1['gppuse'+str(i+1)])
                else:
                    gppuse=np.ones(len(ptime))
                print('gppusei', 'gppuse'+str(i+1))
        else:
            ptime, pflux, perror, pexptime = np.append(ptime,ptime1), np.append(pflux,pflux1), np.append(perror,perror1), np.append(pexptime,pexptime1)
            pfilter=np.append(pfilter, np.zeros(len(ptime1))+filternumber[i])
            pdataset=np.append(pdataset, np.zeros(len(ptime1))+i+1)
            if args.gp:
                if any('gppuse'+str(i+1) in s for s in index):
                    gppuse=np.append(gppuse,np.zeros(len(ptime1))+np.int(struc1['gppuse'+str(i+1)]))
                else:
                    gppuse=np.append(gppuse,np.ones(len(ptime1)))


 

    #flux ratio if doing EB
    if args.binary:
        if any('binfflag' in s for s in index): 
            binfflag=struc1['binfflag']
        else:
            binfflag='rprsfluxr'

            
        fluxrat=np.array(invals[[i for i, s in enumerate(index) if 'fluxrat' in s]], dtype=np.float)#struc1['fluxrat']


            
if args.photometry or args.tomography or args.rm:
    if any('rhostar' in s for s in index): rhostar=np.float(struc1['rhostar'])
    if any('aors' in s for s in index): aors=np.array(invals[[i for i, s in enumerate(index) if 'aors' in s]], dtype=np.float) #NOTE: if args.binary, aors is actually (a/(R1+R2)), not (a/R*)!
    if any ('cosi' in s for s in index): cosi=np.array(invals[[i for i, s in enumerate(index) if 'cosi' in s]], dtype=np.float)
    rprs=np.array(invals[[i for i, s in enumerate(index) if 'rprs' in s]], dtype=np.float)
    bpar=np.array(invals[[i for i, s in enumerate(index) if 'bpar' in s]], dtype=np.float)

    if any('rhobaflag' in s for s in index):
        rhobaflag=struc1['rhobaflag']
    else:
        rhobaflag='rhostarb'

if args.tomography or args.line or args.rm:
    if any('g1t' in s for s in index): g1t=np.float(struc1['g1t'])
    if any('g2t' in s for s in index): g2t=np.float(struc1['g2t'])
    if any('q1t' in s for s in index): q1t=np.float(struc1['q1t'])
    if any('q2t' in s for s in index): q2t=np.float(struc1['q2t'])
    if any('macroturb' in s for s in index): macroturb=np.float(struc1['macroturb'])
    if any('tomlcflag' in s for s in index): 
        tomlcflag=struc1['tomlcflag']
    else:
        tomlcflag='q'
    if tomlcflag == 'q':
        try:
            q1t, q2t
        except NameError:
            q1t=(g1t+g2t)**2
            q2t=g1t/(2.0*(g1t+g2t))
            index=np.append(index, ['q1t', 'q2t'],axis=0)
            invals=np.append(invals, [q1t, q2t],axis=0)
            struc1['q1t'], struc1['q2t'] = q1t, q2t
        
        if (any('g1t' in s for s in priorindex)) & (any('g2t' in s for s in priorindex)):
            priorindex=np.append(priorindex,['q1t','q2t'],axis=0)
            sq1t=np.sqrt(2.0*(priorstruc['g1t']**2+priorstruc['g2t']**2))
            sq2t=g1t/(2.0*(g1t+g2t))*np.sqrt(priorstruc['g1t']**2/g1t**2+(priorstruc['g1t']**2+priorstruc['g2t']**2)/(g1t+g2t)**2)
            priorinvals=np.append(priorinvals,[sq1t,sq2t],axis=0)
            priorstruc['q1t'], priorstruc['q2t'] = sq1t, sq2t
    
    #if args.binary: #handle double-lined binary parameters

if args.tomography or args.rm:
    llambda=np.array(invals[[i for i, s in enumerate(index) if 'lambda' in s]], dtype=np.float)

if args.tomography:
    tomfile=np.array(invals[[i for i, s in enumerate(index) if 'tomfile' in s]], dtype=np.str)#struc1['tomfile']

    if any('tomdrift' in s for s in index):
        tomdriftc=np.array(invals[[i for i, s in enumerate(index) if 'tomdriftc' in s]], dtype=np.float) #constant term
        tomdriftl=np.array(invals[[i for i, s in enumerate(index) if 'tomdriftl' in s]], dtype=np.float) #linear term

    #now read in the data
    ntomsets=len(tomfile)
    tomdict={}
    tomdict['ntomsets']=ntomsets
    tomdict['nexptot'], tomdict['nvabsfinemax'],tomdict['whichvabsfinemax'] =0, 0, 0
    for i in range (0,ntomsets):
        tomnlength=len(tomfile[i])
        lastthree=tomfile[i][tomnlength-3:tomnlength]
        if lastthree == 'sav':
            import idlsave
            try:
                datain=idlsave.read(struc1['tomfile'+str(i+1)])
            except IOError:
                print('Your tomographic input file either does not exist or is in the wrong format.')
                print('Please supply a file in IDL save format.')
                print('Exiting now.')
                sys.exit()

    
            profarr = datain.profarr
            tomdict['avgprof'+str(i+1)] = datain.avgprof
            tomdict['avgproferr'+str(i+1)] = datain.avgproferr
            profarrerr = datain.profarrerr
            profarr = np.transpose(profarr) #idlsave reads in the arrays with the axes flipped
            tomdict['profarr'+str(i+1)]=profarr*(-1.0)
            tomdict['profarrerr'+str(i+1)] = np.transpose(profarrerr)
            tomdict['ttime'+str(i+1)]=datain.bjds
            tomdict['tnexp'+str(i+1)] = tomdict['ttime'+str(i+1)].size
            tomdict['texptime'+str(i+1)]=datain.exptimes
            tomdict['vabsfine'+str(i+1)] = datain.vabsfine
            tomdict['nexptot']+=tomdict['tnexp'+str(i+1)]
            if len(tomdict['vabsfine'+str(i+1)]) > tomdict['nvabsfinemax']: tomdict['nvabsfinemax'], tomdict['whichvabsfinemax'] = len(tomdict['vabsfine'+str(i+1)]), i+1

        if lastthree == 'pkl':
            import pickle

            try:
                datain=pickle.load(open(struc1['tomfile'+str(i+1)],"rb"))
            except IOError:
                print('Your tomographic input file either does not exist or is in the wrong format.')
                print('Please supply a file in pickle format.')
                print('Exiting now.')
                sys.exit()

            if not 'george' in struc1['tomfile'+str(i+1)]:
                tomdict['profarr'+str(i+1)], tomdict['profarrerr'+str(i+1)], tomdict['avgprof'+str(i+1)], tomdict['avgproferr'+str(i+1)], tomdict['ttime'+str(i+1)], tomdict['texptime'+str(i+1)], tomdict['vabsfine'+str(i+1)] = np.array(datain['profarr'],dtype=float), np.array(datain['profarrerr'],dtype=float), np.array(datain['avgprof'],dtype=float), np.array(datain['avgproferr'],dtype=float), np.array(datain['ttime'],dtype=float), np.array(datain['texptime'],dtype=float), np.array(datain['vabsfine'],dtype=float)

                tomdict['profarr'+str(i+1)]*=(-1.)
            else:
                tomdict['profarr'+str(i+1)], tomdict['ttime'+str(i+1)], tomdict['vabsfine'+str(i+1)], tomdict['texptime'+str(i+1)], tomdict['avgprof'+str(i+1)] = np.array(datain[0], dtype=float), np.array(datain[2], dtype=float)-2400000., np.array(datain[3], dtype=float), np.array(datain[4], dtype=float), np.array(datain[5], dtype=float)
                if struc1['obs'+str(i+1)] == 'harpsn': Resolve=120000.0
                if struc1['obs'+str(i+1)] == 'tres' : Resolve=44000.0
                tomdict['profarr'+str(i+1)]/=1.1 #TEMPORARY KLUGE!!!
                tomdict['profarr'+str(i+1)]*=(-1.0)
                 #downsample!
                vabsfinetemp=np.arange(np.min(tomdict['vabsfine'+str(i+1)]), np.max(tomdict['vabsfine'+str(i+1)]),(2.9979e5/Resolve)/2.)
                ntemp=len(vabsfinetemp)
                profarrtemp=np.zeros((len(tomdict['texptime'+str(i+1)]),ntemp))
                for iter in range(0,len(tomdict['texptime'+str(i+1)])):
                    profarrtemp[iter,:]=np.interp(vabsfinetemp,tomdict['vabsfine'+str(i+1)],tomdict['profarr'+str(i+1)][iter,:])
                avgproftemp=np.interp(vabsfinetemp,tomdict['vabsfine'+str(i+1)],tomdict['avgprof'+str(i+1)])
                tomdict['profarr'+str(i+1)]=profarrtemp
                tomdict['vabsfine'+str(i+1)]=vabsfinetemp
                tomdict['avgprof'+str(i+1)]=avgproftemp
                outs=np.where(np.abs(tomdict['vabsfine'+str(i+1)] ) > np.float(struc1['vsini'])*1.1)
                tomdict['profarrerr'+str(i+1)]=tomdict['profarr'+str(i+1)]*0.0+np.std(tomdict['profarr'+str(i+1)][:,outs[0]])
                tomdict['avgproferr'+str(i+1)]=tomdict['avgprof'+str(i+1)]*0.0+np.std(tomdict['avgprof'+str(i+1)][outs[0]])
               
                
                
            

        
        #cut off the few pixels on the edges, which are often bad
        tomdict['nvabsfine'+str(i+1)]=len(tomdict['vabsfine'+str(i+1)])
        tomdict['profarr'+str(i+1)]=tomdict['profarr'+str(i+1)][:,3:tomdict['nvabsfine'+str(i+1)]-2]
        tomdict['profarrerr'+str(i+1)]=tomdict['profarrerr'+str(i+1)][:,3:tomdict['nvabsfine'+str(i+1)]-2]
        tomdict['vabsfine'+str(i+1)]=tomdict['vabsfine'+str(i+1)][3:tomdict['nvabsfine'+str(i+1)]-2]
        tomdict['avgprof'+str(i+1)]=tomdict['avgprof'+str(i+1)][3:tomdict['nvabsfine'+str(i+1)]-2]
        tomdict['avgproferr'+str(i+1)]=tomdict['avgproferr'+str(i+1)][3:tomdict['nvabsfine'+str(i+1)]-2]
        nvabsfine1=tomdict['nvabsfine'+str(i+1)]
        tomdict['nvabsfine'+str(i+1)]=len(tomdict['vabsfine'+str(i+1)])
        tomdict['whichplanet'+str(i+1)] = struc1['whichtomplanet'+str(i+1)]

        

    if any('tomflat' in s for s in index): 
        if struc1['tomflat'] == 'True':
            tomflat=True
        else:
            tomflat=False
    else:
        tomflat=False

    if any('fitflat' in s for s in index): 
        if struc1['fitflat'] == 'True':
            fitflat=True
        else:
            fitflat=False




    else:
        struc1['fitflat'] = 'False'
        index=np.append(index,'fitflat')
        invals=np.append(invals,'False')
        fitflat=False

    if not tomflat and fitflat:
        for j in range (0,ntomsets):
            for i in range (0,tnexp) : 
                tomdict['profarr[i, : ]'+str(j+1)]-=tomdict['avgprof'+str(j+1)]
                tomdict['profarrerr'+str(j+1)][i,:]=np.sqrt(tomdict['profarrerr'+str(j+1)][i,:]**2+tomdict['avgproferr'+str(j+1)]**2)

    if any('tomfftfile' in s for s in index):
        if ntomsets > 1:
            print('FFT is not yet implemented for multi-tomographic dataset fits!!!')
            print('exiting now.')
            sys.exit()
        fftin = idlsave.read(struc1['tomfftfile'])
        ffterrin = idlsave.read(struc1['tomffterrfile'])
        profarrerr = np.transpose(ffterrin.filterr)
        profarrerr=profarrerr[:,3:nvabsfine1-2]
        mask = np.transpose(fftin.mask)
        mask=mask[:,3:nvabsfine1-2]
        profarr = horus.fourierfilt(profarr, mask)
        dofft = True
    else:
        dofft = False

    if any('spot' in s for s in index):
        args.spots = True
        nspots = struc1['nspots']
        t0spots = np.array(invals[[i for i, s in enumerate(index) if 'spott0' in s]], dtype=np.float)
        rspots = np.array(invals[[i for i, s in enumerate(index) if 'spotrad' in s]], dtype=np.float)
        Prot = np.float(struc1['Prot'])
    else:
        args.spots = False

if args.line or args.tomography:

    if any('linecenter' in s for s in index):
        linecenter=np.array(invals[[i for i, s in enumerate(index) if 'linecenter' in s]], dtype=np.float)
    else:
        linecenter=0.0
        #struc1['linecenter']=0.0
        #perturbstruc['linecenter']=0.1
        #perturbindex=np.append(perturbindex,'linecenter')
        #perturbinvals=np.append(perturbinvals,0.1)
        
if args.line:
        
    if any('tomfile' in i for i in struc1['linefile']):
        lnum=np.float(struc1['linefile'][7])
        if args.tomography:
            lineprof = tomdict['avgprof'+str(lnum)]
            lineerr = tomdict['avgproferr'+str(lnum)]
            linevel = tomdict['vabsfine'+str(lnum)]
        else:
            import idlsave
            try:
                datain=idlsave.read(struc1['tomfile'])
            except IOError:
                print('Your tomographic input file either does not exist or is in the wrong format.')
                print('Please supply a file in IDL save format.')
                print('Exiting now.')
                sys.exit()

            lineprof  = datain.avgprof
            lineerr = datain.avgproferr
            linevel = datain.vabsfine

    else:
        namelength=len(struc1['linefile'])
        if struc1['linefile'][namelength-4:namelength] == '.sav':
            if not args.tomography:
                import idlsave
                datain=idlsave.read(struc1['linefile'])
                lineprof  = datain.avgprof
                lineerr = datain.avgproferr
                linevel = datain.vabsfine


if args.rvs or args.rm:
    #read in the data
    if not args.binary:
        temp=ascii.read(struc1['rvfile'])#np.loadtxt(struc1['rvfile'])
        rtime,rv,rverror,rdataset=temp[0][:],temp[1][:],temp[2][:],temp[3][:]
    else:
        rtime, rv1, rverror1, rv2, rverror2, rdataset=np.loadtxt(struc1['rvfile'])
        rv=np.append(rv1,rv2) 
        rverror=np.append(rverror1, rverror2) 
    rndatasets=np.max(rdataset)
    if any('gamma' in s for s in index): 
        gamma=np.array(invals[[i for i, s in enumerate(index) if 'gamma' in s]], dtype=np.float)
    else:
        gamma=np.zeros(rndatasets)
        for i in range (0,rndatasets):
            struc1['gamma'+str(i+1)]=gamma[i]
            invals=np.append(invals,0)
            index=np.append(index,'gamma'+str(i+1))

    if any('fixgam' in s for s in index): 
        fixgamma=struc1['fixgam']
    else:
        fixgamma='False'
    
    if any('rvtrend' in s for s in index):
        fittrend=True
        rvtrend=np.float(struc1['rvtrend'])
        if any('rvtrendquad' in s for s in index):
            rvtrendquad=np.float(struc1['rvtrendquad'])
    else:
        fittrend=False

    #check for jitter and use if being fit
    if any('jitter' in s for s in index):
        jitter=np.array(invals[[i for i, s in enumerate(index) if 'jitter' in s]], dtype=np.float)
        args.fitjitter=True
    else:
        args.fitjitter=False

    if args.rvs: semiamp=np.array(invals[[i for i, s in enumerate(index) if 'semiamp' in s]], dtype=np.float)

    
#copy in the TTV model parameters
if args.ttvs:
    
    #dottvs1=np.array(invals[[i for i, s in enumerate(index) if 'dottv' in s]], dtype=bool)
    
    dottvs=np.zeros(nplanets, dtype=bool)
    for i in range (0,nplanets):
        if any('dottv'+str(i+1) in s for s in index): 
            if struc1['dottv'+str(i+1)] == 'True':
                dottvs[i]=True
 
    #whichttvs=np.where(dottvs == True)
    #nttvs=len(whichttvs[0])
    ttvpars=np.array(invals[[i for i, s in enumerate(index) if (('ttv' in s) & ('par' in s))]], dtype=np.float)
    ttvparnames=np.array(index[[i for i, s in enumerate(index) if (('ttv' in s) & ('par' in s))]], dtype=np.str)
    nttvpars=len(ttvpars)
    modtype=np.array(invals[[i for i, s in enumerate(index) if ('ttvmodtype' in s)]], dtype=np.str)#struc1['ttvmodtype']

if args.dilution:
    if any('dilution' in s for s in index): 
        dilution=np.array(invals[[i for i, s in enumerate(index) if 'dilution' in s]], dtype=np.float)
        dilutionnames=np.array(index[[i for i, s in enumerate(index) if 'dilution' in s]], dtype=np.str)
        ndilute=len(dilution)

#copy in the Gaussian process parameters
if args.gp:
    gppars=np.array(invals[[i for i, s in enumerate(index) if (('gp' in s) & ('par' in s))]], dtype=np.float)
    gpparnames=np.array(index[[i for i, s in enumerate(index) if (('gp' in s) & ('par' in s))]], dtype=np.str)
    ngppars=len(gppars)
    gpmodval=np.array(invals[[i for i, s in enumerate(index) if ('gpmodtype' in s)]], dtype=np.str)
    gpmodname=np.array(index[[i for i, s in enumerate(index) if ('gpmodtype' in s)]], dtype=np.str)
    gpmodtype={}
    for i in range (0,len(gpmodval)):
        gpmodtype[gpmodname[i]]=gpmodval






#all of the necessary functions will go here

def ttvmodel(tin, Per, epoch, modtype, instruc, halflength, pnumber):
    #calculate predicted transit times
    inindex=list(instruc.keys())
    time1, time2 = np.min(tin),np.max(tin)
    firstepoch = np.ceil((time1-epoch)/Per)
    current = epoch + Per*firstepoch
    #ttvepoch = epoch + Per*firstepoch
    ttvepoch=0.0
    while current <= time2:
        if np.min(np.abs(current-tin)) <= halflength:
            ttvepoch=np.append(ttvepoch,current)
        current+=Per
    nepochs=len(ttvepoch)-1
    ttvepoch=ttvepoch[1:nepochs+1]
    namehere='ttv'+str(pnumber)+'par'
    if modtype == 'sines':
        nsines=len(np.array(instruc[[i for i, s in enumerate(inindex) if 'parP' in s]], dtype=np.float))
        for i in range (0,nsines):
            #inargs: period, amplitude, offset
            ttvepoch+=instruc[namehere+'A'+str(i+1)]*np.sin((ttvepoch-instruc[namehere+'O'+str(i+1)])*2.*np.pi/instruc[namehere+'P'+str(i+1)])
    return ttvepoch


def ttvshift(tin, Per, epoch, ttvepochs, halflength):
    nepochs, nexp =len(ttvepochs), len(tin)
    thalfspan=dur+maxdev
    tout=np.zeros(nexp)
    for i in range (0,nepochs):
        herein=np.where((tin >= ttvepochs[i]-halflength*1.2) & (tin <= ttvepochs[i]+halflength*1.2))
        thisepoch=np.round((ttvepochs[i]-epoch)/Per)
        calcepoch=epoch+Per*thisepoch
        diff=calcepoch-ttvepochs[i]
        tout[herein]=tin[herein]+diff
    return tout
        
    

def resampler(modin,t,tprime,cadence):
    #t is the raw time, tprime is what we want to resample to
    #cadence has the same length as tprime
    nexp=len(tprime)
    nmodps=len(t)
    modout=np.zeros(nexp)
    for i in range (0,nexp):
        heres=np.where(np.abs(t-tprime[i]) <= cadence[i]/2.0)
        modout[i]=np.mean(modin[heres[0]])
    return modout

def photmodel(struc):

    #startmodel=timemod.time()-1458836000.
    #print 'Starting the model at ',startmodel
    if struc['photmodflag'] == 'batman':
        params=batman.TransitParams()
        params.t0=0.0
        if any('secondary' in s for s in list(struc.keys())):
            params.t0+=struc['secondary']
        params.per=struc['Per']
        params.rp=struc['rprs']
        params.a=struc['aors']
        params.inc=struc['inc']
        params.ecc=struc['ecc']
        params.w=struc['omega']
        params.limb_dark='quadratic' #will have to de-hardcode this eventually...
        params.u=[struc['g1'],struc['g2']]

        #print struc

        tenminutes=10.0/(60.*24.)
        nexp=len(struc['t'])
        flux=np.zeros(nexp)
        #print 'The params at ',startmodel,' are ',params.per, params.rp, params.a, params.inc, params.ecc, params.w, params.u, nexp
        #print 'The times are ',struc['t'],startmodel
        if any(t < tenminutes for t in struc['exptime']):
            shorts=np.where(struc['exptime'] < tenminutes)
            #print 'calling batman short ',startmodel
            ms=batman.TransitModel(params,struc['t'][shorts],nthreads=1)
            #print 'batman short done',startmodel
            flux[shorts]=ms.light_curve(params)
        if any(t >= tenminutes for t in struc['exptime']):
            longs=np.where(struc['exptime'] >= tenminutes)
            if struc['longflag'] == 'batman':
                #print 'calling batman long ',startmodel
                ml=batman.TransitModel(params,struc['t'][longs],nthreads=1, supersample_factor=50, exp_time=np.mean(struc['exptime'][longs])) #for now, just use the mean of the long exposure time--fix later to be able to handle multiple exposure lengths
                #print 'batman long done',startmodel
                flux[longs]=ml.light_curve(params)
            else:
                #if any(t < tenminutes for t in struc['exptime']):
                #if already have short cadence model
                #    flux[longs]=resampler(flux[shorts], struc['t'][shorts], struc['t'][longs], struc['exptime'][longs])
                #else:
                #need to make new model
                maxexp=np.max(struc['exptime'][longs])
                ttemp=np.arange(np.min(struc['t'][longs])-maxexp*2., np.max(struc['t'][longs])+maxexp*2., np.min(struc['exptime'][longs])/100.)
                #print 'calling batman long ',startmodel
                ml=batman.TransitModel(params,ttemp,nthreads=1)
                #print 'batman long done',startmodel
                ftemp=ml.light_curve(params)
                flux[longs]=resampler(ftemp, ttemp, struc['t'][longs], struc['exptime'][longs])

            

                #print 'Done with photmodel ',startmodel

    elif struc['photmodflag'] == 'jktebop':
        #timemod.sleep(np.random.random()*0.1)
        timestamp=str(multiprocessing.current_process().pid)+'.'+str(timemod.time()-1537000000.)#str((timemod.time()-1537000000.)+np.random.randint(1,1e7))#str(timemod.time()-1537000000.)
        f=open('temp'+timestamp+'.in','w')
        f.write('2 1 Task to do (from 2 to 9)   Integ. ring size (deg) \n')
        f.write(str(1./struc['aors'])+'  '+str(struc['rprs'])+' Sum of the radii           Ratio of the radii \n')
        f.write(str(struc['inc'])+'  '+str(-1.)+' Orbital inclination (deg)  Mass ratio of system \n')
        f.write(str(struc['ecc']+10.)+'  '+str(struc['omega'])+' ecosw or eccentricity      esinw or periastron long \n')
        f.write('1.0 1.0 Gravity darkening (star A) Grav darkening (star B) \n')
        f.write(str(1./struc['fluxrat'])+'  '+str(struc['dilution']*(-1.))+' Surface brightness ratio   Amount of third light \n')
        f.write('quad  quad  LD law type for star A     LD law type for star B \n')
        f.write(str(struc['g1'])+'  '+str(struc['g1'])+' LD star A (linear coeff)   LD star B (linear coeff) \n')
        f.write(str(struc['g2'])+'  '+str(struc['g2'])+' LD star A (nonlin coeff)   LD star B (nonlin coeff) \n')
        f.write('0.0  0.0  Reflection effect star A   Reflection effect star B \n')
        f.write('0.0  0.0  Phase of primary eclipse   Light scale factor (mag) \n')
        f.write('temp'+timestamp+'.out Output file name (continuous character string) \n')
        f.close()

        os.system('rm -f temp'+timestamp+'.out')

        os.system('./../../system/jktebop/jktebop/jktebop temp'+timestamp+'.in')

        phase,mag,l1,l2,l3=np.loadtxt('temp'+timestamp+'.out')
        os.system('rm -f temp'+timestamp+'.out')
        os.system('rm -f temp'+timestamp+'.in')
        highs=np.where(phase > 0.5)
        phase[highs[0]]-=1.0

        #os.system('rm -f temp'+timestamp+'.out')
        #os.system('rm -f temp'+timestamp+'.in')

        mflux1=10.**(((-1.)*mag)/2.5)
        
        flux=resampler(mflux1,phase*struc['Per'],struc['t'],struc['exptime'])

        #pl.plot(phase*struc['Per'],mflux1,'ro')
        #pl.plot(struc['t'],flux,'bo')
        #pl.xlim([-0.1,0.1])
        #pl.draw()
        #pl.pause(5.0)
        #pl.clf()
            
    return flux

def rmmodel(struc):

    Per=struc['Per']
    v0=struc['intwidth']
    vsini=struc['vsini']
    ecc=struc['ecc']
    omega=struc['omega']*np.pi/180. #ASSUMED IN DEGREES
    llambda=struc['lambda']*np.pi/180.0
    b=struc['b']
    nexp=len(struc['t'])
    t=struc['t'] 
    aors=struc['aors']
    struc['exptime']=np.zeros(nexp)+np.mean(np.diff(t,n=1))#estimate!

    Omega=2.0*np.pi/Per*(1.0-ecc**2)**(-3.0/2.0)*(1.0+ecc*np.sin(omega))**2 #angular velocity of planet
    thetat=Omega*t+np.pi/2.0-omega #angular position of planet at time t
    z=aors*(1.0-ecc**2)/(1.0+ecc*np.cos(thetat))*(np.sin(omega)*np.sin(thetat)-np.cos(omega)*np.cos(thetat)) #position of planet on coordinate along star. This should be in units of stellar radii since aors is providing the units.
    x=z*np.cos(llambda)+b*np.sin(llambda) #coordinate along axis perpendicular to the stellar rotation axis
    vp=vsini*x

    #oots=np.abs(x) > 1.+struc['rprs']
    #vp[oots]*=0.0

    f=(photmodel(struc)-1.0)*(-1.)

    vrm=(-1.)*f*vp*((2.*v0**2+2.*vsini**2)/(2.*v0**2+vsini**2))**(3./2.)*(1.-(vp**2)/(2.*v0**2+vsini**2))

    return vrm#/1000. #m/s->km/s

def lnlike(theta, parstruc, data, nplanets, inposindex, instruc, args):
    #temptime=timemod.time()
    if args.time: temptime=timemod.time()-1458836000.
    #print 'starting lnlike ',temptime
    index=instruc['index']
    nfigs=1
    #if args.time: print 'starting to parse out the eccentricity ',temptime
    #parse out the eccentricity
    if any('sinw' in s for s in inposindex) or any('ecc' in s for s in inposindex):
        if any('sesinw' in s for s in inposindex):
            #omega=np.arctan(theta[[i for i, s in enumerate(inposindex) if 'sesinw' in s]]/theta[[i for i, s in enumerate(inposindex) if 'secosw' in s]])
            #ecc=theta[[i for i, s in enumerate(inposindex) if 'secosw' in s]]**2/np.cos(omega)**2
            ecc=theta[[i for i, s in enumerate(inposindex) if 'sesinw' in s]]**2+theta[[i for i, s in enumerate(inposindex) if 'secosw' in s]]**2
            omega=np.arccos(theta[[i for i, s in enumerate(inposindex) if 'secosw' in s]]/np.sqrt(ecc))
            news=np.where(theta[[i for i, s in enumerate(inposindex) if 'sesinw' in s]] < 0.)
            omega[news]=2.*np.pi-omega[news]
            if any(not np.isfinite(t) for t in omega):
                bads=np.where(np.isfinite(omega) == True)
                omega[bads]=np.pi/2.0
                temp=theta[[i for i, s in enumerate(inposindex) if 'sesinw' in s]]
                ecc[bads]=temp[bads]**2
        if any('ecc' in s for s in inposindex):
            ecc=theta[[i for i, s in enumerate(inposindex) if 'ecc' in s]]
            omega=theta[[i for i, s in enumerate(inposindex) if 'omega' in s]]

        if any('ecsinw' in s for s in inposindex):
            ecc=np.sqrt(theta[[i for i, s in enumerate(inposindex) if 'ecsinw' in s]]**2+theta[[i for i, s in enumerate(inposindex) if 'eccosw' in s]]**2)
            omega=np.arccos(theta[[i for i, s in enumerate(inposindex) if 'eccosw' in s]]/ecc)
            news=np.where(theta[[i for i, s in enumerate(inposindex) if 'ecsinw' in s]] < 0.)
            omega[news]=2.*np.pi-omega[news]
            if any(not np.isfinite(t) for t in omega):
                bads=np.where(np.isfinite(omega) == True)
                omega[bads]=np.pi/2.0
                temp=theta[[i for i, s in enumerate(inposindex) if 'ecsinw' in s]]
                ecc[bads]=temp[bads]

        omega*=180./np.pi #radians->degrees
    #will need to do this for esinw, ecosw. code up later.



    else:
        ecc=np.zeros(nplanets)
        omega=np.zeros(nplanets)+90.

    if args.photometry or args.tomography or args.rm:
        if any('rhostar' in s for s in inposindex) and any('bpar' in s for s in inposindex):
            aors=215.*parstruc['rhostar']**(1./3.)*(theta[[i for i, s in enumerate(inposindex) if 'Per' in s]]/365.25)**(2./3.)*((1.+ecc*np.sin(omega*np.pi/180.))/np.sqrt(1.-ecc**2))
            cosi=theta[[i for i, s in enumerate(inposindex) if 'bpar' in s]]/aors*(1.0+ecc*np.sin(omega*np.pi/180.))/(1.0-ecc**2)
            inc=np.arccos(cosi)*180./np.pi
        if any('aors' in s for s in inposindex) and any('bpar' in s for s in inposindex):
            aors=theta[[i for i, s in enumerate(inposindex) if 'aors' in s]]
            cosi=theta[[i for i, s in enumerate(inposindex) if 'bpar' in s]]/aors*(1.0+ecc*np.sin(omega*np.pi/180.))/(1.0-ecc**2)
            inc=np.arccos(cosi)*180./np.pi
            #NOTE that for this option for multiplanet systems, there is currently no enforcement of rhostar being the same for all planets!!!
        if any(t > 1.0 for t in np.abs(theta[[i for i, s in enumerate(inposindex) if 'bpar' in s]])-theta[[i for i, s in enumerate(inposindex) if 'rprs' in s]]): #bpar and rprs must be in the same planet order for this to work--fix later
            #if args.time: print 'ending lnlike, no transit ',temptime
            return -np.inf #handle if no transit
        if any('cosi' in s for s in inposindex) and any('aors' in s for s in inposindex):
            aors=theta[[i for i, s in enumerate(inposindex) if 'aors' in s]]
            cosi=theta[[i for i, s in enumerate(inposindex) if 'cosi' in s]]
            inc=np.arccos(cosi)*180./np.pi
            
        
        if not args.binary:
            dur=theta[[i for i, s in enumerate(inposindex) if 'Per' in s]]/np.pi*1./aors*np.sqrt(1.0-ecc**2)/(1.0+ecc*np.cos(omega*np.pi/180.))*np.sqrt((1.0+theta[[i for i, s in enumerate(inposindex) if 'rprs' in s]])**2-theta[[i for i, s in enumerate(inposindex) if 'bpar' in s]]**2)
        else:
            dur=theta[[i for i, s in enumerate(inposindex) if 'Per' in s]]*10. #hack for now!!!--just use all of the datapoints
        #print 'The duration is ',dur


    lnl=0.0
    #do photometric fit
    if args.photometry:
        nexpp=len(data['ptime'])
        model=np.zeros(nexpp)
        #if args.time:  print 'starting the planet loop ',temptime
        for i in range (0,nplanets):
            #if args.time: print 'starting the loop for planet ',i+1,' at ',temptime
            modelstruc={'Per':parstruc['Per'+str(i+1)], 'rprs':parstruc['rprs'+str(i+1)], 'aors':aors[i], 'inc':inc[i], 'ecc':ecc[i], 'omega':omega[i]}
            if args.plotbest: mydict[str(i)] = {} ## creating Ellie's subdict for this photometry file

            
            phased = np.mod(data['ptime']-parstruc['epoch'+str(i+1)], parstruc['Per'+str(i+1)])
            highs = np.where(phased > parstruc['Per'+str(i+1)]/2.0)
            phased[highs]-=parstruc['Per'+str(i+1)] #this is still in days
            maxexp=np.max(data['pexptime'])
            if not args.ttvs: 
                closes = np.where(np.abs(phased) <= (dur[i]/2.+maxexp)*1.5)
            elif not data['dottvs'][i]:
                closes = np.where(np.abs(phased) <= (dur[i]/2.+maxexp)*1.5)
            else:
                #get the expected ttv epochs from the model here
                
                inargs=np.array(theta[[k for k, s in enumerate(inposindex) if 'ttv'+str(i+1)+'par' in s]], dtype=np.float)
                inargdex=np.array(inposindex[[k for k, s in enumerate(inposindex) if 'ttv'+str(i+1)+'par' in s]], dtype=np.str)
                nargs=len(inargs)
                inargstruc={}
                for tparcount in range (0,nargs):
                    inargstruc[inargdex[tparcount]]=inargs[tparcount]
                
                halflength=dur[i]/2.+maxexp+np.sum(np.array(theta[[k for k, s in enumerate(inposindex) if 'ttv'+str(i+1)+'parA' in s]], dtype=np.float))
                ttvepochs=ttvmodel(data['ptime'], parstruc['Per'+str(i+1)], parstruc['epoch'+str(i+1)], data['ttvmodtype'][i], inargstruc, halflength, i+1)
                halflength=dur[i]/2.+maxexp+maxttv
                closes = np.where(np.abs(phased) <= (halflength)*1.1)
                newt=ttvshift(data['ptime'][closes], parstruc['Per'+str(i+1)], parstruc['epoch'+str(i+1)], ttvepochs, halflength)
                phased[closes] = np.mod(newt-parstruc['epoch'+str(i+1)], parstruc['Per'+str(i+1)]) #this still leaves un-shifted elements in phased, but I'm pretty sure that modifying closes below will permanently exclude these problem elements
                closes1 = np.where(np.abs(phased) <= (dur[i]/2.+maxexp)*1.1)
                closes = closes[closes1] #not totally sure this is right
                
            ##print(dur[i],(dur[i]/2.+maxexp)*1.1)
            
            if any('longflag' in s for s in instruc):
                modelstruc['longflag']=instruc['longflag']
            else:
                modelstruc['longflag']='batman'

            if any('photmodflag' in s for s in instruc):
                modelstruc['photmodflag']=instruc['photmodflag']
            else:
                modelstruc['photmodflag']='batman'

            if args.binary and modelstruc['photmodflag'] == 'batman': modelstruc['aors'] = aors[i]*(1.+parstruc['rprs'+str(i+1)]) #PRIMARY eclipse, so PRIMARY in background, R*=R1, Rp=R2
                
            for j in range (0,instruc['pnfilters']):
                if args.plotbest: mydict[str(i)][str(j)] = {} ## creating Ellie's subdict for this photometry file

                #if args.time: print 'starting the filter loop ',temptime
                if any('q1p' in s for s in inposindex) and any('q1p' in s for s in inposindex):
                    modelstruc['g1'], modelstruc['g2'] = 2.0*parstruc['q2p'+str(j+1)]*np.sqrt(parstruc['q1p'+str(j+1)]), np.sqrt(parstruc['q1p'+str(j+1)])*(1.0-2.0*parstruc['q2p'+str(j+1)])
                else:
                    modelstruc['g1'], modelstruc['g2'] = parstruc['g1p'+str(j+1)], parstruc['g2p'+str(j+1)]

  
                closefilter=np.where((np.abs(phased) <= (dur[i]/2.+maxexp)*1.5) & (data['pfilter'] == j+1))
                modelstruc['t'], modelstruc['exptime'] = phased[closefilter], data['pexptime'][closefilter]
                if args.dilution: 
                    if any('dilution'+str(j+1) in s for s in inposindex):
                        modelstruc['dilution'] = parstruc['dilution'+str(j+1)]
                    else:
                        modelstruc['dilution'] = 0.0
                else:
                    modelstruc['dilution'] = 0.0
                if args.binary:
                    if data['binfflag'] == 'mycomb':
                        fluxrat=parstruc['fluxrat'+str(j+1)]/parstruc['rprs'+str(i+1)]**2
                    else:
                        fluxrat=parstruc['fluxrat'+str(j+1)]
                    modelstruc['fluxrat']=fluxrat
                        
                if args.plotbest and args.fullcurve:
                    if np.sign(np.min(phased[closefilter])) == np.sign(np.max(phased[closefilter])):
                        modelstruc['t']=np.append(modelstruc['t'],np.linspace(0.,np.min(np.abs(phased[closefilter])),100))
                        modelstruc['exptime']=np.append(modelstruc['exptime'],np.zeros(100)+np.mean(modelstruc['exptime']))
                        phased=np.append(phased,np.linspace(0.,np.min(np.abs(phased[closefilter])),100))
                        buh=1
                    else:
                        buh=0
                                                  

                if args.time: print('about to call the model ',temptime,len(phased[closefilter]))
                rawmodel=photmodel(modelstruc)#-1.0
                if args.binary and modelstruc['photmodflag'] == 'batman':
                    modelstruc['omega']-=180.
                    fsecondary=np.pi/2.-omega[i]*np.pi/180.-np.pi #true anomaly at secondary eclipse
                    Esecondary=2.*np.arctan(np.sqrt((1.-ecc[i])/(1.+ecc[i]))*np.tan(fsecondary/2.)) #eccentric anomaly at secondary eclipse
                    timesince=parstruc['Per'+str(i+1)]/(2.*np.pi)*(Esecondary-ecc[i]*np.sin(Esecondary))
                    modelstruc['aors'], modelstruc['rprs'], modelstruc['secondary'] = aors[i]*(1.+1./parstruc['rprs'+str(i+1)]), 1./parstruc['rprs'+str(i+1)], timesince #SECONDARY eclipse, so SECONDARY in background, R*=R2, Rp=R1
                    #if modelstruc['photmodflag'] == 'jktebop': modelstruc['fluxrat']=parstruc['fluxrat'+str(j+1)]
                    rawmodel2=photmodel(modelstruc)
                    rawmodel+=1./fluxrat #this is the PRIMARY eclipse--so still see the SECONDARY
                    rawmodel/=1.+1./fluxrat
                    rawmodel2+=fluxrat #this is the SECONDARY eclipse--so still see the PRIMARY
                    rawmodel2/=1.+fluxrat
                    rawmodel-=1.
                    rawmodel2-=1.
                    rawmodel+=rawmodel2+1.
                    #reset to the primary
                    modelstruc['aors'], modelstruc['rprs'], modelstruc['secondary'] = aors[i]*(1.+parstruc['rprs'+str(i+1)]), parstruc['rprs'+str(i+1)], 0.0 #PRIMARY eclipse, so PRIMARY in background, R*=R1, Rp=R2
                    modelstruc['omega']+=180.
                if args.dilution and modelstruc['photmodflag'] != 'jktebop':
                    if any('dilution'+str(j+1) in s for s in inposindex) and modelstruc['photmodflag'] == 'batman':
                        rawmodel+= 10.**(-0.4*parstruc['dilution'+str(j+1)])
                        rawmodel/= (1.+10.**(-0.4*parstruc['dilution'+str(j+1)]))
                rawmodel-=1.0
                if args.time: print('model called ',temptime)
                if args.plotbest and args.fullcurve:
                    if buh == 1:
                        orlength=len(closefilter[0])
                        print(orlength,'ooom')
                        model[closefilter]=rawmodel[0:orlength]
                        model=np.append(model,rawmodel[orlength:])
                        nexpp+=100
                        data['ptime']=np.append(data['ptime'],np.zeros(100)+np.mean(data['ptime'][closefilter]))
                        data['pflux']=np.append(data['pflux'],np.zeros(100))
                        data['perror']=np.append(data['perror'],np.zeros(100))
                        data['pdataset']=np.append(data['pdataset'],np.zeros(100)-1)
                        data['pfilter']=np.append(data['pfilter'],np.zeros(100)+j)
                else:
                    model[closefilter]+=rawmodel
                #np.save('model'+j+1'.npy',data['pdataset'])
                #np.save('fuh.npy',data['pfilter'])

                if args.gp:
                    if any('gpmodtypep' in s for s in data['gpmodtype']):
                        if data['gpmodtype']['gpmodtypep'] == 'Matern32':
                            pkern=gppack.kernels.Matern32Kernel(parstruc['gpppartau']**2)*parstruc['gppparamp']**2
                        elif data['gpmodtype']['gpmodtypep'] == 'Cosine':
                            pkern=gppack.kernels.CosineKernel(parstruc['gppparP'])*parstruc['gppparamp']**2
                        elif data['gpmodtype']['gpmodtypep'] == 'ExpSine2':
                            pkern=gppack.kernels.ExpSine2Kernel(parstruc['gpppartau'],parstruc['gppparP'])*parstruc['gppparamp']**2
                        elif data['gpmodtype']['gpmodtypep'] == 'Haywood14QP':
                            if instruc['gppackflag'] == 'celerite':
                                pkern=celeritekernel(np.log(parstruc['gppparamp']**2),np.log(parstruc['gppparGamma']),np.log(1./np.sqrt(2.)/parstruc['gpppartau']),np.log(parstruc['gppparP']*2.))
                            else:
                                pkern1=gppack.kernels.ExpSine2Kernel(parstruc['gppparGamma'],parstruc['gppparP'])
                                pkern2=gppack.kernels.ExpSquaredKernel(parstruc['gpppartau'])
                                pkern=pkern1*pkern2*parstruc['gppparamp']**2
                        elif data['gpmodtype']['gpmodtypep'] == 'SHO':
                            Q = parstruc['gppparLogQ1']
                            P = np.exp(parstruc['gppparLogP'])
                            log_omega0 = np.log(4*np.pi*Q) - np.log(P) - 0.5*np.log(4.0*Q*Q-1.0)
                            log_S0 = parstruc['gppparLogamp'] - log_omega0 - parstruc['gppparLogQ1']
                            pkern1=celeritekernel(
                                log_S0=log_S0, ## amplitude of the main peak
                                log_Q=parstruc['gppparLogQ1'], ## decay timescale of the main peak (width of the spike in the FT)
                                log_omega0=log_omega0) ## period
                            #pkern3 = terms.JitterTerm(log_sigma=parstruc['gppparLogSigma'])
                            pkern = pkern1 #+ pkern3
                        elif data['gpmodtype']['gpmodtypep'] == 'SHOMixture':
                            pkern1=celeritekernel(
                                log_a=parstruc['gppparLogamp'], ## amplitude of the main peak
                                log_Q1=parstruc['gppparLogQ1'], ## decay timescale of the main peak (width of the spike in the FT)
                                mix_par=parstruc['gppparmix'], ## height of second peak relative to first peak
                                log_Q2=parstruc['gppparLogQ2'], ## decay timescale of the second peak
                                log_P=parstruc['gppparLogP']) ## period
                            #pkern3 = terms.JitterTerm(log_sigma=parstruc['gppparLogSigma'])
                            pkern = pkern1 #+ pkern3
                        elif data['gpmodtype']['gpmodtypep'] == 'SHOMixture_sys':
                            pkern1=celeritekernel(
                                log_a=parstruc['gppparLogamp'], ## amplitude of the main peak
                                log_Q1=parstruc['gppparLogQ1'], ## decay timescale of the main peak (width of the spike in the FT)
                                mix_par=parstruc['gppparmix'], ## height of second peak relative to first peak
                                log_Q2=parstruc['gppparLogQ2'], ## decay timescale of the second peak
                                log_P=parstruc['gppparLogP']) ## period
                            pkern2 = terms.SHOTerm(
                                log_S0=parstruc['gppparLogS0'],
                                log_Q=-np.log(4.0),
                                log_omega0=parstruc['gppparLogOmega'])
                            pkern2.freeze_parameter('log_Q')
                            pkern3 = terms.JitterTerm(log_sigma=parstruc['gppparLogSigma'])
                            pkern = pkern1 + pkern2 + pkern3
                            
                        
                        #gp=gppack.GP(pkern)
                        useforgp=np.where(data['gppuse'] == 1)
                        notforgp=np.where(data['gppuse'] == 0)
                        useforgp, notforgp = useforgp[0], notforgp[0]
                        useforgp = useforgp[np.argsort(data['ptime'][useforgp])] ## Ellie sort this
                        gp=gppack.GP(pkern,  mean=np.median(data['pflux'][useforgp]-1))
                        gp.compute(np.array(data['ptime'][useforgp]),np.array(data['perror'][useforgp]))
                
                if args.plotstep or args.plotbest:
                    #pl.plot(np.array(data['ptime'], dtype=float),np.array(data['pflux'], dtype=float),'ro')
                    #pl.plot(np.array(data['ptime'], dtype=float),model+1.)
                    #pl.show()
                    #pl.clf()
                    #pl.plot(np.array(data['ptime'], dtype=float),np.array(data['pflux'], dtype=float)-(model+1.),'ro')
                    #pl.show()
                    #pl.clf()
                    ##sys.exit()
                    setsfilter=np.array(list(set(data['pdataset'][np.where(data['pfilter'] == j+1)])))
                    nsetsfilter=len(setsfilter)
                    for k in range(0,nsetsfilter):
                        closedataset=np.where((np.abs(phased) <= (dur[i]/2.+maxexp)*1.25) & (data['pdataset'] == setsfilter[k]))
                        if len(closedataset[0]) > 0:
                            if args.plotstep:
                                fignum=1
                            else:
                                fignum=nfigs
                                nfigs+=1
                            pl.plot(phased[closedataset], np.array(data['pflux'][closedataset], dtype=float), 'ro')
                            pl.plot(phased[closedataset], model[closedataset]+1.0, 'bo')
                            if args.gp:
                                if any('gpmodtypep' in s for s in data['gpmodtype']):

                                    if False:
                                        try:
                                            burnin = struc1['burnin']
                                        except:
                                            burnin=chainin.shape[1]//5
                                            
                                        t_lin = np.linspace(data['ptime'][0],data['ptime'][-1],len(data['ptime']))
                                        ndimensions = chainin.shape[2]
                                        tmp = chainin[:, burnin:, :].reshape((-1, ndimensions))
                                        for st in tmp[np.random.randint(len(tmp), size=3)]:
                                            s2 = [np.log(st[np.array([i for i,s in enumerate(inposindex) if s == 'gppparamp'])]**2),np.log(st[np.array([i for i,s in enumerate(inposindex) if s == 'gppparGamma'])]),np.log(1./np.sqrt(2.)/(st[np.array([i for i,s in enumerate(inposindex) if s == 'gpppartau'])])),np.log(st[np.array([i for i,s in enumerate(inposindex) if s == 'gppparP'])]*2)]
                                            gp.set_parameter_vector(np.array(s2).flatten())
                                            mu = gp.predict(np.array(data['pflux'][useforgp])-(model[useforgp]+1.),t_lin, return_cov=False)
                                            #pl.plot(t_lin, mu,  alpha=0.3)
                                            #pl.show()
                                            
                                    #gpmodel=gp.sample_conditional(np.array(data['pflux'][useforgp])-(model[useforgp]+1.),np.array(data['ptime'][useforgp]), size=1)+(model[useforgp]+1.)
                                    print("predicting GP on data")
                                    temp = np.where((np.abs(phased) <= (1.5*dur[i])*2) & (data['pdataset'] == setsfilter[k]))[0]

                                    print(len(temp),temp[0],temp[-1],data['ptime'][temp[0]], data['ptime'][temp[-1]])
                                    if len(temp)>5000: #use this for Spitzer since so much data...
                                        bins = np.linspace(data['ptime'][temp[0]], data['ptime'][temp[-1]], 2000) 
                                        t_means = np.histogram(data['ptime'][temp], bins, weights=data['ptime'][temp])[0] / np.histogram(data['ptime'][temp], bins)[0]
                                        model_means = np.histogram(data['ptime'][temp], bins, weights=model[temp])[0] / np.histogram(data['ptime'][temp], bins)[0]
                                        y_means = np.histogram(data['ptime'][temp], bins, weights=data['pflux'][temp])[0] / np.histogram(data['ptime'][temp], bins)[0]
                                    else:
                                        t_means=data['ptime'][temp]
                                        model_means=model[temp]
                                        y_means=data['pflux'][temp]
                                        
 
                                    mu, var = gp.predict(np.array(data['pflux'][useforgp])-(model[useforgp]+1.), t_means, return_var=True)
                                    #mu = mu_i + (model[useforgp]+1.)
                                    
                                    print("predicitng GP on lin", k, len(data['ptime'][data['pdataset'] == setsfilter[k]]),data['ptime'][data['pdataset'] == setsfilter[k]][0])
                                    t_lin = np.linspace(data['ptime'][data['pdataset'] == setsfilter[k]][0],data['ptime'][data['pdataset'] == setsfilter[k]][-1],1000)
                                    mu_lin, var_lin = gp.predict(np.array(data['pflux'][useforgp])-(model[useforgp]+1.), t_lin, return_var=True)

                                    closedatasetg=np.where((np.abs(phased) <= (dur[i]/2.+maxexp)*2) & (data['pdataset'] == setsfilter[k]) & (data['gppuse'] == 1))
                                    #pl.plot(phased[closedatasetg], mu[closedatasetg], 'go')
                                    #pl.draw()
                                    #pl.clf()
                                    #pl.plot(np.array(data['ptime'][useforgp]),np.array(data['pflux'][useforgp]),'ro')
                                    #pl.show()
                                    #pl.clf()
                                    #pl.plot(np.array(data['ptime'][useforgp]),np.array(data['pflux'][useforgp])-gpmodel,'ro')
                                    #pl.show()
                                    #pl.clf()
                                    #sys.exit()

                                    if any('picklefile' in s for s in index):
                                        print("Writing Ellie's pickle file ln 1059", i, j, k)
                                        mydict[str(i)][str(j)][str(k)] = {}
                                        mydict[str(i)][str(j)][str(k)]['y'] = data['pflux'][data['pdataset'] == setsfilter[k]]
                                        mydict[str(i)][str(j)][str(k)]['y_gp'] = data['pflux'][data['pdataset'] == setsfilter[k]]-(model[data['pdataset'] == setsfilter[k]]+1.)
                                        mydict[str(i)][str(j)][str(k)]['t'] = np.array(data['ptime'][data['pdataset'] == setsfilter[k]])
                                        mydict[str(i)][str(j)][str(k)]['phase'] = phased[data['pdataset'] == setsfilter[k]]
                                        mydict[str(i)][str(j)][str(k)]['transit_model'] = model[data['pdataset'] == setsfilter[k]]+1.
                                        #mydict['gpmodel'] = gpmodel

                                        # these are for the transits
                                        mydict[str(i)][str(j)][str(k)]['t_bins'] = t_means
                                        mydict[str(i)][str(j)][str(k)]['y_bins'] = y_means
                                        mydict[str(i)][str(j)][str(k)]['model_bins'] = model_means
                                        mydict[str(i)][str(j)][str(k)]['mu_bins'] = mu
                                        mydict[str(i)][str(j)][str(k)]['var_bins'] = var

                                        # these are for the overall lightcurve
                                        mydict[str(i)][str(j)][str(k)]['t_lin'] = t_lin
                                        mydict[str(i)][str(j)][str(k)]['mu_lin'] = mu_lin
                                        mydict[str(i)][str(j)][str(k)]['var_lin'] = var_lin
                                    
                                                            
                            print(len(closedataset))
                            #pl.ylim([1.-(parstruc['rprs'+str(i+1)]**2)*1.5,1.+(parstruc['rprs'+str(i+1)]**2)*1.5])
                            if args.plotstep:
                                pl.draw()
                                pl.pause(0.01)
                                pl.clf()
                            if args.plotbest:
                                if args.ploterrors: pl.errorbar(phased[closedataset], np.array(data['pflux'][closedataset], dtype=float), yerr=np.array(data['perror'][closedataset], dtype=float),fmt='none',ecolor='red')
                                #if j == 0: pl.xlim([-0.1,0.1])#1.9,2.3])
                                pl.xlabel('time from center of transit (days)')
                                pl.ylabel('normalized flux')
                                namelength=len(instruc['plotfile'])
                                if instruc['plotfile'][namelength-4:namelength] == '.pdf':
                                    pl.savefig(pp, format='pdf')
                                    #pl.draw()
                                    #pl.pause(1)
                                    pl.clf()
                                    if args.plotresids:
                                        pl.plot(phased[closedataset], np.array(data['pflux'][closedataset]-(model[closedataset]+1.0)), 'ro')
                                        if args.ploterrors: pl.errorbar(phased[closedataset], np.array(data['pflux'][closedataset]-(model[closedataset]+1.0)), yerr=np.array(data['perror'][closedataset], dtype=float),fmt='none',ecolor='red')
                                        pl.plot([np.min(phased[closedataset]),np.max(phased[closedataset])],[0.0,0.0],color='blue')
                                        if j == 0: pl.xlim([-0.1,0.1])#1.9,2.3])#
                                        pl.savefig(pp, format='pdf')
                                        pl.clf()
                    #if args.plotbest:
                                else:
                                    pl.savefig(instruc['plotfile'], format=instruc['plotfile'][namelength-3:namelength])
                                    print('Plot complete. If this is a multiplanet system and you want') 
                                    print('more than the first planet, you must use PDF format.')
                                    #sys.exit()
                                if i == nplanets-1 and j == instruc['pnfilters']-1 and k == nsetsfilter-1 and (not args.rvs and not args.tomography or args.binary): 
                                    print('Plots complete.')
                                    if (not args.rvs and not args.tomography and not args.line and not args.rm) or args.binary: pp.close()

                                    #sys.exit()
        
        if args.plotbest:
            print("Actually writing it")
            print(mydict['0']['0'].keys())
            pickle.dump(mydict, open(struc1['picklefile'],'wb'))
                                
                
                #if args.time: print 'ending the filter loop ',temptime
            #if args.time: print 'ending the loop for planet ',i+1,' at ',temptime

        model+=1.0
        if args.plotbest: 
            f=open('photmodel.txt','w')
            for i in range (0,nexpp):
                f.write(str(data['ptime'][i])+', '+str(phased[i])+', '+str(data['pflux'][i])+', '+str(data['perror'][i])+', '+str(model[i])+', '+str(data['pflux'][i]-model[i])+', '+str(int(data['pdataset'][i]))+' \n')
            f.close()
            
        inv_sigma2 = 1.0/data['perror']**2
        if not args.gp:
            lnl+=np.sum(((data['pflux']-model)**2)*inv_sigma2 - np.log(inv_sigma2))
        else:
            if any('gpmodtypep' in s for s in data['gpmodtype']):
                #gpmodel=gp.sample_conditional(np.array(data['pflux'])-(model),np.array(data['ptime']))+(model)
                #lnl1=np.sum(((data['pflux']-gpmodel)**2)*inv_sigma2 - np.log(inv_sigma2))
                #Kalpha=np.matrix(gp.get_matrix(data['ptime']))
                #Kalphai=Kalpha.I
                #resids=np.matrix(data['pflux']-model)
                #temp11=np.linalg.slogdet(Kalpha)
                #lnl1=-1./2.*(resids*Kalphai*resids.T)-1./2.*temp11[0]*temp11[1]-len(data['ptime'])/2.*np.log(2.*np.pi)
                if struc1['gppackflag'] == 'celerite':
                    lnl2=gp.log_likelihood(np.array(data['pflux'][useforgp])-model[useforgp])
                else:
                    lnl2=gp.lnlikelihood(np.array(data['pflux'][useforgp])-model[useforgp])
                lnl-=2.*lnl2   #gp.lnlike gives the actual lnlike, but needs to be *'d to add in with the other types      
                
                if len(notforgp) > 1:
                    lnl+=np.sum(((data['pflux'][notforgp]-model[notforgp])**2)*inv_sigma2[notforgp] - np.log(inv_sigma2[notforgp]))
                #TEMPORARY
                #f=open('lnlout.txt','a')
               
                #f.write(str(lnl1)+', '+str(lnl2)+' \n')
                #f.close()
                #print lnl1,lnl2
            else:
                lnl+=np.sum(((data['pflux']-model)**2)*inv_sigma2 - np.log(inv_sigma2))
        if args.getprob:
            print('The total photometric chisq is ',np.sum(((data['pflux']-model)**2)*inv_sigma2),' and chisq_red=',np.sum(((data['pflux']-model)**2)*inv_sigma2)/len(data['pflux']))
            for i in range(0,instruc['pndatasets']):
                heres=np.where(data['pdataset'] == i+1)
                print('The contribution of photometric dataset ',i+1,' to lnprob is ',np.sum(((data['pflux'][heres]-model[heres])**2)*inv_sigma2[heres] - np.log(inv_sigma2[heres])), ' or chisq=',np.sum(((data['pflux'][heres]-model[heres])**2)*inv_sigma2[heres])/len(heres[0]))
    #if args.time: print 'ending the planet loop ',temptime

    if args.tomography:
        ntomsets=data['tomdict']['ntomsets']
        dur=np.array(dur,dtype=np.float)
        for i in range(0,ntomsets):
            whichplanet=int(data['tomdict']['whichplanet'+str(i+1)])
            tomphase=np.mod(data['tomdict']['ttime'+str(i+1)]-parstruc['epoch'+str(whichplanet)], parstruc['Per'+str(whichplanet)])
            highphase=np.where(tomphase >= parstruc['Per'+str(whichplanet)]/2.)
            tomphase[highphase]-=parstruc['Per'+str(whichplanet)]
            tomphase*=24.0*60.0 #to minutes
            texptime=data['tomdict']['texptime'+str(i+1)]/60.0 #to minutes
            if not any('tomdrift' in s for s in inposindex) and not args.spots:
                tomins=np.where(np.abs(tomphase) <= dur[whichplanet-1]*24.*60.*1.25)
                tomouts=np.where(np.abs(tomphase) > dur[whichplanet-1]*24.*60.*1.25)
            else:
                tomins=np.isfinite(tomphase)
                tomouts=0
            horusstruc = {'vsini': parstruc['vsini'], 'sysname': instruc['sysname'], 'obs': instruc['obs'+str(i+1)], 'vabsfine': data['tomdict']['vabsfine'+str(i+1)], 'Pd': parstruc['Per'+str(whichplanet)], 'lambda': parstruc['lambda'+str(whichplanet)], 'b': parstruc['bpar'+str(whichplanet)], 'rplanet': np.array(parstruc['rprs'+str(whichplanet)]), 't': tomphase[tomins], 'times': texptime[tomins], 'e': ecc[whichplanet-1], 'periarg': omega[whichplanet-1]*np.pi/180., 'a': aors[whichplanet-1]}
            
            if any('linecenter' in s for s in inposindex): 
                if np.abs(parstruc['linecenter'+str(i+1)]) > np.max(np.abs(horusstruc['vabsfine'])): return -np.inf
                horusstruc['vabsfine']+=parstruc['linecenter'+str(i+1)]
            if any('q1t' in s for s in inposindex) and any('q2t' in s for s in inposindex):
                horusstruc['gamma1'], horusstruc['gamma2'] = 2.0*parstruc['q2t']*np.sqrt(parstruc['q1t']), np.sqrt(parstruc['q1t'])*(1.0-2.0*parstruc['q2t'])
            else:
                horusstruc['gamma1'], horusstruc['gamma2'] = parstruc['g1t'], parstruc['g2t']
            if any('intwidth' in s for s in inposindex):
                horusstruc['width'] = parstruc['intwidth']
            elif any('intwidth' in s for s in index):
                horusstruc['width'] = np.float(instruc['intwidth'])
            else:
                horusstruc['width'] = 10.

            if any('macroturb' in s for s in inposindex):
                horusstruc['zeta'] = parstruc['macroturb']
                domacroturb='y'
            else:
                domacroturb='n'

            if args.spots:
                horusstruc['nspots'] = data['nspots']
                horusstruc['t0'] = theta[[i for i, s in enumerate(inposindex) if 'spott0' in s]]
                horusstruc['rplanet'] = np.append(horusstruc['rplanet'],theta[[i for i, s in enumerate(inposindex) if 'spotrad' in s]])
                horusstruc['Prot'] = parstruc['Prot']
                resnum=100
            else:
                resnum=50

            if any('tomdrift' in s for s in inposindex):
                lineshift=(data['tomdict']['ttime'+str(i+1)]-data['tomdict']['ttime'+str(i+1)][0])*parstruc['tomdriftl'+str(i+1)]+parstruc['tomdriftc'+str(i+1)]
                if np.max(np.abs(lineshift) > np.max(np.abs(horusstruc['vabsfine']))): return -np.inf
                horusstruc['lineshifts'] = lineshift
                outstruc = horus.model(horusstruc, resnum=resnum,lineshifts='y',macroturb=domacroturb,starspot=args.spots)

            elif args.binary:
                horusstruc['a'] = aors[i]*(1.+parstruc['rprs'+str(i+1)])
                if ecc[0] == 0.:
                    lineshift1=parstruc['semiamp'+str(i+1)+'a']*np.sin((data['rtime']-parstruc['epoch'+str(i+1)]+parstruc['Per'+str(i+1)]/2.)*2.*np.pi/parstruc['Per'+str(i+1)]) #Per/2 is because transit will occur on \ part of sine curve, not / part
                    lineshift2=parstruc['semiamp'+str(i+1)+'b']*np.sin((data['rtime']-parstruc['epoch'+str(i+1)])*2.*np.pi/parstruc['Per'+str(i+1)])
                    secondaryphase=0.5
                else:
                    ftransit=np.pi/2.-omega[i]*np.pi/180.#-np.pi #true anomaly at transit
                    Etransit=2.*np.arctan(np.sqrt((1.-ecc[i])/(1.+ecc[i]))*np.tan(ftransit/2.)) #eccentric anomaly at transit
                    timesince=parstruc['Per'+str(i+1)]/(2.*np.pi)*(Etransit-ecc[i]*np.sin(Etransit)) #time since periastron to transit

                    fsecondary=np.pi/2.-omega[i]*np.pi/180.-np.pi #true anomaly at secondary eclipse
                    Esecondary=2.*np.arctan(np.sqrt((1.-ecc[i])/(1.+ecc[i]))*np.tan(fsecondary/2.)) #eccentric anomaly at secondary eclipse
                    timesince2=parstruc['Per'+str(i+1)]/(2.*np.pi)*(Esecondary-ecc[i]*np.sin(Esecondary))
                    secondaryphase=np.abs((timesince2-timesince)/parstruc['Per'+str(i+1)])

                    lineshift1=radvel.kepler.rv_drive(data['rtime'],np.array([parstruc['Per'+str(i+1)],parstruc['epoch'+str(i+1)]-timesince,ecc[i],omega[i]*np.pi/180.,parstruc['semiamp'+str(i+1)+'a']]))
                    lineshift2=radvel.kepler.rv_drive(data['rtime'],np.array([parstruc['Per'+str(i+1)],parstruc['epoch'+str(i+1)]-timesince,ecc[i],omega[i]*np.pi/180.-np.pi,parstruc['semiamp'+str(i+1)+'b']]))

                
                #get phases
                primaries=(np.abs(tomphase) <= np.abs(tomphase-secondaryphase)) | (np.abs(tomphase) <= np.abs(tomphase-(secondaryphase-1.)))
                secondaries=(np.abs(tomphase) > np.abs(tomphase-secondaryphase)) & (np.abs(tomphase) > np.abs(tomphase-(secondaryphase-1.)))
                #primaries, secondaries = primaries[0], secondaries[0]
                #nprimary, nsecondary = len(primaries), len(secondaries)
                #if nprimary > 1 and nsecondary > 1:
                lineshifts, lineshiftsforeground = lineshift1[:], lineshift2[:]
                lineshifts[secondaries], lineshiftsforeground[primaries] = lineshift2[secondaries], lineshift1[primaries]
                goodprimary, goodsecondary = len(np.where(primaries == True)), len(np.where(secondaries == True))
                profarr1=np.zeros((len(tomphase),len(vabsfine)))
                #first do the primary eclipse
                if goodprimary > 1:
                    #horusstruc = {'vsini': parstruc['vsini'], 'sysname': instruc['sysname'], 'obs': instruc['obs'+str(i+1)], 'vabsfine': data['tomdict']['vabsfine'+str(i+1)], 'Pd': parstruc['Per'+str(whichplanet)], 'lambda': parstruc['lambda'+str(whichplanet)], 'b': parstruc['bpar'+str(whichplanet)], 'rplanet': parstruc['rprs'+str(whichplanet)], 't': tomphase[tomins], 'times': texptime[tomins], 'e': ecc[whichplanet-1], 'periarg': omega[whichplanet-1]*np.pi/180., 'a': aors[whichplanet-1]}
                    horusstruc['lineshifts']=lineshifts[primaries]
                    horusstruc['lineshifts'], horusstruc['t'], horusstruc['times']=lineshifts[primaries], tomphase[primaries], texptime[primaries]
                    outstruc1 = horus.model(horusstruc, resnum=resnum,lineshifts='y',macroturb=domacroturb,starspot=args.spots)
                    horusstruc['vsini']=parstruc['vsinib']
                    profarr11, basearr1, baseline1 = outstruc1['profarr'], outstruc1['baseline'], outstruc1['basearr']
                    horusstruc['lineshifts']=lineshiftsforegrounds[primaries]
                    outstruc2 = horus.model(horusstruc, resnum=resnum, onespec='y',convol='y',starspot=args.spots)
                    profarr2, basearr2, baseline2 = outstruc2['profarr'], outstruc2['baseline'], outstruc2['basearr']
                    profarr11=(profarr1+profarr2*(np.sum(baseline1)/np.sum(baseline2))*parstruc['fluxratt'])
                    profarr1[primaries,:]=profarr11/np.max(profarr11) #I hope this is right...
                #and then do the secondary
                if goodsecondary > 1:
                    horusstruc['vsini'], horusstruc['rplanet'], horusstruc['periarg'], horusstruc['lambda'] = parstruc['vsinib'], 1./parstruc['rprs'+str(whichplanet)], omega[whichplanet-1]*np.pi/180.-np.pi, parstruc['lambda'+str(whichplanet+1)]
                    horusstruc['lineshifts'], horusstruc['t'], horusstruc['times']=lineshifts[secondaries], tomphase[secondaries], texptime[secondaries]
                    outstruc1 = horus.model(horusstruc, resnum=resnum,lineshifts='y',macroturb=domacroturb,starspot=args.spots)
                    horusstruc['vsini']=parstruc['vsini']
                    profarr11, basearr1, baseline1 = outstruc1['profarr'], outstruc1['baseline'], outstruc1['basearr']
                    horusstruc['lineshifts']=lineshiftsforegrounds[secondaries]
                    outstruc2 = horus.model(horusstruc, resnum=resnum, onespec='y',convol='y',starspot=args.spots)
                    profarr2, basearr2, baseline2 = outstruc2['profarr'], outstruc2['baseline'], outstruc2['basearr']
                    profarr11=(profarr1+profarr2*(np.sum(baseline1)/np.sum(baseline2))/parstruc['fluxratt'])
                    profarr1[secondaries,:]=profarr11/np.max(profarr11) #I hope this is right...
                    
            else:
                outstruc = horus.model(horusstruc, resnum=resnum,macroturb=domacroturb,starspot=args.spots)
            profarr1, baseline, basearr = outstruc['profarr'], outstruc['baseline'], outstruc['basearr']
            #np.save('horustestimage.npy',outstruc['imarr'])
            tnexp = len(data['tomdict']['ttime'+str(i+1)])
            nvabsfine=len(data['tomdict']['vabsfine'+str(i+1)])
            vabsfine=data['tomdict']['vabsfine'+str(i+1)]
            model=np.zeros((tnexp, nvabsfine))
            model[tomins,:]=profarr1
            if tomouts and not args.binary:
                if  len(tomins[0]) == 0:
                    return -np.inf #protects against the model transit not happening during the observations, but we don't need this for binaries where we just fit the full line profile over the whole range of phases
                for j in range (0, len(tomouts[0])): model[tomouts[0][j],:]=outstruc['basearr'][0,:]
            if args.skyline:
                skyline=horus.mkskyline(vabsfine,parstruc['skydepth'],parstruc['skycen'],instruc['obs'+str(i+1)])
                for j in range (np.int(instruc['skyfirst']), np.int(instruc['skylast'])+1):
                    model[j,:]+=skyline

            vgood=np.where(np.abs(vabsfine) <= np.float(instruc['vsini'])*1.25)

            if args.gp:
                lnl1=0.
                if any('gpmodtypet' in s for s in data['gpmodtype']):
                    if data['gpmodtype']['gpmodtypet'] == 'Matern32':
                        #tkernt=george.kernels.Matern32Kernel(parstruc['gptpartaut'],dim=0)
                        #tkernv=george.kernels.Matern32Kernel(parstruc['gptpartauv'],dim=1)
                        tkern=parstruc['gptparamp']*george.kernels.Matern32Kernel([parstruc['gptpartaut'],parstruc['gptpartauv']],ndim=2)#tkernt+tkernv
                #elif 'gpmodtypep' == 'Cosine':
                    #pkern=george.kernels.CosineKernel(parstruc['gppparP'])
                    gp=george.GP(tkern)
                    bjdfloor=np.floor(data['tomdict']['ttime'+str(i+1)])
                    ntnights=len(np.unique(bjdfloor))
                    for night in range (0,ntnights):
                        thisnight=np.where(bjdfloor == np.unique(bjdfloor)[night])
                        ttemp, vtemp = np.meshgrid(data['tomdict']['ttime'+str(i+1)][thisnight], data['tomdict']['vabsfine'+str(i+1)], indexing="ij")
                        profarrinds=np.vstack((ttemp.flatten(), vtemp.flatten())).T
                        profarrforgp=data['tomdict']['profarr'+str(i+1)][thisnight,:].flatten()#.T
                        profarrerrforgp=data['tomdict']['profarrerr'+str(i+1)][thisnight,:].flatten()#.T
                        modelforgp=model[thisnight,:].flatten()#.T
                        gp.compute(profarrinds,profarrerrforgp)

                        wherebad=np.array(np.where(np.isnan(modelforgp)))
                        if wherebad.size != 0:
                            return -np.inf
                        lnl1=-2.*gp.lnlikelihood(profarrforgp-modelforgp*(-1.))

            if args.plotstep:
                fignum=1
                for j in range (0, tnexp):
                    pl.plot(data['tomdict']['vabsfine'+str(i+1)],data['tomdict']['profarr'+str(i+1)][j,:]+0.1*j)
                    if instruc['fitflat'] != 'True':
                        pl.plot(data['tomdict']['vabsfine'+str(i+1)],model[j,:]*(-1.)+0.1*j)#
                    else:
                        pl.plot(data['tomdict']['vabsfine'+str(i+1)],(model[j,:]-outstruc['basearr'][0,:])*(-1.)+0.1*j)#
                #pl.imshow(profarr1,figure=fignum)
                pl.savefig('active_misttborn.eps',format='eps')
                pl.draw()
                pl.pause(1.)
                pl.clf()
            if args.plotbest:
                import dtutils
                fignum=1
                profarrflat=np.zeros((tnexp,nvabsfine))
                modelflat=np.zeros((tnexp,nvabsfine))
                for j in range (0, tnexp):
                    #pl.figure(fignum)
                    pl.plot(data['tomdict']['vabsfine'+str(i+1)],data['tomdict']['profarr'+str(i+1)][j,:]+0.1*j)
                    pl.plot(data['tomdict']['vabsfine'+str(i+1)],model[j,:]*(-1.)+0.1*j)#
                    if instruc['fitflat'] == 'False': 
                        profarrflat[j,:]=data['tomdict']['profarr'+str(i+1)][j,:]-outstruc['basearr'][0,:]*(-1.)
                        modelflat[j,:]=model[j,:]+outstruc['basearr'][0,:]*(-1.)
                    else:
                        profarrflat[j,:]=model[j,:]
                #pl.imshow(profarr1,figure=fignum)
                pl.xlabel('velocity (km s$^{-1}$)')
                pl.ylabel('normalized profile + offset')
                pl.savefig(pp, format='pdf')
                pl.clf()
                pl.plot(data['tomdict']['vabsfine'+str(i+1)],model[0,:]*(-1.),color='red')
                pl.plot(data['tomdict']['vabsfine'+str(i+1)],basearr[0,:]*(-1.),color='blue')
                pl.savefig(pp, format='pdf')
                pl.clf()
                #pl.imshow(profarrflat)
                tphase, tphase2=np.mod((data['tomdict']['ttime'+str(i+1)]-parstruc['epoch'+str(whichplanet)]),parstruc['Per'+str(whichplanet)]),np.mod((data['tomdict']['ttime'+str(i+1)]+data['tomdict']['texptime'+str(i+1)]/(60.*60.*24.)-parstruc['epoch'+str(whichplanet)]),parstruc['Per'+str(whichplanet)])
                bads=np.where(tphase > parstruc['Per'+str(whichplanet)]/2.)
                tphase[bads[0]]-=parstruc['Per'+str(whichplanet)]
                bads=np.where(tphase2 > parstruc['Per'+str(whichplanet)]/2.)
                tphase2[bads[0]]-=parstruc['Per'+str(whichplanet)]
                tphase += dur[whichplanet-1]/2.
                tphase2 += dur[whichplanet-1]/2.
                tphase /= dur[whichplanet-1]
                tphase2 /= dur[whichplanet-1]
                dtutils.mktslprplot(profarrflat*(-1.),data['tomdict']['profarrerr'+str(i+1)],data['tomdict']['vabsfine'+str(i+1)]-parstruc['tomdriftc'+str(i+1)],tphase,tphase2,parstruc['vsini'],parstruc['bpar'+str(whichplanet)],parstruc['rprs'+str(whichplanet)],filename='none')
                pl.savefig(pp, format='pdf')
                pl.clf()
                #pl.imshow(modelflat*(-1.))
                dtutils.mktslprplot(modelflat,data['tomdict']['profarrerr'+str(i+1)],data['tomdict']['vabsfine'+str(i+1)]-parstruc['tomdriftc'+str(i+1)],tphase,tphase2,parstruc['vsini'],parstruc['bpar'+str(whichplanet)],parstruc['rprs'+str(whichplanet)],filename='none',weighted=False)
                pl.savefig(pp, format='pdf')
                pl.clf()
                #pl.imshow(data['tomdict']['profarr'+str(i+1)]-model*(-1.))
                dtutils.mktslprplot((data['tomdict']['profarr'+str(i+1)]-model*(-1.))*(-1.),data['tomdict']['profarrerr'+str(i+1)]-parstruc['tomdriftc'+str(i+1)],data['tomdict']['vabsfine'+str(i+1)],tphase,tphase2,parstruc['vsini'],parstruc['bpar'+str(whichplanet)],parstruc['rprs'+str(whichplanet)],filename='none')
                pl.savefig(pp, format='pdf')
                pl.clf()
                if args.gp and any('gptpar' in s for s in gpparnames):
                    #compute the GP model
                    ttemp, vtemp2 = np.meshgrid(data['tomdict']['ttime'+str(i+1)], data['tomdict']['vabsfine'+str(i+1)], indexing="ij")
                    profarrinds2=np.vstack((ttemp.flatten(), vtemp2.flatten())).T
                    modelforgp2=model.flatten()#.T
                    modelGP1d1=gp.sample_conditional(profarrforgp-modelforgp*(-1.), profarrinds2)
                    modelGP1d=modelGP1d1+modelforgp2*(-1.)
                    #now need to stack it back up into a 2d array
                    modelGP2d=np.zeros((tnexp,nvabsfine))
                    modelGP2d1=np.zeros((tnexp,nvabsfine))
                    for ecount in range(0,tnexp):
                        modelGP2d[ecount,:]=modelGP1d[ecount*nvabsfine:(ecount+1)*nvabsfine]
                        modelGP2d1[ecount,:]=modelGP1d1[ecount*nvabsfine:(ecount+1)*nvabsfine]
                        pl.plot(data['tomdict']['vabsfine'+str(i+1)],profarr[ecount,:]+0.1*ecount)
                        pl.plot(data['tomdict']['vabsfine'+str(i+1)],modelGP2d[ecount,:]+0.1*ecount)#
                        pl.plot(data['tomdict']['vabsfine'+str(i+1)],model[ecount,:]*(-1.)+0.1*ecount)
                        modelGP2d[ecount,:]+=model[0,:]

                    pl.xlabel('velocity (km s$^{-1}$)')
                    pl.ylabel('normalized profile + offset')
                    pl.savefig(pp, format='pdf')
                    pl.clf()
                    pl.imshow(modelGP2d1)
                    pl.savefig(pp, format='pdf')
                    pl.clf()
                    pl.imshow(profarr-(model-modelGP2d1)*(-1.))
                    pl.savefig(pp, format='pdf')
                    pl.clf()
                if not args.rvs and not args.line and i == ntomsets-1: pp.close()
                #np.save(profarrflat,'misttborn_profarr.npy')
                #np.save(profarr-model*(-1.),'misttborn_prof_residuals.npy')
                f=open('lineprofile.dat','w')
                for j in range (0,nvabsfine):
                    f.write(str(outstruc['basearr'][0,j])+' \n')
                f.close()
                for j in range (0,tnexp):
                    f=open('lineprofile.'+str(i)+'.'+str(j)+'.dat','w')
                    print('lineprofile.'+str(i)+'.'+str(j)+'.dat')
                    for k in range (0,nvabsfine):
                        #f.write(str(model[j,k])+' \n')
                        f.write(str(model[j,k]-outstruc['basearr'][0,k])+' \n')
                        #if j >= np.int(instruc['skyfirst']) and j < np.int(instruc['skylast'])+1:
                            #f.write(str(profarr[j,k]-(outstruc['basearr'][0,k]+skyline[k])*(-1.))+' \n')
                        #else:
                            #f.write(str(profarr[j,k]-outstruc['basearr'][0,k]*(-1.))+' \n')
                    f.close
                
            if instruc['fitflat'] == 'True':
                for j in range (0, tnexp): model[j, : ]-=baseline
                if data['dofft']:
                    model1=horus.fourierfilt(model,data['mask'])
                    model=model1
            model*=(-1.)
            #tgood=np.where(np.abs(tomphase) <= dur*24.0*60.0) #to minutes
            inv_sigma2 = 1.0/data['tomdict']['profarrerr'+str(i+1)][:,vgood]**2
            if not args.gp:
                lnl+=np.sum((data['tomdict']['profarr'+str(i+1)][:,vgood]-model[:,vgood])**2*inv_sigma2 - np.log(inv_sigma2))
            else:
                if any('gpmodtypet' in s for s in data['gpmodtype']):
                    #print np.where(np.isnan(profarrforgp)),'prof',np.where(np.isnan(modelforgp))
                    #b/c multiply model by -1 above after make modelforgp out of it


                    #gpmodel=gp.sample_conditional(profarrforgp-modelforgp*(-1.), profarrinds)+modelforgp*(-1.)
                    #lnl2=np.sum((profarrforgp-gpmodel)**2/(profarrerrforgp)**2)
                    #print lnl1,lnl2
                    lnl+=lnl1
                else:
                    lnl+=np.sum((data['tomdict']['profarr'+str(i+1)][:,vgood]-model[:,vgood])**2*inv_sigma2 - np.log(inv_sigma2))
            if args.getprob:
                print('The tomographic chisq for dataset ',i,' is ',np.sum((data['tomdict']['profarr'+str(i+1)][:,vgood]-model[:,vgood])**2*inv_sigma2),' and chisq_red=',np.sum((data['tomdict']['profarr'+str(i+1)][:,vgood]-model[:,vgood])**2*inv_sigma2)/(len(vgood[0])*tnexp))
                print('The contribution of tomography to lnprob is ',np.sum((data['tomdict']['profarr'+str(i+1)][:,vgood]-model[:,vgood])**2*inv_sigma2 - np.log(inv_sigma2)))

    if args.line:
        if args.tomography:
            linemodel=basearr[0,:]
            
            
        else:
            horusstruc = {'vsini': parstruc['vsini'], 'sysname': instruc['sysname'], 'obs': instruc['obs1'], 'vabsfine': data['linevel']+parstruc['linecenter1']}
            if np.abs(parstruc['linecenter1']) > np.max(np.abs(data['linevel'])): return -np.inf
            #print tomphase
            if any('q1t' in s for s in inposindex) and any('q2t' in s for s in inposindex):
                horusstruc['gamma1'], horusstruc['gamma2'] = 2.0*parstruc['q2t']*np.sqrt(parstruc['q1t']), np.sqrt(parstruc['q1t'])*(1.0-2.0*parstruc['q2t'])
            else:
                horusstruc['gamma1'], horusstruc['gamma2'] = parstruc['g1t'], parstruc['g2t']
            if any('intwidth' in s for s in inposindex):
                horusstruc['width'] = parstruc['intwidth']
            elif any('intwidth' in s for s in index):
                horusstruc['width'] = np.float(instruc['intwidth'])
            else:
                horusstruc['width'] = 10.

            if horusstruc['obs'] != 'igrins':
                outstruc = horus.model(horusstruc, resnum=50.0, onespec='y',convol='y')
            else:
                outstruc = horus.model(horusstruc, resnum=50.0, onespec='y', convol='n')
            linemodel=outstruc['baseline']
            
            
        if args.plotbest or args.plotstep:
            if args.plotstep:
                fignum=1
            else:
                fignum=nfigs
                nfigs+=1
            pl.plot(np.array(data['linevel'], dtype=float), np.array(data['lineprof'], dtype=float), 'red')
            pl.plot(np.array(data['linevel'], dtype=float), linemodel, 'blue')
            if args.ploterrors: 
                pl.errorbar(np.array(data['linevel'], dtype=float), np.array(data['lineprof'], dtype=float), yerr=np.array(data['lineerr'], dtype=float),fmt='none',ecolor='red')
                     
                            
            if args.plotstep:
                pl.draw()
                pl.pause(0.5)
                pl.clf()
            if args.plotbest:
                pl.xlabel('velocity (km s$^{-1}$)')
                pl.ylabel('normalized flux')
                namelength=len(instruc['plotfile'])
                if instruc['plotfile'][namelength-4:namelength] == '.pdf':
                    pl.savefig(pp, format='pdf')
                    pl.clf()
                    if not args.rvs: pp.close()
                else:
                    pl.savefig(instruc['plotfile'], format=instruc['plotfile'][namelength-3:namelength])
                    print('Plot complete. If this is a multiplanet system and you want') 
                    print('more than the first planet, you must use PDF format.')

                f=open('lineprofile.dat','w')
                for j in range (0,len(data['linevel'])):
                    f.write(str(linemodel[j])+' \n')
                f.close()


        inv_sigma2 = 1.0/data['lineerr']**2
        lnl+=np.sum((data['lineprof']-linemodel)**2*inv_sigma2 - np.log(inv_sigma2))

        if args.getprob:
            print('The total chisq of the line data is ',np.sum(((data['lineprof']-linemodel)**2)*inv_sigma2),' and chisq_red=',np.sum(((data['lineprof']-linemodel)**2)*inv_sigma2)/len(data['lineprof']))
            print('The contribution of the line to lnprob is',np.sum((data['lineprof']-linemodel)**2*inv_sigma2 - np.log(inv_sigma2)))

    if args.rvs:
        nrvs=len(data['rtime'])
        rmodel=np.zeros(nrvs)
        if args.binary: rmodel2=np.zeros(nrvs)
        for i in range(0,nplanets):
            if ecc[i] == 0.0: #just use a sine model for the RVs
                if not args.binary:
                    rmodel+=parstruc['semiamp'+str(i+1)]*np.sin((data['rtime']-parstruc['epoch'+str(i+1)]+parstruc['Per'+str(i+1)]/2.)*2.*np.pi/parstruc['Per'+str(i+1)]) #Per/2 is because transit will occur on \ part of sine curve, not / part
                else:
                    rmodel+=parstruc['semiamp'+str(i+1)+'a']*np.sin((data['rtime']-parstruc['epoch'+str(i+1)]+parstruc['Per'+str(i+1)]/2.)*2.*np.pi/parstruc['Per'+str(i+1)]) #Per/2 is because transit will occur on \ part of sine curve, not / part
                    rmodel2+=parstruc['semiamp'+str(i+1)+'b']*np.sin((data['rtime']-parstruc['epoch'+str(i+1)])*2.*np.pi/parstruc['Per'+str(i+1)])
                
                
                

            else:
                #need to convert transit epoch to epoch of periastron
                #equations from https://exoplanetarchive.ipac.caltech.edu/docs/transit_algorithms.html#epoch_periastron
                if args.photometry:
                    ftransit=np.pi/2.-omega[i]*np.pi/180.#-np.pi #true anomaly at transit
                    Etransit=2.*np.arctan(np.sqrt((1.-ecc[i])/(1.+ecc[i]))*np.tan(ftransit/2.)) #eccentric anomaly at transit
                    timesince=parstruc['Per'+str(i+1)]/(2.*np.pi)*(Etransit-ecc[i]*np.sin(Etransit)) #time since periastron to transit
                else:
                    timesince=0.0 #use epoch of periastron if only have RVs

                if not args.binary:
                    rmodel+=radvel.kepler.rv_drive(data['rtime'],np.array([parstruc['Per'+str(i+1)],parstruc['epoch'+str(i+1)]-timesince,ecc[i],omega[i]*np.pi/180.,parstruc['semiamp'+str(i+1)]]))
                else:
                    rmodel+=radvel.kepler.rv_drive(data['rtime'],np.array([parstruc['Per'+str(i+1)],parstruc['epoch'+str(i+1)]-timesince,ecc[i],omega[i]*np.pi/180.,parstruc['semiamp'+str(i+1)+'a']]))
                    rmodel2+=radvel.kepler.rv_drive(data['rtime'],np.array([parstruc['Per'+str(i+1)],parstruc['epoch'+str(i+1)]-timesince,ecc[i],omega[i]*np.pi/180.-np.pi,parstruc['semiamp'+str(i+1)+'b']]))

        #add any RV offsets
        rndatasets=np.max(data['rdataset'])
        if args.fitjitter: jitters=np.zeros(len(data['rdataset']))
        for j in range(0,rndatasets):
            thisdataset=np.where(data['rdataset'] == j+1)
            if any('gamma' in s for s in inposindex):
                rmodel[thisdataset]+=parstruc['gamma'+str(j+1)]
                if args.binary: rmodel2[thisdataset]+=parstruc['gamma'+str(j+1)]
            else:
                rmodel[thisdataset]+=instruc['gamma'+str(j+1)]
                if args.binary: rmodel2[thisdataset]+=instruc['gamma'+str(j+1)]
            if args.fitjitter:
                jitters[thisdataset]=parstruc['jitter'+str(j+1)]

        #add any RV trend
        if any('rvtrend' in s for s in inposindex):
            rmodel+=(data['rtime']-parstruc['epoch1'])*parstruc['rvtrend']
            if any('rvtrendquad' in s for s in inposindex):
                rmodel+=(data['rtime']-parstruc['epoch1'])**2*parstruc['rvtrendquad']
            if args.binary: 
                rmodel2+=(data['rtime']-parstruc['epoch1'])*parstruc['rvtrend']
                if any('rvtrendquad' in s for s in inposindex):
                    rmodel2+=(data['rtime']-parstruc['epoch1'])**2*parstruc['rvtrendquad']
        
        if args.plotbest or args.plotstep and not args.binary:
            for j in range(0,rndatasets):
                    
                if args.binary:
                    alldatasets=np.append(data['rdataset'], data['rdataset']) 
                    thisdataset=np.where(data['rdataset'] == j+1)
                    thisdataset1=np.where(alldatasets == j+1)
                    allrvtimes=np.append(data['rtime'],data['rtime'])
                else:
                    thisdataset=np.where(data['rdataset'] == j+1)
                    thisdataset1=thisdataset[0]
                    allrvtimes=data['rtime']#[thisdataset1]
                if any('gamma' in s for s in inposindex):
                    print(allrvtimes[thisdataset1])
                    pl.plot(np.mod(allrvtimes[thisdataset1]-parstruc['epoch1'],parstruc['Per1']),data['rv'][thisdataset1]-parstruc['gamma'+str(j+1)],'ro')
                    if args.ploterrors: pl.errorbar(np.mod(allrvtimes[thisdataset1]-parstruc['epoch1'],parstruc['Per1']),data['rv'][thisdataset1]-parstruc['gamma'+str(j+1)], yerr=data['rverror'][thisdataset1],fmt='none',ecolor='red')
                    pl.plot(np.mod(data['rtime'][thisdataset]-parstruc['epoch1'],parstruc['Per1']),rmodel[thisdataset]-parstruc['gamma'+str(j+1)],'bo')
                    if args.binary: 
                        pl.plot(np.mod(data['rtime'][thisdataset]-parstruc['epoch1'],parstruc['Per1']),rmodel2[thisdataset]-parstruc['gamma'+str(j+1)],'go')
                        pl.plot(np.mod(allrvtimes[thisdataset1]-parstruc['epoch1'],parstruc['Per1']),data['rv'][thisdataset1]-np.append(rmodel[thisdataset],rmodel2[thisdataset]),'^',color='magenta')
                else:
                    pl.plot(np.mod(allrvtimes[thisdataset1]-parstruc['epoch1'],parstruc['Per1']),data['rv'][thisdataset1]-instruc['gamma'+str(j+1)],'ro')
                    if args.ploterrors: pl.errorbar(np.mod(allrvtimes[thisdataset1]-parstruc['epoch1'],parstruc['Per1']),data['rv'][thisdataset1]-instruc['gamma'+str(j+1)], yerr=data['rverror'][thisdataset1],fmt='none',ecolor='red')
                    pl.plot(np.mod(data['rtime'][thisdataset]-parstruc['epoch1'],parstruc['Per1']),rmodel[thisdataset]-instruc['gamma'+str(j+1)],'bo')
                    if args.binary: pl.plot(np.mod(data['rtime'][thisdataset]-parstruc['epoch1'],parstruc['Per1']),rmodel2[thisdataset]-parstruc['gamma'+str(j+1)],'go')
            if args.plotstep:
                pl.draw()
                pl.pause(0.1)
                pl.clf()
            if args.plotbest:
                pl.xlabel('orbital phase (days)')
                pl.ylabel('dRV')
                namelength=len(instruc['plotfile'])
                if instruc['plotfile'][namelength-4:namelength] == '.pdf':
                    pl.savefig(pp, format='pdf')
                    pl.clf()
                    if args.plotresids:
                        pl.plot(np.mod(data['rtime']-parstruc['epoch1'],parstruc['Per1']),data['rv']-rmodel,'ro')
                        if args.ploterrors:
                            pl.errorbar(np.mod(data['rtime']-parstruc['epoch1'],parstruc['Per1']),data['rv']-rmodel, yerr=data['rverror'],fmt='none',ecolor='red')
                        pl.plot([np.min(np.mod(data['rtime'][thisdataset]-parstruc['epoch1'],parstruc['Per1'])),np.max(np.mod(data['rtime'][thisdataset]-parstruc['epoch1'],parstruc['Per1']))],[0.0,0.0],color='blue')
                        pl.savefig(pp, format='pdf')
                    if not args.rm: pp.close()
                else:
                    pl.savefig(instruc['plotfile'], format=instruc['plotfile'][namelength-3:namelength])
                    print('Plot complete. If this is a multiplanet system and you want') 
                    print('more than the first planet, you must use PDF format.')
                                    
                                    
        #print parstruc['gamma1']

        if args.fitjitter:
            inv_sigma2=1.0/(data['rverror']**2+jitters**2)
            inv_sigma22=1.0/(data['rverror']**2)
            
        else:
            inv_sigma2 = 1.0/data['rverror']**2
            inv_sigma22 = 1.0/data['rverror']**2
        if args.binary: rmodel=np.append(rmodel,rmodel2) 
        if not args.rm: lnl+=np.sum((data['rv']-rmodel)**2*inv_sigma2 - np.log(inv_sigma2))

            
        if args.getprob and not args.rm:
                print('The RV chisq is ',np.sum((data['rv']-rmodel)**2*inv_sigma2),' and chisq_red=',np.sum((data['rv']-rmodel)**2*inv_sigma2)/(len(data['rv'])))
                print('The contribution of RVs to lnprob is ',np.sum((data['rv']-rmodel)**2*inv_sigma2 - np.log(inv_sigma2)))


    if args.rm:
        if not args.rvs: 
            nrvs=len(data['rtime'])
            rmodel=np.zeros(nrvs)
            rndatasets=np.max(data['rdataset'])
            if args.binary: rmodel2=np.zeros(nrvs)
            if any('rvtrend' in s for s in inposindex):
                rmodel+=(data['rtime']-parstruc['epoch1'])*parstruc['rvtrend']
                if any('rvtrendquad' in s for s in inposindex):
                    rmodel+=(data['rtime']-parstruc['epoch1'])**2*parstruc['rvtrendquad']
                if args.binary: rmodel2+=(data['rtime']-parstruc['epoch1'])*parstruc['rvtrend']
            if args.fitjitter: jitters=np.zeros(len(data['rdataset']))
            for j in range(0,rndatasets):
                thisdataset=np.where(data['rdataset'] == j+1)
                if any('gamma' in s for s in inposindex):
                    rmodel[thisdataset]+=parstruc['gamma'+str(j+1)]
                    if args.binary: rmodel2[thisdataset]+=parstruc['gamma'+str(j+1)]
                else:
                    rmodel[thisdataset]+=instruc['gamma'+str(j+1)]
                    if args.binary: rmodel2[thisdataset]+=instruc['gamma'+str(j+1)]
                if args.fitjitter:
                    jitters[thisdataset]=parstruc['jitter'+str(j+1)]

        rmdatasets=np.max(data['rdataset'])
        for i in range (0,rmdatasets):
            thisplanet=instruc['whichtomplanet'+str(i+1)]
            thesedata=np.where(data['rdataset'] == i+1)
            thesedata=thesedata[0]
            modelstruc={'Per':parstruc['Per'+thisplanet], 'rprs':parstruc['rprs'+thisplanet], 'aors':aors[i], 'inc':inc[i], 'ecc':ecc[i], 'omega':omega[i], 'b':parstruc['bpar'+thisplanet], 'lambda':parstruc['lambda'+thisplanet], 'vsini':parstruc['vsini'], 'photmodflag':'batman'}
            phased = np.mod(data['rtime'][thesedata]-parstruc['epoch'+thisplanet], parstruc['Per'+thisplanet])
            highs = np.where(phased > parstruc['Per'+thisplanet]/2.0)
            phased[highs]-=parstruc['Per'+thisplanet] #this is still in days
            modelstruc['longflag'], modelstruc['t'] = 'batman', phased
            if any('intwidth' in s for s in inposindex):
                modelstruc['intwidth'] = parstruc['intwidth']
            elif any('intwidth' in s for s in index):
                modelstruc['intwidth'] = np.float(instruc['intwidth'])
            else:
                modelstruc['intwidth'] = 5.
            if any('macroturb' in s for s in inposindex):
                modelstruc['intwidth'] = np.sqrt(modelstruc['intwidth']**2+parstruc['macroturb']**2) #account for macroturbulence if using.
            if any('q1t' in s for s in inposindex) and any('q1t' in s for s in inposindex):
                modelstruc['g1'], modelstruc['g2'] = 2.0*parstruc['q2t']*np.sqrt(parstruc['q1t']), np.sqrt(parstruc['q1t'])*(1.0-2.0*parstruc['q2t'])
            else:
                modelstruc['g1'], modelstruc['g2'] = parstruc['g1t'], parstruc['g2t']
            rmodel[thesedata]+=rmmodel(modelstruc)

        if args.plotbest or args.plotstep:
            pl.plot(data['rtime']-parstruc['epoch'+thisplanet],data['rv'],'ro')
            pl.plot(data['rtime']-parstruc['epoch'+thisplanet],rmodel,'bo')
            f=open('HD63433_e1_rm_out.txt','w')
            for i in range (0,len(data['rtime'])): f.write(str(data['rtime'][i])+' '+str(data['rv'][i])+' '+str(data['rverror'][i])+' '+str(rmodel[i])+' \n')
            f.close()
            if args.ploterrors: pl.errorbar(data['rtime']-parstruc['epoch1'],data['rv'], yerr=data['rverror'],fmt='none',ecolor='red')
            if args.plotstep:
                pl.draw()
                pl.pause(2.0)
                pl.clf()
            if args.plotbest:
                pl.xlabel('Days from center of transit')
                pl.ylabel('RV (km s$^{-1}$)')
                namelength=len(instruc['plotfile'])
                if instruc['plotfile'][namelength-4:namelength] == '.pdf':
                    pl.savefig(pp, format='pdf')
                    pl.clf()
                    if args.plotresids:
                        pl.plot(data['rtime']-parstruc['epoch1'],data['rv']-rmodel,'ro')
                        if args.ploterrors:
                            pl.errorbar(data['rtime']-parstruc['epoch1'],data['rv']-rmodel, yerr=data['rverror'],fmt='none',ecolor='red')
                        pl.plot([np.min(data['rtime']-parstruc['epoch1']),np.max(data['rtime']-parstruc['epoch1'])],[0.0,0.0],color='blue')
                        pl.savefig(pp, format='pdf')
                    pp.close()
                else:
                    pl.savefig(instruc['plotfile'], format=instruc['plotfile'][namelength-3:namelength])
                    print('Plot complete. If this is a multiplanet system and you want') 
                    print('more than the first planet, you must use PDF format.')
        if args.fitjitter:
            inv_sigma2=1.0/(data['rverror']**2+jitters**2)
            inv_sigma22=1.0/(data['rverror']**2)
            
        else:
            inv_sigma2 = 1.0/data['rverror']**2
            inv_sigma22 = 1.0/data['rverror']**2
        lnl+=np.sum((data['rv']-rmodel)**2*inv_sigma2 - np.log(inv_sigma2))

        if args.getprob:
                print('The RV chisq is ',np.sum((data['rv']-rmodel)**2*inv_sigma2),' and chisq_red=',np.sum((data['rv']-rmodel)**2*inv_sigma2)/(len(data['rv'])))
                print('The contribution of RVs to lnprob is ',np.sum((data['rv']-rmodel)**2*inv_sigma2 - np.log(inv_sigma2)))
    if args.getprob and (args.rvs or args.rm):
        print(np.sum(np.log(inv_sigma2)),np.sum(np.log(inv_sigma22)),np.sum(np.log(1./inv_sigma2)))
        #print np.log(len(data['rv'])+len(data['pflux'])),len(data['rv']),len(data['pflux']),len(parstruc),-0.5*lnl, len(data['rv']), len(data['pflux'])
        ##print 'The BIC is ',np.log(len(data['rv'])+len(data['pflux']))*len(parstruc)+np.sum((data['rv']-rmodel)**2*inv_sigma2),np.log(len(data['rv'])+len(data['pflux']))*len(parstruc)+lnl
        print('The BIC is ',np.log(len(data['rv']))*len(parstruc)+np.sum((data['rv']-rmodel)**2*inv_sigma2),np.log(len(data['rv']))*len(parstruc)+lnl)
    #if args.time: print 'ending lnlike ',temptime
    return -0.5*lnl

#LNPRIOR
def lnprior(parstruc, priorstruc, instruc, theta, inposindex):
    #prevent parameters from going outside of physical bounds
    #startprior=timemod.time()-1457540000.
    #print 'starting the prior ',startprior
    lp=0.0
    if any('sinw' in s for s in inposindex) or any('ecc' in s for s in inposindex):
        if any('sesinw' in s for s in inposindex):
            #omega=np.arctan(theta[[i for i, s in enumerate(inposindex) if 'sesinw' in s]]/theta[[i for i, s in enumerate(inposindex) if 'secosw' in s]])
            #ecc=theta[[i for i, s in enumerate(inposindex) if 'secosw' in s]]**2/np.cos(omega)**2
            ecc=theta[[i for i, s in enumerate(inposindex) if 'sesinw' in s]]**2+theta[[i for i, s in enumerate(inposindex) if 'secosw' in s]]**2
            omega=np.arccos(theta[[i for i, s in enumerate(inposindex) if 'secosw' in s]]/np.sqrt(ecc))
            news=np.where(theta[[i for i, s in enumerate(inposindex) if 'sesinw' in s]] < 0.)
            omega[news]=2.*np.pi-omega[news]
            if any(not np.isfinite(t) for t in omega):
                bads=np.where(np.isfinite(omega) == True)
                omega[bads]=np.pi/2.0
                temp=theta[[i for i, s in enumerate(inposindex) if 'sesinw' in s]]
                ecc[bads]=temp[bads]**2
        if any('ecc' in s for s in inposindex):
            ecc=theta[[i for i, s in enumerate(inposindex) if 'ecc' in s]]
            omega=theta[[i for i, s in enumerate(inposindex) if 'omega' in s]]

        if any('ecsinw' in s for s in inposindex):
            ecc=np.sqrt(theta[[i for i, s in enumerate(inposindex) if 'ecsinw' in s]]**2+theta[[i for i, s in enumerate(inposindex) if 'eccosw' in s]]**2)
            omega=np.arccos(theta[[i for i, s in enumerate(inposindex) if 'eccosw' in s]]/ecc)
            news=np.where(theta[[i for i, s in enumerate(inposindex) if 'ecsinw' in s]] < 0.)
            omega[news]=2.*np.pi-omega[news]
            if any(not np.isfinite(t) for t in omega):
                bads=np.where(np.isfinite(omega) == True)
                omega[bads]=np.pi/2.0
                temp=theta[[i for i, s in enumerate(inposindex) if 'ecsinw' in s]]
                ecc[bads]=temp[bads]

        omega*=180./np.pi #radians->degrees
            
            
    #will need to do this for esinw, ecosw. code up later.
    else:
        ecc=np.zeros(nplanets)
        omega=np.zeros(nplanets)+90.
    #will need to do this for esinw, ecosw. code up later.

    #reject values that will make JKTEBOP crash if using JKTEBOP
    if args.binary and args.photometry:
        if instruc['photmodflag'] == 'jktebop':
            if any (t < 1./0.8 for t in theta[[i for i, s in enumerate(inposindex) if 'aors' in s]]):
                return -np.inf
    
    #reject unphysical values
    if any(t >= 1.0 for t in theta[[i for i, s in enumerate(inposindex) if 'rprs' in s]]) and not args.binary:
        return -np.inf #don't want to disallow radius ratio >1 if fitting EB

    #reject unphysical values of limb darkening when using Kipping sampling
    if any('q1' in s for s in inposindex):
        g1, g2 = 2.0*theta[[i for i, s in enumerate(inposindex) if 'q2' in s]]*np.sqrt(theta[[i for i, s in enumerate(inposindex) if 'q1' in s]]), np.sqrt(theta[[i for i, s in enumerate(inposindex) if 'q1' in s]])*(1.0-2.0*theta[[i for i, s in enumerate(inposindex) if 'q2' in s]])
        if np.min(g1) < 0. or np.min(g2) < 0. or np.max(g1) > 1. or np.max(g2) > 1.:
            return -np.inf
        
    elimit = 0.95
    if any(t <= 0.0 for t in theta[[i for i, s in enumerate(inposindex) if 'Per' in s]]) or any(t <= 0.0 for t in theta[[i for i, s in enumerate(inposindex) if 'rprs' in s]]) or any(np.abs(t) > 1.0 for t in theta[[i for i, s in enumerate(inposindex) if 'sesinw' in s]]) or any(np.abs(t) > 1.0 for t in theta[[i for i, s in enumerate(inposindex) if 'secosw' in s]]) or any(t <= 0 for t in theta[[i for i, s in enumerate(inposindex) if 'rhostar' in s]]) or any(t < 1. for t in theta[[i for i, s in enumerate(inposindex) if 'aors' in s]]) or any(t <= 0.0 for t in theta[[i for i, s in enumerate(inposindex) if 'q1' in s]]) or any(t <= 0.0 for t in theta[[i for i, s in enumerate(inposindex) if 'q2' in s]]) or any(t >= 1.0 for t in theta[[i for i, s in enumerate(inposindex) if 'q1' in s]]) or any(t >= 1.0 for t in theta[[i for i, s in enumerate(inposindex) if 'q2' in s]]) or any(t < 0.0 for t in ecc) or any(t >= elimit for t in ecc) or any(t <= 0 for t in theta[[i for i, s in enumerate(inposindex) if 'vsini' in s]]) or any(t <= 0 for t in theta[[i for i, s in enumerate(inposindex) if 'semiamp' in s]]) or any(t <= 0 for t in theta[[i for i, s in enumerate(inposindex) if 'intwidth' in s]]) or any(t <= 0 for t in theta[[i for i, s in enumerate(inposindex) if (('gp' in s) & ('partau' in s))]]) or any(t <= 0 for t in theta[[i for i, s in enumerate(inposindex) if (('gp' in s) & ('paramp' in s))]]) or any(t <= 0 for t in theta[[i for i, s in enumerate(inposindex) if (('gp' in s) & ('parGamma' in s))]]) or any(t <= 0 for t in theta[[i for i, s in enumerate(inposindex) if (('gp' in s) & ('parP' in s))]]) or any(t < 0. for t in theta[[i for i, s in enumerate(inposindex) if 'jitter' in s]]) or any(t <= 0. for t in theta[[i for i, s in enumerate(inposindex) if 'fluxrat' in s]]) or any(np.abs(t) > 1. for t in theta[[i for i, s in enumerate(inposindex) if 'cosi' in s]]) or any(t <= 0 for t in theta[[i for i, s in enumerate(inposindex) if 'macroturb' in s]]) or any(t <= 0 for t in theta[[i for i, s in enumerate(inposindex) if 'spotrad' in s]]): #not totally sure this will work... #for now, set ecc limit to 0.99 to avoid mysterious batman crashes
        return -np.inf

    # for celerite SHOMixture (limit all gpppar to be log(p) between -10 and 20:
    if any(t < -10 for t in theta[[i for i, s in enumerate(inposindex) if 'gppparmix' in s]]) or any(t > 10 for t in theta[[i for i, s in enumerate(inposindex) if 'gppparmix' in s]]) or any(t < 1 for t in theta[[i for i, s in enumerate(inposindex) if 'gppparQ1' in s]]) or any(t < 1 for t in theta[[i for i, s in enumerate(inposindex) if 'gppparQ2' in s]]) or any(t < 0 for t in theta[[i for i, s in enumerate(inposindex) if 'gppparLogQ1' in s]]) or any(t < 0 for t in theta[[i for i, s in enumerate(inposindex) if 'gppparLogQ2' in s]]):
        return -np.inf

        
    #handle the user-given priors
    priorindex=priorstruc['index']
    npriors=len(priorindex)
    lp=0.0
    #print 'starting the prior loop'
    for i in range (0, npriors):
        if any(priorindex[i] in s for s in inposindex): 
            lp+=(parstruc[priorindex[i]]-np.float(instruc[priorindex[i]]))**2/np.float(priorstruc[priorindex[i]])**2
            if args.getprob:
                print('The value of the prior for ',priorindex[i],' is ',(parstruc[priorindex[i]]-np.float(instruc[priorindex[i]]))**2/np.float(priorstruc[priorindex[i]])**2)
        #doing it this way handles any entries in the prior file that don't correspond to fit variables
        #but will probably want to modify this if want to be able to put penalties on parameters that are derived from fit variables
        #I'll work on that later...
    #print 'ending the prior loop


    lp*=(-0.5)

    if not np.isfinite(lp):
        return 0.0

    return lp

def lnpriorPT(theta, data, nplanets, priorstruc, inposindex, instruc, args):

    parstruc = dict(list(zip(inposindex, theta)))
    if not any('none' in s for s in priorstruc): 
        #if args.time: print 'about to do the prior',thisisthestart
        lp = lnprior(parstruc, priorstruc, instruc, theta, inposindex)
        if not np.isfinite(lp) or np.isnan(lp):
            if args.time: print('ending, prior out of range ',thisisthestart)
            return -np.inf

        else:
            return lp

    else:
        return 0.0

def lnprob(theta, data, nplanets, priorstruc, inposindex, instruc, args):
    data['count']+=1

    if args.time: 
        thisisthestart=timemod.time()-1457540000.
        print('starting ',thisisthestart)
    parstruc = dict(list(zip(inposindex, theta)))

    lnl=0.

    if not any('none' in s for s in priorstruc) and not args.pt: 
        #if args.time: print 'about to do the prior',thisisthestart
        lp = lnprior(parstruc, priorstruc, instruc, theta, inposindex)
        if not np.isfinite(lp) or np.isnan(lp):
            if args.time: print('ending, prior out of range ',thisisthestart)
            return -np.inf
        lnl+=lp
        #if args.time: print 'done with the prior',thisisthestart

    #if args.time: print 'about to do lnlike',thisisthestart
    lnl+= lnlike(theta, parstruc, data, nplanets, inposindex, instruc, args)
    #if args.time: print 'finished with lnlike',thisisthestart

    if np.isnan(lnl):
        print('The likelihood returned a NaN',lp,lnl)
        return -np.inf #take care of nan crashes, hopefully...

    if args.verbose:
        print(data['count'], lnl)

    if args.getprob and args.rvs:
        print('The total value of lnprob is ',lnl)
        #print 'The BIC is ',np.log(len(data['rv'])+len(data['pflux']))*len(parstruc)-2.*lnl
        print('The BIC is ',np.log(len(data['rv'])+len(data['pflux']))*len(parstruc)-2.*lnl)
        

    if args.time:
        thisistheend=timemod.time()
        #print 'The elapsed time for the model call was ',thisistheend-thisisthestart,' seconds'
        print('ending ',thisisthestart)

    return lnl

#set up the initial position for emcee and get all of the data that are needed
data={}

#photometric parameters
if args.photometry:
    #inpos = np.array((epoch, Per, rprs, bpar))
    inpos, inposindex, perturbs = np.array(epoch), np.array(index[[i for i, s in enumerate(index) if 'epoch' in s]]), np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'epoch' in s]], dtype=np.float)
    inpos, inposindex, perturbs = np.append(inpos, Per), np.append(inposindex, index[[i for i, s in enumerate(index) if 'Per' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'Per' in s]], dtype=np.float))
    inpos, inposindex, perturbs = np.append(inpos, rprs), np.append(inposindex, index[[i for i, s in enumerate(index) if 'rprs' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'rprs' in s]], dtype=np.float))
    if rhobaflag != 'aorscosi':
        inpos, inposindex, perturbs = np.append(inpos, bpar), np.append(inposindex, index[[i for i, s in enumerate(index) if 'bpar' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'bpar' in s]], dtype=np.float))
    else:
        inpos, inposindex, perturbs = np.append(inpos, cosi), np.append(inposindex, index[[i for i, s in enumerate(index) if 'cosi' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'cosi' in s]], dtype=np.float))

    if rhobaflag == 'rhostarb':
        inpos, inposindex, perturbs = np.append(inpos, rhostar), np.append(inposindex, 'rhostar'), np.append(perturbs, np.float(perturbstruc['rhostar']))
    if rhobaflag == 'aorsb' or rhobaflag == 'aorscosi':
        inpos, inposindex, perturbs = np.append(inpos, aors), np.append(inposindex, index[[i for i, s in enumerate(index) if 'aors' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'aors' in s]], dtype=np.float))


    if photlcflag == 'q':
        for i in range (0, pnfilters): inpos, inposindex, perturbs = np.append(inpos, [q1p[i], q2p[i]], axis=0), np.append(inposindex, ['q1p'+str(i+1), 'q2p'+str(i+1)], axis=0), np.append(perturbs, [perturbstruc['q1p'+str(i+1)], perturbstruc['q2p'+str(i+1)]], axis=0)
    if photlcflag == 'g':
        for i in range (0, pnfilters): inpos, inposindex, perturbs = np.append(inpos, [g1p[i], g2p[i]], axis=0), np.append(inposindex, ['g1p'+str(i+1), 'g2p'+str(i+1)], axis=0), np.append(perturbs, [perturbstruc['g1p'+str(i+1)], perturbstruc['g2p'+str(i+1)]], axis=0)

    if args.binary: 
        data['binfflag']=binfflag
        if binfflag == 'rprsfluxr':
            inpos, inposindex, perturbs = np.append(inpos,fluxrat), np.append(inposindex, index[[i for i, s in enumerate(index) if 'fluxrat' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'fluxrat' in s]], dtype=np.float))
        if binfflag == 'mycomb':
            combs=rprs**2*fluxrat
            inpos, inposindex, perturbs = np.append(inpos,combs), np.append(inposindex, index[[i for i, s in enumerate(index) if 'fluxrat' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'fluxrat' in s]], dtype=np.float))
            
        

    data['ptime'], data['pflux'], data['perror'], data['pexptime'], data['pfilter'], data['pdataset']  =ptime,pflux,perror,pexptime, pfilter, pdataset 

#tomographic parameters will go here




if args.tomography or args.line or args.rm:
    if not args.photometry and (args.tomography or args.rm):
        inpos, inposindex, perturbs = np.array(epoch), np.array(index[[i for i, s in enumerate(index) if 'epoch' in s]]), np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'epoch' in s]], dtype=np.float)
        inpos, inposindex, perturbs = np.append(inpos, Per), np.append(inposindex, index[[i for i, s in enumerate(index) if 'Per' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'Per' in s]], dtype=np.float))
        inpos, inposindex, perturbs = np.append(inpos, rprs), np.append(inposindex, index[[i for i, s in enumerate(index) if 'rprs' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'rprs' in s]], dtype=np.float))
        inpos, inposindex, perturbs = np.append(inpos, bpar), np.append(inposindex, index[[i for i, s in enumerate(index) if 'bpar' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'bpar' in s]], dtype=np.float))

        if rhobaflag == 'rhostarb':
            inpos, inposindex, perturbs = np.append(inpos, rhostar), np.append(inposindex, 'rhostar'), np.append(perturbs, np.float(perturbstruc['rhostar']))
        if rhobaflag == 'aorsb':
            inpos, inposindex, perturbs = np.append(inpos, aors), np.append(inposindex, index[[i for i, s in enumerate(index) if 'aors' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'aors' in s]], dtype=np.float))

    if not args.photometry and not (args.tomography or args.rm):
        inpos, inposindex, perturbs = np.array(np.float(struc1['vsini'])), np.array('vsini'), np.array(np.float(perturbstruc['vsini']))
    else:
        inpos, inposindex, perturbs = np.append(inpos, np.float(struc1['vsini'])), np.append(inposindex, 'vsini'), np.append(perturbs, np.float(perturbstruc['vsini']))
    if args.tomography or args.rm: inpos, inposindex, perturbs = np.append(inpos, llambda), np.append(inposindex, index[[i for i, s in enumerate(index) if 'lambda' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'lambda' in s]], dtype=np.float))
    if tomlcflag == 'q':
        inpos, inposindex, perturbs = np.append(inpos, [q1t, q2t]), np.append(inposindex, ['q1t', 'q2t']), np.append(perturbs, [perturbstruc['q1t'], perturbstruc['q2t']])#, axis=0
    if tomlcflag == 'g':
        inpos, inposindex, perturbs = np.append(inpos, [g1t, g2t], axis=0), np.append(inposindex, ['g1t', 'g2t'], axis=0), np.append(perturbs, [perturbstruc['g1t'], perturbstruc['g2t']], axis=0)
    
    if any('fitintwidth' in s for s in index):
        if struc1['fitintwidth'] == 'True':
            inpos, inposindex, perturbs = np.append(inpos, np.float(struc1['intwidth'])), np.append(inposindex, 'intwidth'), np.append(perturbs, np.float(perturbstruc['intwidth']))

    if args.skyline:
        inpos, inposindex, perturbs = np.append(inpos, [np.float(struc1['skycen']), np.float(struc1['skydepth'])], axis=0), np.append(inposindex, ['skycen', 'skydepth'], axis=0), np.append(perturbs, [perturbstruc['skycen'], perturbstruc['skydepth']], axis=0)

    if any('tomdrift' in s for s in index):
        for i in range (0,len(tomdriftc)):
            inpos, inposindex, perturbs = np.append(inpos, [tomdriftc[i], tomdriftl[i]], axis=0), np.append(inposindex, ['tomdriftc'+str(i+1),'tomdriftl'+str(i+1)], axis=0), np.append(perturbs, [perturbstruc['tomdriftc'+str(i+1)],perturbstruc['tomdriftl'+str(i+1)]], axis=0)

    if any('macroturb' in s for s in index):
        inpos, inposindex, perturbs = np.append(inpos, macroturb), np.append(inposindex,'macroturb'), np.append(perturbs,perturbstruc['macroturb'])

    if args.tomography: 
        if args.spots:
            inpos, inposindex, perturbs = np.append(inpos, rspots), np.append(inposindex, index[[i for i, s in enumerate(index) if 'spotrad' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'spotrad' in s]], dtype=np.float))
            inpos, inposindex, perturbs = np.append(inpos, t0spots), np.append(inposindex, index[[i for i, s in enumerate(index) if 'spott0' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'spott0' in s]], dtype=np.float))
            inpos, inposindex, perturbs = np.append(inpos, np.float(struc1['Prot'])), np.append(inposindex, 'Prot'), np.append(perturbs, np.float(perturbstruc['Prot']))
            data['nspots']=nspots

        data['tomdict']=tomdict
        #data['ttime'], data['profarr'], data['profarrerr'], data['texptime'], data['vabsfine'], data['dofft'] = ttime, profarr, profarrerr, texptime, vabsfine, dofft
        if dofft:
            data['mask']=mask

    if args.binary:
        inpos, inposindex, perturbs = np.append(inpos, np.float(struc1['vsinib'])), np.append(inposindex, 'vsinib'), np.append(perturbs, np.float(perturbstruc['vsinib']))

if args.line or args.tomography: 
    if any('linecenter' in s for s in index):
        inpos, inposindex, perturbs = np.append(inpos, linecenter), np.append(inposindex, index[[i for i, s in enumerate(index) if 'linecenter' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'linecenter' in s]], dtype=np.float))



    if args.line: data['lineprof'], data['lineerr'], data['linevel'] = lineprof, lineerr, linevel    
    
    
if args.rvs or ((args.tomography or args.line) and args.binary):
    if not args.tomography and not args.photometry:
        inpos, inposindex, perturbs = np.array(epoch), np.array(index[[i for i, s in enumerate(index) if 'epoch' in s]]), np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'epoch' in s]], dtype=np.float)
        inpos, inposindex, perturbs = np.append(inpos, Per), np.append(inposindex, index[[i for i, s in enumerate(index) if 'Per' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'Per' in s]], dtype=np.float))
    
    inpos, inposindex, perturbs = np.append(inpos, semiamp), np.append(inposindex, index[[i for i, s in enumerate(index) if 'semiamp' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'semiamp' in s]], dtype=np.float))

if args.rvs or args.rm or ((args.tomography or args.line) and args.binary):
    if fixgamma != 'True':
        inpos, inposindex, perturbs = np.append(inpos, gamma), np.append(inposindex, index[[i for i, s in enumerate(index) if 'gamma' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'gamma' in s]], dtype=np.float))
        
    if fittrend:
        inpos, inposindex, perturbs = np.append(inpos, rvtrend), np.append(inposindex, 'rvtrend'), np.append(perturbs, perturbinvals[[i for i, s in enumerate(perturbindex) if 'rvtrend' in s]])     
        if any('rvtrendquad' in s for s in struc1):
            inpos, inposindex = np.append(inpos, rvtrendquad), np.append(inposindex, 'rvtrendquad') #quad already handled by above line if present

    if args.fitjitter:
        inpos, inposindex, perturbs = np.append(inpos, jitter), np.append(inposindex, index[[i for i, s in enumerate(index) if 'jitter' in s]]), np.append(perturbs, np.array(perturbinvals[[i for i, s in enumerate(perturbindex) if 'jitter' in s]], dtype=np.float))
    
    data['rtime'], data['rv'], data['rverror'], data['rdataset'] = rtime,rv,rverror,rdataset
        
        

#add eccentricity if it's being fit
if fitecc == True:
    for i in range (0, nplanets):
        inpos, inposindex, perturbs = np.append(inpos, [eccpar[i], omegapar[i]]), np.append(inposindex, [enames[0]+str(i+1), enames[1]+str(i+1)], axis=0), np.append(perturbs, [perturbstruc['ecc'+str(i+1)], perturbstruc['omega'+str(i+1)]])
        #inpos, inposindex, perturbs = np.append(inpos, [0.2134013996723604, -0.28886219846381167]), np.append(inposindex, [enames[0]+str(i+1), enames[1]+str(i+1)], axis=0), np.append(perturbs, [perturbstruc['ecc'+str(i+1)], perturbstruc['omega'+str(i+1)]]) #HACK--remove this later!!!

#add ttvs if being fit
if args.ttvs:
    for i in range (0, nttvpars):
        inpos, inposindex, perturbs = np.append(inpos, ttvpars[i]), np.append(inposindex, ttvparnames[i]), np.append(perturbs, perturbstruc[ttvparnames[i]])


    data['ttvmodtype'], data['dottvs'] = modtype, dottvs

#add dilution if being included
if args.dilution:
    for i in range (0, ndilute):
        inpos, inposindex, perturbs = np.append(inpos, dilution[i]), np.append(inposindex, dilutionnames[i]), np.append(perturbs, perturbstruc[dilutionnames[i]])

if any ('perturbfac' in s for s in perturbindex):
    perturbs*=perturbstruc['perturbfac']

#add Gaussian process parameters if being fit
if args.gp:
    for i in range (0, ngppars):
        inpos, inposindex, perturbs = np.append(inpos, gppars[i]), np.append(inposindex, gpparnames[i]), np.append(perturbs, perturbstruc[gpparnames[i]])


    data['gpmodtype']= gpmodtype
    if args.photometry:
        data['gppuse'] = gppuse


ndim=len(inpos)

inpos=np.array(inpos,dtype=np.float)

#see if there is already an old chain file to load
try:
    chainin=np.load(chainfile)
except IOError:
    if not args.pt:
        pos = [inpos + perturbs*np.random.randn(ndim) for i in range(nwalkers)]
    else:
        ntemps=np.int(struc1['ntemps'])
        pos = np.zeros((ntemps,nwalkers,ndim))
        for i in range (0,ntemps): 
            for j in range (0,nwalkers): 
                pos[i,j,:] = inpos + perturbs*np.random.randn(ndim)
    loaded=False
else:
    if not args.startnew:
        chainshape=chainin.shape
        nin=chainshape[1]
        pos=chainin[:,nin-1,:]
        probin=np.load(probfile)
        loaded=True
    else:
        if not args.pt:
            pos = [inpos + perturbs*np.random.randn(ndim) for i in range(nwalkers)]
        else:
            ntemps=np.int(struc1['ntemps'])
            pos = np.zeros((ntemps,nwalkers,ndim))
            for i in range (0,ntemps): 
                for j in range (0,nwalkers): 
                    pos[i,j,:] = inpos + perturbs*np.random.randn(ndim)
        loaded=False

#check for bad starting values
#still need to add GP parameters
pos=np.array(pos)
if not loaded:
    for i in range (0,nwalkers):
        for j in range (0,ndim):
            if not args.pt:
                while ('Per' in inposindex[j] and pos[i,j] <= 0.0) or ('rprs' in inposindex[j] and (pos[i,j] <= 0.0 or pos[i,j] >= 1.0)) or ('sesinw'  in inposindex[j] and np.abs(pos[i,j]) > 1.0) or ('secosw'  in inposindex[j] and np.abs(pos[i,j]) > 1.0) or ('rhostar' in inposindex[j] and pos[i,j] <= 0.0) or ('aors' in inposindex[j] and pos[i,j] < 1.0) or ('q1' in inposindex[j] and pos[i,j] <= 0.0) or ('q2' in inposindex[j] and pos[i,j] <= 0.0) or ('q1' in inposindex[j] and pos[i,j] >= 1.0) or ('q2' in inposindex[j] and pos[i,j] >= 1.0) or ('ecc' in inposindex[j] and 'cos' not in inposindex[j] and pos[i,j] >= 0.99) or ('ecc' in inposindex[j] and 'cos' not in inposindex[j] and pos[i,j] < 0.0) or ('vsini' in inposindex[j] and pos[i,j] <= 0.0) or ('semiamp' in inposindex[j] and pos[i,j] <= 0.0) or ('intwidth' in inposindex[j] and pos[i,j] <= 0.0) or ('jitter' in inposindex[j] and pos[i,j] <= 0.0) or ('cosi' in inposindex[j] and pos[i,j] < 0.)  or ('parP' in inposindex[j] and pos[i,j] <= 0.) or ('paramp'in  inposindex[j] and pos[i,j] <= 0.) or ('partau' in inposindex[j] and pos[i,j] <= 0.) or ('parGamma' in inposindex[j] and pos[i,j] < 0.) or ('spotrad' in inposindex[j] and pos[i,j] <= 0.):
                    print(inposindex[j],pos[i,j],j,i)
                    pos[i,j]=inpos[j]+perturbs[j]*np.random.randn(1)
                if args.binary and args.photometry: 
                    if struc1['photmodflag'] == 'jktebop': #JKTEBOP can't handle (R1+R2)/a < 1.25
                        while ('aors' in inposindex[j] and pos[i,j] < 1.25):
                            print(inposindex[j],pos[i,j],j,i)
                            pos[i,j]=inpos[j]+perturbs[j]*np.random.randn(1)
            else: 
                for k in range (0,ntemps):
                    while ('Per' in inposindex[j] and pos[k,i,j] <= 0.0) or ('rprs' in inposindex[j] and (pos[k,i,j] <= 0.0 or pos[k,i,j] >= 1.0)) or ('sesinw'  in inposindex[j] and np.abs(pos[k,i,j]) > 1.0) or ('secosw'  in inposindex[j] and np.abs(pos[k,i,j]) > 1.0) or ('rhostar' in inposindex[j] and pos[k,i,j] <= 0.0) or ('aors' in inposindex[j] and pos[k,i,j] <= 0.0) or ('q1' in inposindex[j] and pos[k,i,j] <= 0.0) or ('q2' in inposindex[j] and pos[k,i,j] <= 0.0) or ('q1' in inposindex[j] and pos[k,i,j] >= 1.0) or ('q2' in inposindex[j] and pos[k,i,j] >= 1.0) or ('ecc' in inposindex[j] and 'cos' not in inposindex[j] and pos[k,i,j] >= 0.99) or ('ecc' in inposindex[j] and 'cos' not in inposindex[j] and pos[k,i,j] < 0.0) or ('vsini' in inposindex[j] and pos[k,i,j] <= 0.0) or ('semiamp' in inposindex[j] and pos[k,i,j] <= 0.0) or ('intwidth' in inposindex[j] and pos[k,i,j] <= 0.0) or ('jitter' in inposindex[j] and pos[k,i,j] <= 0.0) or ('cosi' in inposindex[j] and pos[k,i,j] < 0.) or ('parP' in inposindex[j] and pos[k,i,j] <= 0.) or ('paramp'in  inposindex[j] and pos[k,i,j] <= 0.) or ('partau' in inposindex[j] and pos[k,i,j] <= 0.) or ('parGamma' in inposindex[j] and pos[k,i,j] < 0.):
                        print(inposindex[j],pos[k,i,j],j,i,k)
                        pos[k,i,j]=inpos[j]+perturbs[j]*np.random.randn(1)
                


if args.plotbest or args.getprob:

    if any('picklefile' in s for s in index):
        print("Writing Ellie's pickle file ln 1859")
        import pickle
        mydict = {}
    if not args.startnew:
    #get the best-fit parameters from the loaded chain and plot them
        inpos=np.zeros(ndim)
        if any('nburnin' in s for s in index): 
            nburnin=int(struc1['nburnin'])
        else:
            nburnin=nin//5
        samples=chainin[:,nburnin:,:].reshape((-1,ndim))
        if args.bestprob:
            best=np.where(probin == np.max(probin))
        for i in range (0,ndim):
        #print goods.shape,'SHAPE'
            #goods=np.where(np.abs(chainin[:,nburnin,i]-np.mean(chainin[:,nburnin,i])) <= 5.*np.std(chainin[:,nburnin:,i])) #not quite correct way to do this, but prob OK for now
            #inpos[i]=np.mean(chainin[goods,nburnin,i])
            if args.bestprob:
                temp=[chainin[best[0][0],best[1][0],i], 0, 0]
            else:
                if 'bpar' in inposindex[i] and not args.tomography:
                    v=np.percentile(np.abs(samples[:,i]), [16, 50, 84], axis=0)
                else: 
                    v=np.percentile(samples[:,i], [16, 50, 84], axis=0)
                temp=[v[1], v[2]-v[1], v[1]-v[0]]
        
            #temp=map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), np.percentile(samples[:,i], [16, 50, 84], axis=0))
            #print 'The best fit value of ',inposindex[i],' is ',inpos[i],' +/- ', np.std(chainin[goods,nburnin,i])
            print('The best fit value of ',inposindex[i],' is ',temp[0],' + ', temp[1],' - ',temp[2])
            ##print('picklefile',index)
            
            if any('picklefile' in s for s in index):
                mydict[inposindex[i]] = temp

                
            inpos[i]=temp[0]
        print('for ',nin,' steps total and cutting off the first ',nburnin,' steps')
    else:
        for i in range (0,ndim):
            print('The starting value of ',inposindex[i],' is ',inpos[i])
    namelength=len(struc1['plotfile'])
    if struc1['plotfile'][namelength-4:namelength] == '.pdf':
        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages(struc1['plotfile'])
    #pos=[inpos for i in range(nwalkers)]
    
    
data['count']=0


#set up and run the MCMC sampler
if not args.getprob and not args.plotbest:
    if not args.pt:
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data, nplanets, priorstruc, inposindex, struc1, args), pool=pool)#threads=nthreads)
            sampler.run_mcmc(pos, nsteps,progress=True)
            
        #tau = sampler.get_autocorr_time(tol=0)
        #print('Tau: ',np.mean(tau),sampler.iteration)
            
    else:
        sampler=emcee.PTSampler(ntemps, nwalkers, ndim, lnprob, lnpriorPT, threads=nthreads,loglargs=(data, nplanets, priorstruc, inposindex, struc1, args),logpargs=(data, nplanets, priorstruc, inposindex, struc1, args))

        for p, lnprob, lnlike in sampler.sample(pos, iterations=nsteps/10):
            pass
        sampler.reset()

        for p, lnprob, lnlike in sampler.sample(p, lnprob0=lnprob, lnlike0=lnlike, iterations=nsteps):
            pass
            #let's see if this works.....




    print('The order of your parameters is,',inposindex)

    if loaded == False:
        np.save(chainfile, np.array(sampler.chain))
        np.save(probfile, np.array(sampler.lnprobability))
        np.save(accpfile,np.array(sampler.acceptance_fraction))
        chain=np.array(sampler.chain)
    if loaded == True:
        chain=np.array(sampler.chain)
        prob=np.array(sampler.lnprobability)
        accp=np.array(sampler.acceptance_fraction)
        chain2=np.append(chainin,chain,axis=1)
        prob2=np.append(probin,prob,axis=1)
        accpin=np.load(accpfile)
        accpin*=nin #to get total number of acceptances
        accp*=nsteps #ditto
        accp2=accpin+accp
        accp2=accp2/(nin+nsteps) #to get back to fraction
        np.save(chainfile, chain2)
        np.save(probfile, prob2)
        np.save(accpfile, accp2)
        chain=chain2

    if any('asciiout' in s for s in index): 
        if any('nburnin' in s for s in index): 
            nburnin=struc1['nburnin']
        else:
            nburnin=0
        samples=chain[:,nburnin:,:].reshape((-1,ndim))
        f=open(struc1['asciiout'], 'w')
        for j in range (0,ndim):
            f.write(str(inposindex[j]))
            if j != ndim-1: f.write('\t')
        f.write('\n')
        for i in range (0,(nsteps-nburnin)*nwalkers):
            for j in range (0,ndim):
                f.write(str(samples[i,j]))
                if j != ndim-1: f.write('\t')
            f.write('\n')
        f.close()

else:
    lnprob1=lnprob(inpos, data, nplanets, priorstruc, inposindex, struc1, args)

#temp=chain.shape
#thewholeend=timemod.time()
#print('This run of MISTTBORN took a total time of ',thewholeend-thewholestart,' seconds, for ',nsteps,' steps and a total length of ',temp[1],' steps')
