import numpy as np
import subprocess
import sys                    #import all necessary modules
import multiprocessing as mp
import time
import pyLikelihood
import gt_apps as gt #these are the fermi tools
from BinnedAnalysis import *
import os
import matplotlib.pyplot as plt
import glob
import re
from make4FGLxml import *
import astropy.io.fits

J1023 = {'name' : 'J1023', 'model_name' : '4FGL J1023.7+0038', 'xcoord' : 155.946, 'ycoord' : 0.645, 'rec_binsz' : 2000000}
J12270 = {'name' : 'J12270', 'model_name' : '4FGL J1228.0-4853', 'xcoord' : 186.995, 'ycoord' : -48.8952, 'rec_binsz' : 4000000}
J18245 = {'name' : 'J18245', 'model_name' : '4FGL J1824.6-2452', 'xcoord' : 276.135, 'ycoord' : -24.8688, 'rec_binsz' : 12000000}

def RunSubprocess(cmd, cwd=None, env=None):
    '''A more sophistacted version of subprocess.check_call
    which can handle errors in a more informative way.'''
    P = subprocess.Popen(cmd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
    l = True
    while l:
        l = P.stdout.read(1)
        sys.stdout.write(l)
    P.wait()
    if P.returncode:
        raise subprocess.CalledProcessError(returncode=P.returncode,
                cmd=cmd)
        
def DownloadPhotons():
    '''This function is used to download the entire LAT photon library in 
    weekly snapshots and retrive any newly released data that isn't currently present'''
    #Photon data retrival
    cmd = ['wget', '-m', '-P', '/home/b7009348/projects/fermi-data/Weekly_Photons', '-nH', '--cut-dirs=4', '-np', '-e', 'robots=off', 
                       'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/photon/']
    RunSubprocess(cmd)

def DownloadSpacecraft():    
    '''This function is used to download the entire LAT spacecraft library in 
    weekly snapshots and retrive any newly released data that isn't currently present'''
    cmd = ['wget', '-m', '-P', '/home/b7009348/projects/fermi-data/Weekly_Spacecraft', '-nH', '--cut-dirs=4', '-np', '-e', 'robots=off', 
                       'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/spacecraft/']
    RunSubprocess(cmd)
    
def GenFileList():
    '''Creates text files containing lists of the paths to the weekly photon and spacecraft
    data files to be used by the fermi tool when carrying out data analysis'''
    spacecraftlist = sorted(glob.glob('/home/b7009348/projects/fermi-data/Weekly_Spacecraft/weekly/spacecraft/*.fits'))
    with open('/home/b7009348/projects/spacecraft.txt', 'w') as f:
        for i in range(0,len(spacecraftlist)):
            F = astropy.io.fits.open(spacecraftlist[i])
            data = F[1].data
            if not data['DATA_QUAL'].sum() == 0:
                f.write(str(spacecraftlist[i]))
                f.write('\n')

    photonlist = sorted(glob.glob('/home/b7009348/projects/fermi-data/Weekly_Photons/weekly/photon/*.fits'))
    with open('/home/b7009348/projects/photons.txt', 'w') as f:
        for i in range(0,len(photonlist)):
            f.write(str(photonlist[i]))
            f.write('\n')   

def SourceModel(name,xcoord,ycoord):
    '''Creates a model of all sources in region of sky to be fit, necessary for source map
    creation and likelihood analysis
    '''
    photonlist = sorted(glob.glob('/home/b7009348/projects/fermi-data/Weekly_Photons/weekly/photon/*.fits'))
    
    Fstart = astropy.io.fits.open(photonlist[0])        
    Fend = astropy.io.fits.open(photonlist[-1])

    start = Fstart[0].header['TSTART']
    end = Fend[0].header['TSTOP']
    
    if not os.path.exists(name+'_allphotons_filtered.fits'): #checks if file already exists
        gt.filter['evclass'] = 128
        gt.filter['evtype'] = 3
        gt.filter['rad'] = 20
        gt.filter['tmin'] = start
        gt.filter['tmax'] = end
        gt.filter['ra'] = xcoord
        gt.filter['dec'] = ycoord
        gt.filter['zmax'] = 90
        gt.filter['emin'] = 100
        gt.filter['emax'] = 500000
        gt.filter['infile'] = '@photons.txt'
        gt.filter['outfile'] = name+'_allphotons_filtered.fits'
        gt.filter.run()
        
    if not os.path.exists(name+'_allphotons_gti.fits'):
        gt.maketime['scfile'] = '@spacecraft.txt'
        gt.maketime['filter'] = '(DATA_QUAL>0)&&(LAT_CONFIG==1)'
        gt.maketime['roicut'] = 'no'
        gt.maketime['evfile'] = name+'_allphotons_filtered.fits'
        gt.maketime['outfile'] = name+'_allphotons_gti.fits'
        gt.maketime.run()


    mymodel = srcList('gll_psc_v21.xml',name+'_allphotons_gti.fits',name+'_model.xml') #generates the model
    mymodel.makeModel('gll_iem_v07.fits','gll_iem_v07','iso_P8R3_SOURCE_V2_v1.txt','iso_P8R3_SOURCE_V2_v1',normsOnly=True,radLim=5,
                 extDir='home/b7009348/projects/fermi-data/Templates')

    with open(name+'_model_clean.xml', 'wt') as f: #cleans up a bug in the make model code which cause wrong path to template folder
        f.write(re.sub(r'\$\(LATEXTDIR\)/', 
                       '', 
                       open(name+'_model.xml').read()))

    return;

def Select(name,xcoord,ycoord,tmid,tmin,tmax,binsz,chatter):
    '''Cuts a radius and time bin from the complete data set and filters it on energy 
    and radius
    '''
    gt.filter['evclass'] = 128
    gt.filter['evtype'] = 3
    gt.filter['rad'] = 20
    gt.filter['ra'] = xcoord
    gt.filter['dec'] = ycoord
    gt.filter['zmax'] = 90
    gt.filter['tmin'] = tmin
    gt.filter['tmax'] = tmax
    gt.filter['emin'] = 100
    gt.filter['emax'] = 500000
    gt.filter['infile'] = '@photons.txt'
    gt.filter['outfile'] = name+'_filtered-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.filter['chatter'] = chatter
    gt.filter.run()
    return;

def GoodTimeInt(name,tmid,binsz,chatter): 
    '''Performs a good time interval cut to remove data taken at 
    poor times - requires filtered data
    '''
    gt.maketime['scfile'] = '@spacecraft.txt'
    gt.maketime['filter'] = '(DATA_QUAL>0)&&(LAT_CONFIG==1)'
    gt.maketime['roicut'] = 'no'
    gt.maketime['evfile'] = name+'_filtered-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.maketime['outfile'] = name+'_gti-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.maketime['chatter'] = chatter
    gt.maketime.run()
    return;

def CountsMap(name,tmid,xcoord,ycoord,binsz,chatter): 
    '''Generates counts map - requires gti file
    '''
    gt.evtbin['evfile'] = name+'_gti-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.evtbin['scfile'] = '@spacecraft.txt'
    gt.evtbin['outfile'] = name+'_cmap-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.evtbin['algorithm'] = 'CMAP'
    gt.evtbin['emin'] = 100
    gt.evtbin['xref'] = xcoord
    gt.evtbin['yref'] = ycoord
    gt.evtbin['emax'] = 500000
    gt.evtbin['nxpix'] = 150
    gt.evtbin['nypix'] = 150
    gt.evtbin['binsz'] = 0.2
    gt.evtbin['coordsys'] = 'CEL'
    gt.evtbin['axisrot'] = 0.0
    gt.evtbin['proj'] = 'AIT'
    gt.evtbin['rafield'] = 'RA'
    gt.evtbin['decfield'] = 'DEC'
    gt.evtbin['chatter'] = chatter
    gt.evtbin.run()
    return;

def CountsCube(name,tmid,xcoord,ycoord,binsz,chatter):
    '''Generates counts cube - requires gti file
    '''
    gt.evtbin['evfile'] = name+'_gti-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.evtbin['scfile'] = '@spacecraft.txt'
    gt.evtbin['outfile'] = name+'_ccube-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.evtbin['algorithm'] = 'CCUBE'
    gt.evtbin['emin'] = 100
    gt.evtbin['xref'] = xcoord
    gt.evtbin['yref'] = ycoord
    gt.evtbin['emax'] = 500000
    gt.evtbin['nxpix'] = 100
    gt.evtbin['nypix'] = 100
    gt.evtbin['binsz'] = 0.2
    gt.evtbin['coordsys'] = 'CEL'
    gt.evtbin['axisrot'] = 0.0
    gt.evtbin['proj'] = 'AIT'
    gt.evtbin['rafield'] = 'RA'
    gt.evtbin['decfield'] = 'DEC'
    gt.evtbin['ebinalg'] = 'LOG'
    gt.evtbin['enumbins'] = 37
    gt.evtbin['chatter'] = chatter
    gt.evtbin.run()
    return;

def LiveTimeCube(name,tmid,binsz,chatter): 
    '''Generates livetime cube - requires gti file
    '''
    gt.expCube['evfile'] = name+'_gti-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.expCube['scfile'] = 'spacecraft.txt'
    gt.expCube['outfile'] = name+'_ltcube-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.expCube['zmax'] = 90
    gt.expCube['dcostheta'] = 0.025
    gt.expCube['binsz'] = 1
    gt.expCube['chatter'] = chatter
    gt.expCube.run()
    return;

def ExpMap(name,tmid,binsz,chatter):
    '''Generates exposure map - requires gti file and livetime cube
    '''
    gt.expMap['evfile'] = name+'_gti-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.expMap['scfile'] = '@spacecraft.txt'
    gt.expMap['expcube'] = name+'_ltcube-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.expMap['outfile'] = name+'_expMap-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.expMap['irfs'] = 'CALDB'
    gt.expMap['srcrad'] = 30
    gt.expMap['nlong'] = 120
    gt.expMap['nlat'] = 120
    gt.expMap['nenergies'] = 37
    gt.expMap['chatter'] = chatter
    gt.expMap.run()
    return;

def ExpCube(name,tmid,xcoord,ycoord,binsz,chatter):
    '''Generates exposure cube - requires livetime cube
    '''
    gt.gtexpcube2['infile'] = name+'_ltcube-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.gtexpcube2['outfile'] = name+'_expCube-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.gtexpcube2['cmap'] = 'none'
    gt.gtexpcube2['irfs'] = 'P8R3_SOURCE_V2'
    gt.gtexpcube2['nxpix'] = 300
    gt.gtexpcube2['nypix'] = 300
    gt.gtexpcube2['binsz'] = 0.7
    gt.gtexpcube2['coordsys'] = 'CEL'
    gt.gtexpcube2['xref'] = xcoord
    gt.gtexpcube2['yref'] = ycoord
    gt.gtexpcube2['axisrot'] = 0
    gt.gtexpcube2['proj'] = 'AIT'
    gt.gtexpcube2['emin'] = 100
    gt.gtexpcube2['emax'] = 500000
    gt.gtexpcube2['enumbins'] = 37
    gt.gtexpcube2['chatter'] = chatter
    gt.gtexpcube2.run()
    return;

def SourceMap(name,tmid,binsz,chatter): #map of sources in region of sky
    '''Generates a sourcemap - requires counts cube, source model, exposure cube, 
    and livetime cube
    '''
    gt.srcMaps['expcube'] = name+'_ltcube-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.srcMaps['scfile'] = '@spacecraft.txt'
    gt.srcMaps['cmap'] = name+'_ccube-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.srcMaps['srcmdl'] = name+'_model_clean.xml'
    gt.srcMaps['bexpmap'] = name+'_expCube-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.srcMaps['outfile'] = name+'_srcMap-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.srcMaps['irfs'] = 'CALDB'
    gt.srcMaps['ptsrc'] = 'yes'
    gt.srcMaps['chatter'] = chatter
    gt.srcMaps.run()
    return;

def CalcFlux(name,tmid,model_name,binsz,chatter): 
    '''Runs the likelihood analysis using source map, livetime cube, and exposure cube 
    with a NEWMINUIT optimizer
    '''
    obs = BinnedObs(srcMaps=name+'_srcMap-'+str(tmid)+'-'+str(binsz)+'.fits', #binned objects
                expCube=name+'_ltcube-'+str(tmid)+'-'+str(binsz)+'.fits', binnedExpMap=name+'_expCube-'+str(tmid)+'-'+str(binsz)+'.fits', 
                irfs='CALDB')
    like = BinnedAnalysis(obs, name+'_model_clean.xml', optimizer='NEWMINUIT')

    likeobj = pyLike.NewMinuit(like.logLike) #likelihood fitting 
    like.fit(verbosity=chatter,covar=True,optObject=likeobj) #save output to xml

    if likeobj.getRetCode() == 0: #test if newminuit converged 

        like.logLike.writeXml(name+'_output-'+str(tmid)+'-'+str(binsz)+'.xml') #write required data to .txt
        with open('Flux'+str(tmid)+'-'+str(binsz)+'.txt','w') as f:
            f.write(str(tmid)+' '
                        +str(like.flux(model_name, emin=100))+' '+str #need to figure out how to generalize this
                        (like.fluxError(model_name, emin=100)))
            f.write('\n')
        
        #removes large leftover files after flux has been succesfully calculated
        os.remove(name+'_srcMap-'+str(tmid)+'-'+str(binsz)+'.fits') 
        os.remove(name+'_ltcube-'+str(tmid)+'-'+str(binsz)+'.fits')
        os.remove(name+'_gti-'+str(tmid)+'-'+str(binsz)+'.fits')
        os.remove(name+'_filtered-'+str(tmid)+'-'+str(binsz)+'.fits')
        os.remove(name+'_expMap-'+str(tmid)+'-'+str(binsz)+'.fits')
        os.remove(name+'_expCube-'+str(tmid)+'-'+str(binsz)+'.fits')
        os.remove(name+'_cmap-'+str(tmid)+'-'+str(binsz)+'.fits')
        os.remove(name+'_ccube-'+str(tmid)+'-'+str(binsz)+'.fits')

    else:
         print 'Analysis did not converge'
            
    return;

def CalcSingleBin(source,tmid,binsz,chatter=0):
    '''Runs all the fermi tools in sequence to generate the necessary files
    and then runs the likelihood analysis. By default, output is minimal
    '''
    name = source['name']
    model_name = source['model_name']
    xcoord = source['xcoord']
    ycoord = source['ycoord']
    
    tmin = tmid - binsz/2
    tmax = tmid + binsz/2
    
    if os.path.exists('Flux'+str(tmid)+'-'+str(binsz)+'.txt'):
        print 'Bin '+str(tmid)+' has been calculated'
        return;
        
    if not os.path.exists(name+'_filtered-'+str(tmid)+'-'+str(binsz)+'.fits'):
        Select(name,xcoord,ycoord,tmid,tmin,tmax,binsz,chatter)

    if not os.path.exists(name+'_gti-'+str(tmid)+'-'+str(binsz)+'.fits'):
        GoodTimeInt(name,tmid,binsz,chatter)

    if not os.path.exists(name+'_cmap-'+str(tmid)+'-'+str(binsz)+'.fits'):     
        CountsMap(name,tmid,xcoord,ycoord,binsz,chatter)

    if not os.path.exists(name+'_ccube-'+str(tmid)+'-'+str(binsz)+'.fits'):
        CountsCube(name,tmid,xcoord,ycoord,binsz,chatter)

    if not os.path.exists(name+'_ltcube-'+str(tmid)+'-'+str(binsz)+'.fits'):
        LiveTimeCube(name,tmid,binsz,chatter)

    if not os.path.exists(name+'_expMap-'+str(tmid)+'-'+str(binsz)+'.fits'):   
        ExpMap(name,tmid,binsz,chatter)

    if not os.path.exists(name+'_expCube-'+str(tmid)+'-'+str(binsz)+'.fits'):
        ExpCube(name,tmid,xcoord,ycoord,binsz,chatter)

    if not os.path.exists(name+'_srcMap-'+str(tmid)+'-'+str(binsz)+'.fits'):
        SourceMap(name,tmid,binsz,chatter)

    CalcFlux(name,tmid,model_name,binsz,chatter)
        
    return;

def ErrorPass(source,tmid,binsz,chatter=0):
    '''Wrapper for fermi tools function to implement error handling'''
    try:
        CalcSingleBin(source,tmid,binsz,chatter=0)
    except RuntimeError as e:
        print e.message+'\nBin = '+str(tmid)
        pass

def CalcAllBins(poolsize,source,binsz=None):
    '''Function exists as a wrapper for applytools, allowing it to be easily
    parallelized. For single core computing this function is not necessary and
    applytools should be used instead
    '''
    
    name = source['name']
    model_name = source['model_name']
    xcoord = source['xcoord']
    ycoord = source['ycoord']
    binsz = source['rec_binsz']
    
    photonlist = sorted(glob.glob('/home/b7009348/projects/fermi-data/Weekly_Photons/weekly/photon/*.fits'))
    
    Fstart = astropy.io.fits.open(photonlist[0])        
    Fend = astropy.io.fits.open(photonlist[-1])

    start = Fstart[0].header['TSTART']
    end = Fend[0].header['TSTOP']
    
    diff = end - start #time of fermi mission
    numbins = np.ceil(diff/binsz) #number of bins
    i_array = np.arange(numbins)
    tmid = start + (i_array+0.5)*binsz
    
    with open('StopTime.txt', 'w') as f:
        f.write(str(end))
            
    global ft
    def ft(tmid):
        
        return ErrorPass(source,tmid,binsz,chatter=0)

    pool = mp.Pool(poolsize)
    pool.map(ft, tmid) #runs the function in parallel on multiple cpus
    pool.close()              
    return;

def PlotCurve(source):
    '''Uses flux values generated from likelihood analysis to plot a
    lightcurve'''
    name = source['name']
    binsz = source['rec_binsz']
    
    spy = 31536000
    start_dec_yrs = 2008.0 + 8.0/12.0 + 4.0/365.0

    files = sorted(glob.glob('Flux*.txt'))
    data = np.zeros((len(files),3))

    for i in range(0,len(files)):
        data[i,:] = np.loadtxt(str(files[i]))
    
    time = (data[:,0] - data[0,0])/spy + start_dec_yrs
    flux = data[:,1]/1E-8
    error = data[:,2]/1E-8

    plt.errorbar(time,flux,yerr=error,marker='',color='red',ecolor='grey',drawstyle='steps-mid',capsize=3)
    plt.ylim(ymin=0)
    plt.xlabel('Time (years)')
    plt.ylabel('Flux ($10^{-8}cm^{-2}s^{-1}$)')
    plt.gcf().set_size_inches(8,6)
    plt.savefig(name+'LightCurve-'+str(binsz)+'.pdf')
    
    return;

def UpdateCurves(poolsize, source, binsz=None):
    
    fluxes = sorted(glob.glob('Flux*.txt'))
    photonlist = sorted(glob.glob('/home/b7009348/projects/fermi-data/Weekly_Photons/weekly/photon/*.fits'))
       
    Fend = astropy.io.fits.open(photonlist[-1])
    end = Fend[0].header['TSTOP']
    
    with open('StopTime.txt','r') as f:
        old_end = float(f.readline())

    if old_end != end:
        print 'New data found, recomputing final bins'
        os.remove(fluxes[-1])
        CalcAllBins(poolsize,source,binsz=None)
    else:
        print 'No new data present'
    
    return;