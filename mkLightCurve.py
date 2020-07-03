import numpy as np                  #import all necessary modules
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

def generatemodel(name,templateDir):
    '''Creates a model of all sources in region of sky to be fit, necessary for source map
    creation and likelihood analysis
    '''
    if os.path.exists(name+'_allphotons_gti.fits'): #checks if file already exists
        pass
    else: #uses fermi tools to make neccesary cuts to data
        gt.filter['evclass'] = 128
        gt.filter['evtype'] = 3
        gt.filter['rad'] = 20
        gt.filter['zmax'] = 90
        gt.filter['emin'] = 100
        gt.filter['emax'] = 500000
        gt.filter['infile'] = '@photons.txt'
        gt.filter['outfile'] = name+'_allphotons_filtered.fits'
        gt.filter.run()
    
        gt.maketime['scfile'] = 'spacecraft.fits'
        gt.maketime['filter'] = '(DATA_QUAL>0)&&(LAT_CONFIG==1)'
        gt.maketime['roicut'] = 'no'
        gt.maketime['evfile'] = name+'_allphotons_filtered.fits'
        gt.maketime['outfile'] = name+'_allphotons_gti.fits'
        gt.maketime.run()


    mymodel = srcList('gll_psc_v21.xml',name+'_allphotons_gti.fits',name+'_model.xml') #generates the model
    mymodel.makeModel('gll_iem_v07.fits','gll_iem_v07','iso_P8R3_SOURCE_V2_v1.txt','iso_P8R3_SOURCE_V2_v1',normsOnly=True,radLim=5,
                 extDir=templateDir)

    with open(name+'_model_clean.xml', 'wt') as f: #cleans up a bug in the make model code which cause wrong path to template folder
        f.write(re.sub(r'\$\(LATEXTDIR\)/', 
                       '', 
                       open(name+'_model.xml').read()))

    return;

def select(name,tmid,tmin,tmax,binsz,chatter):
    '''Cuts a time bin from the complete data set and filters it on energy 
    and radius
    '''
    gt.filter['evclass'] = 128
    gt.filter['evtype'] = 3
    gt.filter['rad'] = 20
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

def goodtimeint(name,tmid,binsz,chatter): 
    '''Performs a good time interval cut to remove data taken at 
    poor times - requires filtered data
    '''
    gt.maketime['scfile'] = 'spacecraft.fits'
    gt.maketime['filter'] = '(DATA_QUAL>0)&&(LAT_CONFIG==1)'
    gt.maketime['roicut'] = 'no'
    gt.maketime['evfile'] = name+'_filtered-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.maketime['outfile'] = name+'_gti-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.maketime['chatter'] = chatter
    gt.maketime.run()
    return;

def countsmap(name,tmid,xcoord,ycoord,binsz,chatter): 
    '''Generates counts map - requires gti file
    '''
    gt.evtbin['evfile'] = name+'_gti-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.evtbin['scfile'] = 'spacecraft.fits'
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

def countscube(name,tmid,xcoord,ycoord,binsz,chatter):
    '''Generates counts cube - requires gti file
    '''
    gt.evtbin['evfile'] = name+'_gti-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.evtbin['scfile'] = 'spacecraft.fits'
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

def livetimecube(name,tmid,binsz,chatter): 
    '''Generates livetime cube - requires gti file
    '''
    gt.expCube['evfile'] = name+'_gti-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.expCube['scfile'] = 'spacecraft.fits'
    gt.expCube['outfile'] = name+'_ltcube-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.expCube['zmax'] = 90
    gt.expCube['dcostheta'] = 0.025
    gt.expCube['binsz'] = 1
    gt.expCube['chatter'] = chatter
    gt.expCube.run()
    return;

def expmap(name,tmid,binsz,chatter):
    '''Generates exposure map - requires gti file and livetime cube
    '''
    gt.expMap['evfile'] = name+'_gti-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.expMap['scfile'] = 'spacecraft.fits'
    gt.expMap['expcube'] = name+'_ltcube-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.expMap['outfile'] = name+'_expMap-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.expMap['irfs'] = 'CALDB'
    gt.expMap['srcrad'] = 20
    gt.expMap['nlong'] = 120
    gt.expMap['nlat'] = 120
    gt.expMap['nenergies'] = 37
    gt.expMap['chatter'] = chatter
    gt.expMap.run()
    return;

def expcube(name,tmid,xcoord,ycoord,binsz,chatter):
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

def srcmap(name,tmid,binsz,chatter): #map of sources in region of sky
    '''Generates a sourcemap - requires counts cube, source model, exposure cube, 
    and livetime cube
    '''
    gt.srcMaps['expcube'] = name+'_ltcube-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.srcMaps['scfile'] = 'spacecraft.fits'
    gt.srcMaps['cmap'] = name+'_ccube-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.srcMaps['srcmdl'] = name+'_model_clean.xml'
    gt.srcMaps['bexpmap'] = name+'_expCube-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.srcMaps['outfile'] = name+'_srcMap-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.srcMaps['irfs'] = 'CALDB'
    gt.srcMaps['ptsrc'] = 'yes'
    gt.srcMaps['chatter'] = chatter
    gt.srcMaps.run()
    return;

def calcflux(name,tmid,model_name,binsz,chatter): 
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

def applytools(name,model_name,xcoord,ycoord,start,binsz,i,chatter=0):
    '''Runs all the fermi tools in sequence to generate the necessary files
    and then runs the likelihood analysis. By default, output is minimal
    '''
    tmin = start + (i-1)*binsz
    tmax = start + i*binsz
    tmid = (tmax+tmin)/2
    
    if os.path.exists('Flux'+str(tmid)+'-'+str(binsz)+'.txt'):
        print 'Bin '+str(tmid)+' has been calculated'
        return
    
    try:
        
        if os.path.exists(name+'_filtered-'+str(tmid)+'-'+str(binsz)+'.fits'):
            pass
        else:
            select(name,tmid,tmin,tmax,binsz,chatter)

        if os.path.exists(name+'_gti-'+str(tmid)+'-'+str(binsz)+'.fits'):
            pass
        else:
            goodtimeint(name,tmid,binsz,chatter)

        if os.path.exists(name+'_cmap-'+str(tmid)+'-'+str(binsz)+'.fits'):
            pass
        else:
            countsmap(name,tmid,xcoord,ycoord,binsz,chatter)

        if os.path.exists(name+'_ccube-'+str(tmid)+'-'+str(binsz)+'.fits'):
            pass
        else:
            countscube(name,tmid,xcoord,ycoord,binsz,chatter)

        if os.path.exists(name+'_ltcube-'+str(tmid)+'-'+str(binsz)+'.fits'):
            pass
        else:
            livetimecube(name,tmid,binsz,chatter)

        if os.path.exists(name+'_expMap-'+str(tmid)+'-'+str(binsz)+'.fits'):
            pass
        else:
            expmap(name,tmid,binsz,chatter)

        if os.path.exists(name+'_expCube-'+str(tmid)+'-'+str(binsz)+'.fits'):
            pass
        else:
            expcube(name,tmid,xcoord,ycoord,binsz,chatter)

        if os.path.exists(name+'_srcMap-'+str(tmid)+'-'+str(binsz)+'.fits'):
            pass
        else:
            srcmap(name,tmid,binsz,chatter)

        calcflux(name,tmid,model_name,binsz,chatter)
        
    except RuntimeError as error:
        
        print error
        pass
        
    return;


def generateflux(poolsize,name,model_name,xcoord,ycoord,start,end,binsz):
    '''Function exists as a wrapper for applytools, allowing it to be easily
    parallelized. For single core computing this function is not necessary and
    applytools should be used instead
    '''
    diff = end - start #time of fermi mission
    numbins = diff/binsz #number of bins
    i = list(range(1,numbins+1))
    
    global ft
    def ft(i):
        return applytools(name,model_name,xcoord,ycoord,start,binsz,i,chatter=0)

    pool = mp.Pool(poolsize)
    pool.map(ft, i) #runs the function in parallel on multiple cpus
    pool.close()
    return;

def plotcurve(name,binsz):
    '''Uses flux values generated from likelihood analysis to plot a
    lightcurve'''
    spy = 31536000
    start_dec_yrs = 2008.0 + 8.0/12.0 + 4.0/365.0

    files = sorted(glob.glob('Flux*'+str(binsz)+'.txt'))
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