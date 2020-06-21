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

def generatemodel(name,templateDir): #creates a model of all sources in region of sky to be fit
    
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

def select(name,tmid,tmin,tmax,binsz): #gtselect to cut a specific time bin from the data
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
    gt.filter.run()
    return;

def goodtimeint(name,tmid,binsz): #good time interval cut removes data taken at poor times
    gt.maketime['scfile'] = 'spacecraft.fits'
    gt.maketime['filter'] = '(DATA_QUAL>0)&&(LAT_CONFIG==1)'
    gt.maketime['roicut'] = 'no'
    gt.maketime['evfile'] = name+'_filtered-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.maketime['outfile'] = name+'_gti-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.maketime.run()
    return;

def countsmap(name,tmid,xcoord,ycoord,binsz): #creates a counts map of the photons from the region of sky
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
    gt.evtbin.run()
    return;

def countscube(name,tmid,xcoord,ycoord,binsz): #creates a 3D counts map
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
    gt.evtbin.run()
    return;

def livetimecube(name,tmid,binsz): #creates a livetime cube
    gt.expCube['evfile'] = name+'_gti-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.expCube['scfile'] = 'spacecraft.fits'
    gt.expCube['outfile'] = name+'_ltcube-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.expCube['zmax'] = 90
    gt.expCube['dcostheta'] = 0.025
    gt.expCube['binsz'] = 1
    gt.expCube.run()
    return;

def expmap(name,tmid,binsz): #creates an exposure map
    gt.expMap['evfile'] = name+'_gti-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.expMap['scfile'] = 'spacecraft.fits'
    gt.expMap['expcube'] = name+'_ltcube-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.expMap['outfile'] = name+'_expMap-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.expMap['irfs'] = 'CALDB'
    gt.expMap['srcrad'] = 20
    gt.expMap['nlong'] = 120
    gt.expMap['nlat'] = 120
    gt.expMap['nenergies'] = 37
    gt.expMap.run()
    return;

def expcube(name,tmid,xcoord,ycoord,binsz): #3D exposure map
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
    gt.gtexpcube2.run()
    return;

def srcmap(name,tmid,binsz): #map of sources in region of sky
    gt.srcMaps['expcube'] = name+'_ltcube-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.srcMaps['scfile'] = 'spacecraft.fits'
    gt.srcMaps['cmap'] = name+'_ccube-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.srcMaps['srcmdl'] = name+'_model_clean.xml'
    gt.srcMaps['bexpmap'] = name+'_expCube-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.srcMaps['outfile'] = name+'_srcMap-'+str(tmid)+'-'+str(binsz)+'.fits'
    gt.srcMaps['irfs'] = 'CALDB'
    gt.srcMaps['ptsrc'] = 'yes'
    gt.srcMaps.run()
    return;

def calcflux(name,tmid,model_name,binsz): #this function actually does the likelihood analysis using all the files generated by
    #the previous functions
    obs = BinnedObs(srcMaps=name+'_srcMap-'+str(tmid)+'-'+str(binsz)+'.fits', #binned objects
                expCube=name+'_ltcube-'+str(tmid)+'-'+str(binsz)+'.fits', binnedExpMap=name+'_expCube-'+str(tmid)+'-'+str(binsz)+'.fits', 
                irfs='CALDB')
    like = BinnedAnalysis(obs, name+'_model_clean.xml', optimizer='NEWMINUIT')

    likeobj = pyLike.NewMinuit(like.logLike) #likelihood fitting 
    like.fit(verbosity=0,covar=True,optObject=likeobj) #save output to xml

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
        #remove unwanted large files

    else:
         print 'Analysis did not converge'
    return;

def applytools(name,model_name,xcoord,ycoord,start,binsz,i,chatter=2):
#this function applys all the previous ones in sequence for a complete anaylsis on a source
    tmin = start + (i-1)*binsz
    tmax = start + i*binsz
    tmid = (tmax+tmin)/2
    
    if os.path.exists('Flux'+str(tmid)+'-'+str(binsz)+'.txt'):
        print 'Bin'+str(tmid)+'has been calculated'
    else:   
        if os.path.exists(name+'_filtered-'+str(tmid)+'-'+str(binsz)+'.fits'):
            pass
        else:
            select(name,tmid,tmin,tmax,binsz)
        
        if os.path.exists(name+'_gti-'+str(tmid)+'-'+str(binsz)+'.fits'):
            pass
        else:
            goodtimeint(name,tmid,binsz)
        
        if os.path.exists(name+'_cmap-'+str(tmid)+'-'+str(binsz)+'.fits'):
            pass
        else:
            countsmap(name,tmid,xcoord,ycoord,binsz)
        
        if os.path.exists(name+'_ccube-'+str(tmid)+'-'+str(binsz)+'.fits'):
            pass
        else:
            countscube(name,tmid,xcoord,ycoord,binsz)
        
        if os.path.exists(name+'_ltcube-'+str(tmid)+'-'+str(binsz)+'.fits'):
            pass
        else:
            livetimecube(name,tmid,binsz)
        
        if os.path.exists(name+'_expMap-'+str(tmid)+'-'+str(binsz)+'.fits'):
            pass
        else:
            expmap(name,tmid,binsz)
        
        if os.path.exists(name+'_expCube-'+str(tmid)+'-'+str(binsz)+'.fits'):
            pass
        else:
            expcube(name,tmid,xcoord,ycoord,binsz)
        
        if os.path.exists(name+'_srcMap-'+str(tmid)+'-'+str(binsz)+'.fits'):
            pass
        else:
            srcmap(name,tmid,binsz)
        
        calcflux(name,tmid,model_name,binsz)
        
    return;

#wrapper function to allow applytools to be passed to multi processing with different bins
def generateflux(poolsize,name,model_name,xcoord,ycoord,start,binsz,i):
    global ft
    def ft(i):
        return applytools(name,model_name,xcoord,ycoord,start,binsz,i,chatter=0)

    pool = mp.Pool(poolsize)
    pool.map(ft, i) #runs the function in parallel on multiple cpus
    pool.close()
    return;

#function simply reads in flux data and plots it on a curve
def plotcurve(name,binsz):    
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