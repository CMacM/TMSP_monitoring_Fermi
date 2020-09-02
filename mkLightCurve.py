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
from astropy.coordinates import SkyCoord

source_list = [{'name' : 'J1023', 'model_name' : '4FGL J1023.7+0038', 'xcoord' : 155.946, 'ycoord' : 0.645, 'rec_binsz' : 2000000, 'catalogue_name' : 'PSR J1023+0038'},
{'name' : 'J12270', 'model_name' : '4FGL J1228.0-4853', 'xcoord' : 186.995, 'ycoord' : -48.8952, 'rec_binsz' : 4000000, 'catalogue_name' : 'XSS J12270-4859'},
{'name' : 'J18245', 'model_name' : '4FGL J1824.6-2452', 'xcoord' : 276.135, 'ycoord' : -24.8688, 'rec_binsz' : 12000000, 'catalogue_name' : 'IGR J18245-2452'},
{'name' : 'J0427', 'model_name' : '4FGL J0427.8-6704', 'xcoord' : 66.958, 'ycoord' : -67.078, 'rec_binsz' : 4000000, 'catalogue_name' : '3FGL J0427-6704'},
{'name' : 'J1723', 'model_name' : 'plchldr', 'xcoord' : 260.846, 'ycoord' : -28.633, 'rec_binsz' : 4000000, 'catalogue_name' : 'PSR J1723-2837'},
{'name' : 'J2215', 'model_name' : '4FGL J2215.6+5135', 'xcoord' : 333.888, 'ycoord' : 51.593, 'rec_binsz' : 4000000, 'catalogue_name' : 'PSR J2215+5135'},
{'name' : 'J1544', 'model_name' : '4FGL J1544.5-1126', 'xcoord' : 236.058, 'ycoord' : -11.412, 'rec_binsz' : 4000000, 'catalogue_name' : 'PSR J1544-1125'},
{'name' : 'J0212', 'model_name' : '4FGL J0212.1+5321', 'xcoord' : 33.042, 'ycoord' : 53.344, 'rec_binsz' : 4000000, 'catalogue_name' : 'PSR J0212+5320'},
{'catalogue_name': 'PSR J1417-4404', 'model_name': '4FGL J1417.6-4403', 'name': 'J1417', 'rec_binsz': 4000000, 'xcoord': 214.321, 'ycoord': -44.078},
{'catalogue_name': 'CXOU J1109-6502', 'model_name': 'plchldr', 'name': 'J1109', 'rec_binsz': 4000000, 'xcoord': 167.358, 'ycoord': -65.04},
{'catalogue_name': 'PSR J1746-2844','model_name': '4FGL J1746.4-2852','name': 'J1746','rec_binsz': 4000000,'xcoord': 266.637,'ycoord': -28.741},
{'catalogue_name': 'CXOGlb J1748-2446','model_name': '"4FGL J1748.0-2446"','name': 'J1748','rec_binsz': 4000000,'xcoord': 267.017,'ycoord': -24.778}]

def retrieve_source_data(source):
    i = 0
    while i < len(source_list):
        if source_list[i]['name'] == source:
            return source_list[i]
        else:
            i = i + 1

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
    cmd = ['wget', '--quiet', '-m', '-P', '/home/b7009348/projects/fermi-data/Weekly_Photons', '-nH', '--cut-dirs=4', '-np', '-e',
           'robots=off', 'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/photon/']
    RunSubprocess(cmd)

def DownloadSpacecraft():    
    '''This function is used to download the entire LAT spacecraft library in 
    weekly snapshots and retrive any newly released data that isn't currently present'''
    cmd = ['wget', '--quiet', '-m', '-P', '/home/b7009348/projects/fermi-data/Weekly_Spacecraft', '-nH', '--cut-dirs=4', '-np', '-e',
           'robots=off', 'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/spacecraft/']
    RunSubprocess(cmd)
    
def GenFileList(source):
    '''Creates text files containing lists of the paths to the weekly photon and spacecraft
    data files to be used by the fermi tool when carrying out data analysis'''

    name = retrieve_source(source)['name']
    
    spacecraftlist = sorted(glob.glob('/home/b7009348/projects/fermi-data/Weekly_Spacecraft/weekly/spacecraft/*.fits'))
    with open('/home/b7009348/projects/fermi-data/'+name+'_Weekly/spacecraft.txt', 'w') as f:
        for i in range(0,len(spacecraftlist)):
            F = astropy.io.fits.open(spacecraftlist[i])
            data = F[1].data
            if not data['DATA_QUAL'].sum() == 0:
                f.write(str(spacecraftlist[i]))
                f.write('\n')

    photonlist = sorted(glob.glob('/home/b7009348/projects/fermi-data/Weekly_Photons/weekly/photon/*.fits'))
    with open('/home/b7009348/projects/fermi-data/'+name+'_Weekly/photons.txt', 'w') as f:
        for i in range(0,len(photonlist)):
            f.write(str(photonlist[i]))
            f.write('\n')   

def SourceModel(source):
    '''Creates a model of all sources in region of sky to be fit, necessary for source map
    creation and likelihood analysis
    '''

    name = retrieve_source(source)['name']
    xcoord = retrieve_source(source)['xcoord']
    ycoord = retrieve_source(source)['ycoord']
    
    os.chdir('/home/b7009348/projects/fermi-data/'+name+'_Weekly')
    
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


    mymodel = srcList('/home/b7009348/projects/fermi-data/gll_psc_v21.xml',name+'_allphotons_gti.fits',name+'_model.xml') #generates the model
    mymodel.makeModel('/home/b7009348/projects/fermi-data/gll_iem_v07.fits','gll_iem_v07','/home/b7009348/projects/fermi-data/iso_P8R3_SOURCE_V2_v1.txt','iso_P8R3_SOURCE_V2_v1',normsOnly=True,radLim=5,
                 extDir='/home/b7009348/projects/fermi-data/')

    with open(name+'_model_clean.xml', 'wt') as f: #cleans up a bug in the make model code which cause wrong path to template folder
        f.write(re.sub(r'\$\(LATEXTDIR\)/', 
                       '', 
                       open(name+'_model.xml').read()))


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
    gt.filter['outfile'] = NameGen('filtered', name, tmid, binsz)
    gt.filter['chatter'] = chatter
    gt.filter.run()


def GoodTimeInt(name,tmid,binsz,chatter): 
    '''Performs a good time interval cut to remove data taken at 
    poor times - requires filtered data
    '''
    gt.maketime['scfile'] = '@spacecraft.txt'
    gt.maketime['filter'] = '(DATA_QUAL>0)&&(LAT_CONFIG==1)'
    gt.maketime['roicut'] = 'no'
    gt.maketime['evfile'] = NameGen('filtered', name, tmid, binsz)
    gt.maketime['outfile'] = NameGen('gti', name, tmid, binsz)
    gt.maketime['chatter'] = chatter
    gt.maketime.run()


def CountsMap(name,tmid,xcoord,ycoord,binsz,chatter): 
    '''Generates counts map - requires gti file
    '''
    gt.evtbin['evfile'] = NameGen('gti', name, tmid, binsz)
    gt.evtbin['scfile'] = '@spacecraft.txt'
    gt.evtbin['outfile'] = NameGen('cmap', name, tmid, binsz)
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


def CountsCube(name,tmid,xcoord,ycoord,binsz,chatter):
    '''Generates counts cube - requires gti file
    '''
    gt.evtbin['evfile'] = NameGen('gti', name, tmid, binsz)
    gt.evtbin['scfile'] = '@spacecraft.txt'
    gt.evtbin['outfile'] = NameGen('ccube', name, tmid, binsz)
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


def LiveTimeCube(name,tmid,binsz,chatter): 
    '''Generates livetime cube - requires gti file
    '''
    gt.expCube['evfile'] = NameGen('gti', name, tmid, binsz)
    gt.expCube['scfile'] = 'spacecraft.txt'
    gt.expCube['outfile'] = NameGen('ltcube', name, tmid, binsz)
    gt.expCube['zmax'] = 90
    gt.expCube['dcostheta'] = 0.025
    gt.expCube['binsz'] = 1
    gt.expCube['chatter'] = chatter
    gt.expCube.run()


def ExpMap(name,tmid,binsz,chatter):
    '''Generates exposure map - requires gti file and livetime cube
    '''
    gt.expMap['evfile'] = NameGen('gti', name, tmid, binsz)
    gt.expMap['scfile'] = '@spacecraft.txt'
    gt.expMap['expcube'] = NameGen('ltcube', name, tmid, binsz)
    gt.expMap['outfile'] = NameGen('expMap', name, tmid, binsz)
    gt.expMap['irfs'] = 'CALDB'
    gt.expMap['srcrad'] = 30
    gt.expMap['nlong'] = 120
    gt.expMap['nlat'] = 120
    gt.expMap['nenergies'] = 37
    gt.expMap['chatter'] = chatter
    gt.expMap.run()


def ExpCube(name,tmid,xcoord,ycoord,binsz,chatter):
    '''Generates exposure cube - requires livetime cube
    '''
    gt.gtexpcube2['infile'] = NameGen('ltcube', name, tmid, binsz)
    gt.gtexpcube2['outfile'] = NameGen('expCube', name, tmid, binsz)
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


def SourceMap(name,tmid,binsz,chatter): #map of sources in region of sky
    '''Generates a sourcemap - requires counts cube, source model, exposure cube, 
    and livetime cube
    '''
    gt.srcMaps['expcube'] = NameGen('ltcube', name, tmid, binsz)
    gt.srcMaps['scfile'] = '@spacecraft.txt'
    gt.srcMaps['cmap'] = NameGen('ccube', name, tmid, binsz)
    gt.srcMaps['srcmdl'] = '/home/b7009348/projects/fermi-data/'+name+'_Weekly/'+name+'_model_clean.xml'
    gt.srcMaps['bexpmap'] = NameGen('expCube', name, tmid, binsz)
    gt.srcMaps['outfile'] = NameGen('srcMap', name, tmid, binsz)
    gt.srcMaps['irfs'] = 'CALDB'
    gt.srcMaps['ptsrc'] = 'yes'
    gt.srcMaps['chatter'] = chatter
    gt.srcMaps.run()


def CalcFlux(name,tmid,model_name,binsz,new_last_photon,chatter): 
    '''Runs the likelihood analysis using source map, livetime cube, and exposure cube 
    with a NEWMINUIT optimizer
    '''
    obs = BinnedObs(srcMaps=NameGen('srcMap', name, tmid, binsz), #binned objects
                expCube=NameGen('ltcube', name, tmid, binsz), binnedExpMap=NameGen('expCube', name, tmid, binsz), 
                irfs='CALDB')
    like = BinnedAnalysis(obs, '/home/b7009348/projects/fermi-data/'+name+'_Weekly/'+name+'_model_clean.xml', optimizer='NEWMINUIT')

    likeobj = pyLike.NewMinuit(like.logLike) #likelihood fitting 
    like.fit(verbosity=chatter,covar=True,optObject=likeobj) #save output to xml

    if likeobj.getRetCode() == 0: #test if newminuit converged 

        like.logLike.writeXml(NameGen('output', name, tmid, binsz)) #write required data to .txt need to save last photon in CalcFlux
        with open('/home/b7009348/projects/fermi-data/'+name+'_Weekly/Flux'+str(tmid)+'-'+str(binsz)+'.txt','w') as f:
            f.write(str(tmid)+' '
                        +str(like.flux(model_name, emin=100))+' '+str #need to figure out how to generalize this
                        (like.fluxError(model_name, emin=100))+' '+str(new_last_photon))
            f.write('\n')
        
        #removes large leftover files after flux has been succesfully calculated
        os.remove(NameGen('srcMap', name, tmid, binsz)) 
        os.remove(NameGen('ltcube', name, tmid, binsz))
        os.remove(NameGen('gti', name, tmid, binsz))
        os.remove(NameGen('filtered', name, tmid, binsz))
        os.remove(NameGen('expMap', name, tmid, binsz))
        os.remove(NameGen('expCube', name, tmid, binsz))
        os.remove(NameGen('cmap', name, tmid, binsz))
        os.remove(NameGen('ccube', name, tmid, binsz))

    else:
         print 'Analysis did not converge'
            


def CalcSingleBin(source,tmid,binsz,end,chatter=0):
    '''Runs all the fermi tools in sequence to generate the necessary files
    and then runs the likelihood analysis. By default, output is minimal
    '''
    name = retrieve_source(source)['name']
    model_name = retrieve_source(source)['model_name']
    xcoord = retrieve_source(source)['xcoord']
    ycoord = retrieve_source(source)['ycoord']
    
    tmin = tmid - binsz/2
    tmax = tmid + binsz/2
    
    new_last_photon = min(tmax,end)
    
    if os.path.exists('Flux'+str(tmid)+'-'+str(binsz)+'.txt'):
        old_last_photon = np.loadtxt('Flux'+str(tmid)+'-'+str(binsz)+'.txt')[3]
        if old_last_photon < new_last_photon:
            print 'New data found, recalculating bin'+str(tmid)
            os.remove('Flux'+str(tmid)+'-'+str(binsz)+'.txt')
        elif old_last_photon == new_last_photon: #change to within certain tolerance 
            print 'Bin '+str(tmid)+' has been calculated'
            return;
        
    if not os.path.exists(NameGen('filtered', name, tmid, binsz)):
        Select(name,xcoord,ycoord,tmid,tmin,tmax,binsz,chatter)

    if not os.path.exists(NameGen('gti', name, tmid, binsz)):
        GoodTimeInt(name,tmid,binsz,chatter)

    if not os.path.exists(NameGen('cmap', name, tmid, binsz)):     
        CountsMap(name,tmid,xcoord,ycoord,binsz,chatter)

    if not os.path.exists(NameGen('ccube', name, tmid, binsz)):
        CountsCube(name,tmid,xcoord,ycoord,binsz,chatter)

    if not os.path.exists(NameGen('ltcube', name, tmid, binsz)):
        LiveTimeCube(name,tmid,binsz,chatter)

    if not os.path.exists(NameGen('expMap', name, tmid, binsz)):   
        ExpMap(name,tmid,binsz,chatter)

    if not os.path.exists(NameGen('expCube', name, tmid, binsz)):
        ExpCube(name,tmid,xcoord,ycoord,binsz,chatter)

    if not os.path.exists(NameGen('srcMap', name, tmid, binsz)):
        SourceMap(name,tmid,binsz,chatter)

    CalcFlux(name,tmid,model_name,binsz,new_last_photon,chatter)

def ErrorPass(source,tmid,binsz,end,chatter=0):
    '''Wrapper for fermi tools function to implement error handling'''
    try:
        CalcSingleBin(source,tmid,binsz,end,chatter=0)
    except RuntimeError as e:
        print e.message+'\nBin = '+str(tmid)
        pass

def CalcAllBins(poolsize,source,binsz=None):
    '''Function exists as a wrapper for applytools, allowing it to be easily
    parallelized. For single core computing this function is not necessary and
    applytools should be used instead
    '''
    
    name = retrieve_source(source)['name']
    model_name = retrieve_source(source)['model_name']
    xcoord = retrieve_source(source)['xcoord']
    ycoord = retrieve_source(source)['ycoord']
    
    if binsz is None:
        binsz = retrieve_source(source)['rec_binsz']
    
    photonlist = sorted(glob.glob('/home/b7009348/projects/fermi-data/Weekly_Photons/weekly/photon/*.fits'))
    
    Fstart = astropy.io.fits.open(photonlist[0])        
    Fend = astropy.io.fits.open(photonlist[-1])

    start = Fstart[0].header['TSTART']
    end = Fend[0].header['TSTOP']
    
    diff = end - start #time of fermi mission
    numbins = np.ceil(diff/binsz) #number of bins
    i_array = np.arange(numbins)
    tmids = start + (i_array+0.5)*binsz
            
    global ft
    def ft(tmid):
        return ErrorPass(source,tmid,binsz,end,chatter=0)
    
    if poolsize > 1:
        pool = mp.Pool(poolsize)
        pool.map(ft, tmids) #runs the function in parallel on multiple cpus
        pool.close()
    else:
        map(ft, tmids)
    

def PlotCurve(source):
    '''Uses flux values generated from likelihood analysis to plot a
    lightcurve'''
    name = retrieve_source(source)['name']
    binsz = retrieve_source(source)['rec_binsz']
    catalogue_name = retrieve_source(source)['catalogue_name']
    
    spy = 31536000
    start_dec_yrs = 2008.0 + 8.0/12.0 + 4.0/365.0

    files = sorted(glob.glob('/home/b7009348/projects/fermi-data/'+name+'_Weekly/Flux*'+str(binsz)+'.txt'))
    data = np.zeros((len(files),4))

    for i in range(0,len(files)):
        data[i,:] = np.loadtxt(str(files[i]))
    
    time = (data[:,0] - data[0,0])/spy + start_dec_yrs
    flux = data[:,1]/1E-8
    error = data[:,2]/1E-8

    plt.errorbar(time,flux,yerr=error,marker='',color='red',ecolor='grey',drawstyle='steps-mid',capsize=3)
    plt.ylim(ymin=0)
    plt.title(catalogue_name+' Light Curve')
    plt.xlabel('Time (years)')
    plt.ylabel('Flux ($10^{-8}cm^{-2}s^{-1}$)')
    plt.gcf().set_size_inches(8,6)
    plt.savefig('/home/b7009348/projects/light-curves/'+name+'LightCurve-'+str(binsz)+'.pdf')

def UpdateCurves(poolsize):
    
    DownloadPhotons()
    DownloadSpacecraft()
    
    iterables = [i for i in range(len(source_list))]
    
    def uc(iterable):
        GenFileList(source=source_list[iterable]['name'])
        CalcAllBins(poolsize=1, source=source_list[iterable]['name'])
    
    pool = mp.Pool(poolsize)
    pool.map(uc, iterables)
    pool.close()
        
    
def NameGen(file, name, tmid, binsz):
        return '/home/b7009348/projects/fermi-data/'+name+'_Weekly/'+name+'_'+file+'-'+str(tmid)+'-'+str(binsz)+'.fits'
    
    
def add_new_source():
    full_name = raw_input('Enter name in format -> PSR J 00 00 00 +/-00 00 00')
    name_list = full_name.split()
    tag = name_list[0]
    calendar = name_list[1]
    RAh = name_list[2]
    RAm = name_list[3]
    RAs = name_list[4]
    DECd = name_list[5]
    DECm = name_list[6]
    DECs = name_list[7]
    catalogue_name = tag+' '+calendar+RAh+RAm+DECd+DECm
    name = calendar+RAh+RAm
    print catalogue_name
    
    c = SkyCoord(RAh+'h'+RAm+'m'+RAs+'s', DECd+'d'+DECm+'m'+DECs+'s', frame='icrs')
    RA = round(c.ra.deg, 3)
    DEC = round(c.dec.deg, 3)
    print 'RA = ', RA, 'DEC = ', DEC
    
    return {'name' : name, 'model_name' : 'plchldr', 'xcoord' : RA, 'ycoord' : DEC, 'rec_binsz' : 4000000, 'catalogue_name' : catalogue_name}
    