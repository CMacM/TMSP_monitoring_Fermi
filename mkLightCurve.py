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
import warnings
from scipy.stats import chi2, norm
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'cmr10'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 16

code_dir = os.path.dirname(os.path.realpath(__file__))

data_dir = code_dir+'/fermi-data'

if not os.path.exists(data_dir+'/Weekly_Photons'):
    os.mkdir(data_dir+'/Weekly_Photons')
    
if not os.path.exists(data_dir+'/Weekly_Spacecraft'):
    os.mkdir(data_dir+'/Weekly_Spacecraft')
    
template_dir = data_dir

source_list = [{'name' : 'J1023', 'model_name' : '4FGL J1023.7+0038', 'xcoord' : 155.946, 'ycoord' : 0.645, 'rec_binsz' : 2000000, 'catalogue_name' : 'PSR J1023+0038', 'start_time' : 394557417.0, 'min_error' : 0.9750000000000008},
               
{'name' : 'J12270', 'model_name' : '4FGL J1228.0-4853', 'xcoord' : 186.995, 'ycoord' : -48.8952, 'rec_binsz' : 4000000, 'catalogue_name' : 'XSS J12270-4859', 'start_time' : 377557417.0, 'min_error' : 0.5710000000000004},
               
{'name' : 'J18245', 'model_name' : '4FGL J1824.6-2452', 'xcoord' : 276.135, 'ycoord' : -24.8688, 'rec_binsz' : 12000000, 'catalogue_name' : 'IGR J18245-2452', 'start_time' : 377557417.0, 'min_error' : 0.3270000000000002},
               
{'name' : 'J0427', 'model_name' : '4FGL J0427.8-6704', 'xcoord' : 66.958, 'ycoord' : -67.078, 'rec_binsz' : 4000000, 'catalogue_name' : '3FGL J0427-6704', 'start_time' : None, 'min_error' : 0.4760000000000003},
               
{'name' : 'J1723', 'model_name' : None , 'xcoord' : 260.846, 'ycoord' : -28.633, 'rec_binsz' : 4000000, 'catalogue_name' : 'PSR J1723-2837', 'start_time' : None, 'min_error' : None},
               
{'name' : 'J2215', 'model_name' : '4FGL J2215.6+5135', 'xcoord' : 333.888, 'ycoord' : 51.593, 'rec_binsz' : 4000000, 'catalogue_name' : 'PSR J2215+5135', 'start_time' : None, 'min_error' : 0.40300000000000025},
               
{'name' : 'J1544', 'model_name' : '4FGL J1544.5-1126', 'xcoord' : 236.058, 'ycoord' : -11.412, 'rec_binsz' : 4000000, 'catalogue_name' : 'PSR J1544-1125', 'start_time' : None, 'min_error' :  0.5350000000000004},
               
{'name' : 'J0212', 'model_name' : '4FGL J0212.1+5321', 'xcoord' : 33.042, 'ycoord' : 53.344, 'rec_binsz' : 4000000, 'catalogue_name' : 'PSR J0212+5320', 'start_time' : None, 'min_error' : 0.18900000000000008},
               
{'catalogue_name': 'PSR J1417-4404', 'model_name': '4FGL J1417.6-4403', 'name': 'J1417', 'rec_binsz': 4000000, 'xcoord': 214.321, 'ycoord': -44.078, 'start_time' : None, 'min_error' : 0.14800000000000005},
               
{'catalogue_name': 'CXOU J1109-6502', 'model_name': None , 'name': 'J1109', 'rec_binsz': 4000000, 'xcoord': 167.358, 'ycoord': -65.04, 'start_time' : None, 'min_error' : None},
               
{'catalogue_name': 'PSR J1746-2844','model_name': '4FGL J1746.4-2852','name': 'J1746','rec_binsz': 4000000,'xcoord': 266.637,'ycoord': -28.741, 'start_time' : None, 'min_error' : 1.0219999999999982},
               
{'catalogue_name': 'CXOGlb J1748-2446','model_name': '4FGL J1748.0-2446','name': 'J1748','rec_binsz': 12000000,'xcoord': 267.017,'ycoord': -24.778, 'start_time' : None, 'min_error' : 0.31800000000000017},
               
{'catalogue_name': 'IGR J1737-3747','model_name': None ,'name': 'J1737','rec_binsz': 4000000,'xcoord': 264.496,'ycoord': -37.788, 'start_time' : None, 'min_error' : None},
               
{'catalogue_name': 'PSR J2339-0530','model_name': '4FGL J2339.6-0533','name': 'J2339','rec_binsz': 4000000,'xcoord': 354.912,'ycoord': -5.501, 'start_time' : None, 'min_error' : 0.23100000000000012},
               
{'catalogue_name': 'PSR J1311-3429','model_name': '4FGL J1311.7-3430','name': 'J1311','rec_binsz': 4000000,'xcoord': 197.958,'ycoord': -34.485, 'start_time' : None, 'min_error' : 0.0},
               
{'catalogue_name': 'PSR J0251+2600','model_name': '4FGL J0251.0+2605','name': 'J0251','rec_binsz': 4000000,'xcoord': 42.75,'ycoord': 26.0, 'start_time' : None, 'min_error' : 0.09770000000000177},
               
{'catalogue_name': 'PSR J1048+2339','model_name': '4FGL J1048.6+2340','name': 'J1048','rec_binsz': 4000000,'xcoord': 162.179,'ycoord': 23.665, 'start_time' : None, 'min_error' : 0.22800000000000012},
               
{'catalogue_name': 'PSR J1805+0600','model_name': '4FGL J1805.6+0615','name': 'J1805','rec_binsz': 4000000,'xcoord': 271.25,'ycoord': 6.0, 'start_time' : None, 'min_error' : 0.14400000000000004},
               
{'catalogue_name': 'PSR J2129-0429','model_name': '4FGL J2129.8-0428','name': 'J2129','rec_binsz': 4000000,'xcoord': 322.437,'ycoord': -4.487, 'start_time' : None, 'min_error' : 0.1961999999999947},
               
{'catalogue_name': '4FGL J0407-5702','model_name': '4FGL J0407.7-5702','name': 'J0407','rec_binsz': 4000000,'xcoord': 61.75,'ycoord': -57.033, 'start_time' : None, 'min_error' : 0.11900000000000002},
               
{'catalogue_name': '4FGL J0935+0901','model_name': '4FGL J0935.3+0901','name': 'J0935','rec_binsz': 4000000,'xcoord': 143.838,'ycoord': 9.027, 'start_time' : None, 'min_error' : 0.5790000000000004},
               
{'catalogue_name': '4FGL J2333-5527','model_name': '4FGL J2333.1-5527','name': 'J2333','rec_binsz': 4000000,'xcoord': 353.258,'ycoord': -55.45, 'start_time' : None, 'min_error' : 0.09840000000000179},
               
{'catalogue_name': 'PSR J0024-7204','model_name': '4FGL J0024.0-7204','name': 'J0024','rec_binsz': 4000000,'xcoord': 6.025,'ycoord': -72.08, 'start_time' : None, 'min_error' : 0.2120000000000001},
               
{'catalogue_name': 'XMMU J0838-2827','model_name': '4FGL J0838.7-2827','name': 'J0838','rec_binsz': 4000000,'xcoord': 129.708,'ycoord': -28.466, 'start_time' : None, 'min_error' : 0.12600000000000003},
               
{'catalogue_name': 'XTE J1814-3346','model_name': None,'name': 'J1814','rec_binsz': 4000000,'xcoord': 273.662,'ycoord': -33.773, 'start_time' : None, 'min_error' : None},
               
{'catalogue_name': '4FGL J2039-5618','model_name': '4FGL J2039.5-5617','name': 'J2039','rec_binsz': 4000000,'xcoord': 309.917,'ycoord': -56.312, 'start_time' : None, 'min_error' : 0.13500000000000004},
              
{'catalogue_name': '4FGL J0940-7610', 'min_error': 0.19900000000000015, 'model_name':'4FGL J0940.3-7610', 'name': 'J0940', 'rec_binsz': 4000000, 'start_time': None, 'xcoord': 145.0, 'ycoord': -76.167}]

def retrieve_source(source):
    i = 0
    while i < len(source_list):
        if source_list[i]['name'] == source:
            return source_list[i]
        else:
            i = i + 1

def run_subprocess(cmd, cwd=None, env=None):
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
        
def download_photons():
    '''This function is used to download the entire LAT photon library in 
    weekly snapshots and retrive any newly released data that isn't currently present'''
    #Photon data retrival
    cmd = ['wget', '--quiet', '-m', '-P', data_dir+'/Weekly_Photons', '-nH', '--cut-dirs=4', '-np', '-e',
           'robots=off', 'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/photon/']
    run_subprocess(cmd)

def download_spacecraft():    
    '''This function is used to download the entire LAT spacecraft library in 
    weekly snapshots and retrive any newly released data that isn't currently present'''
    cmd = ['wget', '--quiet', '-m', '-P', data_dir+'/Weekly_Spacecraft', '-nH', '--cut-dirs=4', '-np', '-e',
           'robots=off', 'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/spacecraft/']
    run_subprocess(cmd)
    
def gen_file_list():
    '''Creates text files containing lists of the paths to the weekly photon and spacecraft
    data files to be used by the fermi tool when carrying out data analysis'''
    
    incomplete = 0
    
    spacecraftlist = sorted(glob.glob(data_dir+'/Weekly_Spacecraft/weekly/spacecraft/*.fits'))
    with open(code_dir+'/spacecraft.txt', 'w') as f:
        for i in range(0,len(spacecraftlist)):
            F = astropy.io.fits.open(spacecraftlist[i])
            data = F[1].data
            if not data['DATA_QUAL'].sum() == 0:
                f.write(str(spacecraftlist[i]))
                f.write('\n')

    photonlist = sorted(glob.glob(data_dir+'/Weekly_Photons/weekly/photon/*.fits'))
    with open(code_dir+'/photons.txt', 'w') as f:
        for i in range(0,len(photonlist)):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error')
                    F = astropy.io.fits.open(photonlist[i], output_verify='warn')
                    f.write(str(photonlist[i]))
                    f.write('\n')   
            except:
                print 'file '+str(photonlist[i])+' is not complete, redownloading'
                os.remove(photonlist[i])
                incomplete = 1
                
    if incomplete == 1:
        download_photons()
    else:
        pass

def source_model(source):
    '''Creates a model of all sources in region of sky to be fit, necessary for source map
    creation and likelihood analysis
    '''

    name = retrieve_source(source)['name']
    xcoord = retrieve_source(source)['xcoord']
    ycoord = retrieve_source(source)['ycoord']
    
    photonlist = sorted(glob.glob(data_dir+'/Weekly_Photons/weekly/photon/*.fits'))
    
    Fstart = astropy.io.fits.open(photonlist[0])        
    Fend = astropy.io.fits.open(photonlist[-1])

    start = Fstart[0].header['TSTART']
    end = Fend[0].header['TSTOP']
    
    if not os.path.exists(data_dir+'/'+name+'_Weekly/'+name+'_allphotons_filtered.fits'): #checks if file already exists
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
        gt.filter['outfile'] = data_dir+'/'+name+'_Weekly/'+name+'_allphotons_filtered.fits'
        gt.filter.run()
        
    if not os.path.exists(data_dir+'/'+name+'_Weekly/'+name+'_allphotons_gti.fits'):
        gt.maketime['scfile'] = '@spacecraft.txt'
        gt.maketime['filter'] = '(DATA_QUAL>0)&&(LAT_CONFIG==1)'
        gt.maketime['roicut'] = 'no'
        gt.maketime['evfile'] = data_dir+'/'+name+'_Weekly/'+name+'_allphotons_filtered.fits'
        gt.maketime['outfile'] = data_dir+'/'+name+'_Weekly/'+name+'_allphotons_gti.fits'
        gt.maketime.run()


    mymodel = srcList(data_dir+'/gll_psc_v21.xml',data_dir+'/'+name+'_Weekly/'+name+'_allphotons_gti.fits',data_dir+'/'+name+'_Weekly/'+name+'_model.xml') #generates the model
    mymodel.makeModel(data_dir+'/gll_iem_v07.fits','gll_iem_v07',data_dir+'/iso_P8R3_SOURCE_V2_v1.txt','iso_P8R3_SOURCE_V2_v1',normsOnly=True,radLim=5,
                 extDir=template_dir)

    with open(data_dir+'/'+name+'_Weekly/'+name+'_model_clean.xml', 'wt') as f: #cleans up a bug in the make model code which cause wrong path to template folder
        f.write(re.sub(r'\$\(LATEXTDIR\)/', 
                       '', 
                       open(data_dir+'/'+name+'_Weekly/'+name+'_model.xml').read()))


def select(name,xcoord,ycoord,tmid,tmin,tmax,binsz,chatter):
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
    gt.filter['outfile'] = name_gen('filtered', name, tmid, binsz)
    gt.filter['chatter'] = chatter
    gt.filter.run()


def good_time_int(name,tmid,binsz,chatter): 
    '''Performs a good time interval cut to remove data taken at 
    poor times - requires filtered data
    '''
    gt.maketime['scfile'] = '@spacecraft.txt'
    gt.maketime['filter'] = '(DATA_QUAL>0)&&(LAT_CONFIG==1)'
    gt.maketime['roicut'] = 'no'
    gt.maketime['evfile'] = name_gen('filtered', name, tmid, binsz)
    gt.maketime['outfile'] = name_gen('gti', name, tmid, binsz)
    gt.maketime['chatter'] = chatter
    gt.maketime.run()


def counts_map(name,tmid,xcoord,ycoord,binsz,chatter): 
    '''Generates counts map - requires gti file
    '''
    gt.evtbin['evfile'] = name_gen('gti', name, tmid, binsz)
    gt.evtbin['scfile'] = '@spacecraft.txt'
    gt.evtbin['outfile'] = name_gen('cmap', name, tmid, binsz)
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


def counts_cube(name,tmid,xcoord,ycoord,binsz,chatter):
    '''Generates counts cube - requires gti file
    '''
    gt.evtbin['evfile'] = name_gen('gti', name, tmid, binsz)
    gt.evtbin['scfile'] = '@spacecraft.txt'
    gt.evtbin['outfile'] = name_gen('ccube', name, tmid, binsz)
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


def live_time_cube(name,tmid,binsz,chatter): 
    '''Generates livetime cube - requires gti file
    '''
    gt.expCube['evfile'] = name_gen('gti', name, tmid, binsz)
    gt.expCube['scfile'] = 'spacecraft.txt'
    gt.expCube['outfile'] = name_gen('ltcube', name, tmid, binsz)
    gt.expCube['zmax'] = 90
    gt.expCube['dcostheta'] = 0.025
    gt.expCube['binsz'] = 1
    gt.expCube['chatter'] = chatter
    gt.expCube.run()


def exp_map(name,tmid,binsz,chatter):
    '''Generates exposure map - requires gti file and livetime cube
    '''
    gt.expMap['evfile'] = name_gen('gti', name, tmid, binsz)
    gt.expMap['scfile'] = '@spacecraft.txt'
    gt.expMap['expcube'] = name_gen('ltcube', name, tmid, binsz)
    gt.expMap['outfile'] = name_gen('expMap', name, tmid, binsz)
    gt.expMap['irfs'] = 'CALDB'
    gt.expMap['srcrad'] = 30
    gt.expMap['nlong'] = 120
    gt.expMap['nlat'] = 120
    gt.expMap['nenergies'] = 37
    gt.expMap['chatter'] = chatter
    gt.expMap.run()


def exp_cube(name,tmid,xcoord,ycoord,binsz,chatter):
    '''Generates exposure cube - requires livetime cube
    '''
    gt.gtexpcube2['infile'] = name_gen('ltcube', name, tmid, binsz)
    gt.gtexpcube2['outfile'] = name_gen('expCube', name, tmid, binsz)
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


def source_map(name,tmid,binsz,chatter): #map of sources in region of sky
    '''Generates a sourcemap - requires counts cube, source model, exposure cube, 
    and livetime cube
    '''
    gt.srcMaps['expcube'] = name_gen('ltcube', name, tmid, binsz)
    gt.srcMaps['scfile'] = '@spacecraft.txt'
    gt.srcMaps['cmap'] = name_gen('ccube', name, tmid, binsz)
    gt.srcMaps['srcmdl'] = data_dir+'/'+name+'_Weekly/'+name+'_model_clean.xml'
    gt.srcMaps['bexpmap'] = name_gen('expCube', name, tmid, binsz)
    gt.srcMaps['outfile'] = name_gen('srcMap', name, tmid, binsz)
    gt.srcMaps['irfs'] = 'CALDB'
    gt.srcMaps['ptsrc'] = 'yes'
    gt.srcMaps['chatter'] = chatter
    gt.srcMaps.run()


def calc_flux(name,tmid,model_name,binsz,new_last_photon,chatter): 
    '''Runs the likelihood analysis using source map, livetime cube, and exposure cube 
    with a NEWMINUIT optimizer
    '''
    obs = BinnedObs(srcMaps=name_gen('srcMap', name, tmid, binsz), #binned objects
                expCube=name_gen('ltcube', name, tmid, binsz), binnedExpMap=name_gen('expCube', name, tmid, binsz), 
                irfs='CALDB')
    like = BinnedAnalysis(obs, data_dir+'/'+name+'_Weekly/'+name+'_model_clean.xml', optimizer='NEWMINUIT')

    likeobj = pyLike.NewMinuit(like.logLike) #likelihood fitting 
    like.fit(verbosity=chatter,covar=True,optObject=likeobj) #save output to xml

    if likeobj.getRetCode() == 0: #test if newminuit converged 

        like.logLike.writeXml(name_gen('output', name, tmid, binsz)) #write required data to .txt need to save last photon in CalcFlux
        with open(data_dir+'/'+name+'_Weekly/Flux'+str(tmid)+'-'+str(binsz)+'.txt','w') as f:
            f.write(str(tmid)+' '
                        +str(like.flux(model_name, emin=100))+' '+str #need to figure out how to generalize this
                        (like.fluxError(model_name, emin=100))+' '+str(new_last_photon))
            f.write('\n')
        
    
        #removes large leftover files after flux has been succesfully calculated
        os.remove(name_gen('ltcube', name, tmid, binsz))
        os.remove(name_gen('gti', name, tmid, binsz))
        os.remove(name_gen('filtered', name, tmid, binsz))
        os.remove(name_gen('expMap', name, tmid, binsz))
        os.remove(name_gen('expCube', name, tmid, binsz))
        os.remove(name_gen('cmap', name, tmid, binsz))
        os.remove(name_gen('ccube', name, tmid, binsz))
        
        #srcmaps must be removed in a different way as multiple srcmaps may be generated
        srcMap_list = glob.glob(name_gen('srcMap', name, tmid, binsz)+'*')

        for srcMap_path in srcMap_list:    
            os.remove(srcMap_path)

    else:
         print str(name)+'Analysis did not converge. Bin = '+str(tmid)
            


def calc_single_bin(source,tmid,binsz,end,chatter=0):
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
    
    if os.path.exists(data_dir+'/'+name+'_Weekly/Flux'+str(tmid)+'-'+str(binsz)+'.txt'):
        old_last_photon = np.loadtxt(data_dir+'/'+name+'_Weekly/Flux'+str(tmid)+'-'+str(binsz)+'.txt')[3]
        if old_last_photon < new_last_photon:
            print 'New data found, recalculating bin'+str(tmid)
            os.remove(data_dir+'/'+name+'_Weekly/Flux'+str(tmid)+'-'+str(binsz)+'.txt')
        elif old_last_photon == new_last_photon: #change to within certain tolerance 
            return;
        
    if not os.path.exists(name_gen('filtered', name, tmid, binsz)):
        select(name,xcoord,ycoord,tmid,tmin,tmax,binsz,chatter)

    if not os.path.exists(name_gen('gti', name, tmid, binsz)):
        good_time_int(name,tmid,binsz,chatter)

    if not os.path.exists(name_gen('cmap', name, tmid, binsz)):     
        counts_map(name,tmid,xcoord,ycoord,binsz,chatter)

    if not os.path.exists(name_gen('ccube', name, tmid, binsz)):
        counts_cube(name,tmid,xcoord,ycoord,binsz,chatter)

    if not os.path.exists(name_gen('ltcube', name, tmid, binsz)):
        live_time_cube(name,tmid,binsz,chatter)

    if not os.path.exists(name_gen('expMap', name, tmid, binsz)):   
        exp_map(name,tmid,binsz,chatter)

    if not os.path.exists(name_gen('expCube', name, tmid, binsz)):
        exp_cube(name,tmid,xcoord,ycoord,binsz,chatter)

    if not os.path.exists(name_gen('srcMap', name, tmid, binsz)):
        source_map(name,tmid,binsz,chatter)

    calc_flux(name,tmid,model_name,binsz,new_last_photon,chatter)

def error_pass(source,tmid,binsz,end,chatter=0):
    '''Wrapper for fermi tools function to implement error handling'''
    try:
        calc_single_bin(source,tmid,binsz,end,chatter=0)
    except RuntimeError as e:
        print e.message+'\nBin = '+str(tmid)+' \nSource = '+source
        remove_badbin(source,tmid)
        pass

def calc_all_bins(poolsize,source,binsz=None):
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
    
    photonlist = sorted(glob.glob(data_dir+'/Weekly_Photons/weekly/photon/*.fits'))
    
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
        return error_pass(source,tmid,binsz,end,chatter=0)
    
    if model_name is None:
        print 'Source {} has no model name'.format(source)
        pass
    else:
        if poolsize > 1:
            pool = mp.Pool(poolsize)
            pool.map(ft, tmids) #runs the function in parallel on multiple cpus
            pool.close()
        else:
            map(ft, tmids)
    

def plot_curve(source,mode='catalogue'):
    '''Uses flux values generated from likelihood analysis to plot a
    lightcurve with a built in chi2 test to give a false positive probability'''
    
    name = retrieve_source(source)['name']
    binsz = retrieve_source(source)['rec_binsz']
    catalogue_name = retrieve_source(source)['catalogue_name']
    min_error = retrieve_source(source)['min_error']
    
    spy = 31536000
    start_dec_yrs = 2008.0 + 8.0/12.0 + 4.0/365.0

    files = sorted(glob.glob(data_dir+'/'+name+'_Weekly/Flux*'+str(binsz)+'.txt'))
    data = np.zeros((len(files),4))

    for i in range(0,len(files)):
        data[i,:] = np.loadtxt(str(files[i]))
    
    sec_time = data[:,0]
    time = (data[:,0] - data[0,0])/spy + start_dec_yrs
    flux = data[:,1]/1E-8
    error = data[:,2]/1E-8
    
    start_time = retrieve_source(source)['start_time']
    fpp_error = error
    fpp_flux = flux
    
    if start_time is None:
        pass
    else:
        for i in range(0,len(files)):
            if sec_time[i] >= start_time:
                fpp_flux = flux[i:]
                fpp_error = error[i:]
                break
            else:
                pass
    
    quad_error = np.sqrt(min_error**2 + fpp_error**2)
    mean_flux = np.average(fpp_flux,  weights = 1/quad_error)
    s = (fpp_flux-mean_flux)/quad_error
    chi2_s = np.sum(s**2)
    fpp = 100*chi2(len(fpp_flux)-1).sf(chi2_s)
    
    if mode == 'catalogue':
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        #plt.errorbar(2008.0, mean_flux, min_error)
        plt.errorbar(time,flux,yerr=error,marker='',color='red',ecolor='grey',drawstyle='steps-mid',capsize=3)
        plt.axhline(y=mean_flux,linestyle='--', color='black', label='Mean flux for current state = %g'%mean_flux)
        plt.axhline(y=0, linestyle='-', color='black')
        plt.plot(np.NaN, np.NaN, '-', color='none', label='False positive probability = %g'%fpp+str(chr(37)))
        plt.legend(markerfirst=True, loc='best')
        plt.ylim(ymin=-2.5)
        plt.title(catalogue_name+' Light Curve')
        plt.xlabel('Time (years)')
        plt.ylabel(r'Flux ($10^{-8}cts cm^{-2} s^{-1}$)')
        plt.gcf().set_size_inches(8,6)
        plt.savefig(code_dir+'/light-curves/'+name+'LightCurve-'+str(binsz)+'.pdf')
        plt.savefig(code_dir+'/light-curves/'+name+'LightCurve-'+str(binsz)+'.png', dpi=300)
    elif mode == 'paper':
        plt.errorbar(time,flux,yerr=error,marker='',color='blue',ecolor='grey',drawstyle='steps-mid',capsize=0, linewidth=0.8)
        plt.axhline(y=mean_flux,linestyle='--', color='black', label='Mean flux for current state = %g'%mean_flux)
        plt.axhline(y=0, linestyle='-', color='black', linewidth=0.6)
        plt.plot(np.NaN, np.NaN, '-', color='none', label='False positive probability = %g'%fpp+str(chr(37)))
        plt.legend(markerfirst=True, loc='best', fontsize=14)
        plt.ylim(ymin=0)
        plt.xlabel('Time (years)', fontsize=14)
        plt.ylabel(r'Flux ($10^{-8}$cts cm$^{-2}$ s$^{-1}$)', fontsize=14)
        plt.xticks([2010, 2012, 2014, 2016, 2018, 2020], fontsize=14)
        plt.yticks(fontsize=14)
        plt.gcf().set_size_inches(8,6)
        plt.savefig(code_dir+'/light-curves/'+name+'Paper-'+str(binsz)+'.pdf')
        plt.savefig(code_dir+'/light-curves/'+name+'Paper-'+str(binsz)+'.png', dpi=300)
    
def export_fluxes():
    '''Reads flux files for each bin and writes values to a single .txt file
    so flux, error, and the centre of each bin can be read out of a single file
    for additional analysis'''
    for i in range(len(source_list)):
        
        if source_list[i]['model_name'] is None:
            pass
        else:
            name = source_list[i]['name']
            binsz = source_list[i]['rec_binsz']
            files = sorted(glob.glob(data_dir+'/'+name+'_Weekly/Flux*'+str(binsz)+'.txt'))
            data = np.zeros((len(files),4))
      
            with open(code_dir+'/'+name+'_All_Fluxes', 'w') as f:
                
                for j in range(len(files)):
            
                    data[j,:] = np.loadtxt(str(files[j]))               
                    f.write(str(data[j,0])+' : '+str(data[j,1])+' +/- '+str(data[j,2]))
                    f.write('\n')
            
def update_fluxes(poolsize=None):
    '''Function to calculate new bins or recalculate old bins when new data
    becomes available. Data must first be downloaded using the download functions'''
    gen_file_list()
    
    iterables = [i for i in range(len(source_list))]

    global uf
    def uf(iterable):
        return calc_all_bins(poolsize=1, source=source_list[iterable]['name'])
    
    if poolsize > 1:
        pool = mp.Pool(poolsize)
        pool.map(uf, iterables)
        pool.close()
    else:
        map(uf, iterables)
            
        
def update_curves(poolsize=None):
    
    for i in range(len(source_list)):
        if source_list[i]['model_name'] is None:
            pass
        else:
            plt.figure()
            plot_curve(source=source_list[i]['name'])
    
def name_gen(file, name, tmid, binsz):
        return data_dir+'/'+name+'_Weekly/'+name+'_'+file+'-'+str(tmid)+'-'+str(binsz)+'.fits'
    
    
def add_new_source():
    full_name = raw_input('Enter name in format -> PSR J 00 00 00 +/-00 00 00')
    distance = raw_input('Enter distance in kpc, if unknown leave blank')
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
    
    os.mkdir(data_dir+'/'+calendar+RAh+RAm+'_Weekly')
    
    if distance == '':
        binsz = 4000000
    elif float(distance) <= 2.0:
        binsz = 2000000
    elif 2.0 < float(distance) <= 4.0:
        binsz = 4000000
    elif 4.0 < float(distance) <= 6.0:
        binsz = 8000000
    elif float(distance) > 6.0:
        binsz = 12000000
    
    
    return {'name' : name, 'model_name' : None, 'xcoord' : RA, 'ycoord' : DEC, 'rec_binsz' : binsz, 'catalogue_name' : catalogue_name, 'start_time' : None, 'min_error' : None}
    
def set_min_error(test_error, source, step = 0.001):
    
    name = retrieve_source(source)['name']
    binsz = retrieve_source(source)['rec_binsz']

    files = sorted(glob.glob(data_dir+'/'+name+'_Weekly/Flux*'+str(binsz)+'.txt'))
    data = np.zeros((len(files),4))

    for i in range(0,len(files)):
        data[i,:] = np.loadtxt(str(files[i]))
    
    time = data[:,0]
    flux = data[:,1]/1E-8
    error = data[:,2]/1E-8
    
    start_time = retrieve_source(source)['start_time']
    
    if start_time is None:
        pass
    else:  
        for i in range(0,len(files)):
            if time[i] >= start_time:
                flux = flux[i:]
                error = error[i:]
                print('Transition cut off: ', time[i])
                break
            else:
                pass
        
    while test_error < 2.0:

        quad_error = np.sqrt(test_error**2 + error**2)
        mean_flux = np.average(flux,  weights = 1/quad_error)
        s = (flux-mean_flux)/quad_error
        chi2_s = np.sum(s**2)
        fpp = 100*chi2(len(flux)-1).sf(chi2_s)
        
        if fpp >= 49.5 and fpp <= 50.5:
            print('False positive probability: ', fpp)
            break
        elif fpp > 50.5:
            break
            
        test_error = test_error + step
        
    min_error = test_error
    print ('min_error : ', min_error)
    
    
def test_fpp(source='J1023'):
    
    name = retrieve_source(source)['name']
    binsz = retrieve_source(source)['rec_binsz']
    catalogue_name = retrieve_source(source)['catalogue_name']
    min_error = retrieve_source(source)['min_error']

    files = sorted(glob.glob(data_dir+'/'+name+'_Weekly/Flux*'+str(binsz)+'.txt'))
    data = np.zeros((len(files),4))

    for i in range(0,len(files)):
        data[i,:] = np.loadtxt(str(files[i]))
    
    time = data[:,0]
    flux = data[:,1]/1E-8
    error = data[:,2]/1E-8
    
    transition_files = sorted(glob.glob(data_dir+'/'+name+'_Weekly/Fpptest*.txt'))
    test_data = np.zeros((len(transition_files),4))
    
    for i in range(0,len(transition_files)):
        test_data[i,:] = np.loadtxt(str(transition_files[i]))
    
    time = np.append(time, [test_data[:,0]])
    flux = np.append(flux, [test_data[:,1]/1E-8])
    error = np.append(error, [test_data[:,2]/1E-8])
    
    start_time = retrieve_source(source)['start_time']
    # 394557417.0 for 1023
    fpp_error = error
    fpp_flux = flux
    
    if start_time is None:
        pass
    else:
        for i in range(0,len(files)):
            if time[i] >= start_time:
                fpp_flux = flux[i:]
                fpp_error = error[i:]
                print('Transition cut off: ', time[i])
                break
            else:
                pass
            
    quad_error = np.sqrt(min_error**2 + fpp_error**2)
    mean_flux = np.average(fpp_flux,  weights = 1/quad_error)
    s = (fpp_flux-mean_flux)/quad_error
    chi2_s = np.sum(s**2)
    fpp = 100*chi2(len(fpp_flux)-1).sf(chi2_s)
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    plt.errorbar(time,flux,yerr=error,marker='',color='red',ecolor='grey',drawstyle='steps-mid',capsize=3)
    plt.axhline(y=mean_flux,linestyle='--', color='black', label='Mean flux for current state = %g'%mean_flux)
    plt.plot(np.NaN, np.NaN, '-', color='none', label='False positive probability = %g'%fpp+str(chr(37)))
    plt.legend(markerfirst=True, loc=0)
    plt.ylim(ymin=0)
    plt.title(catalogue_name+' Light Curve')
    plt.xlabel('Time (years)')
    plt.ylabel('Flux ($10^{-8}cm^{-2}s^{-1}$)')
    plt.gcf().set_size_inches(8,6)
    plt.savefig(code_dir+'/light-curves/FPP_TEST_'+str(name)+'.png')
    
def remove_badbin(source, badbin):
    name = retrieve_source(source)['name']
    badfiles = glob.glob(data_dir+'/'+name+'_Weekly/*-'+str(badbin)+'*')
    
    for badfile in badfiles:
        os.remove(badfile)
        
def set_binsz(source):
    name = retrieve_source(source)['name']
    binsz = retrieve_source(source)['rec_binsz']
    catalogue_name = retrieve_source(source)['catalogue_name']
    min_error = retrieve_source(source)['min_error']

    files = sorted(glob.glob(data_dir+'/'+name+'_Weekly/Flux*'+str(binsz)+'.txt'))
    data = np.zeros((len(files),4))

    for i in range(0,len(files)):
        data[i,:] = np.loadtxt(str(files[i]))
    
    start_time = retrieve_source(source)['start_time']

    sec_time = data[:,0]
    if start_time is None:
        flux = data[:,1]
        error = data[:,2]
        time = sec_yt
    else:
        for i in range(0,len(files)):
            if sec_time[i] >= start_time:
                flux = data[i:,1]
                error = error[i:,2]
                break
            else:
                pass
    
    quad_error = np.sqrt(min_error**2 + error**2)
    mean_flux = np.average(flux,  weights = 1/quad_error)
    
    if mean_flux >= 6.0:
        print('Recomended bin size : 2,000,000s')

    
    
    
    
    