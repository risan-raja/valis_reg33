import SimpleITK as sitk
import os
import glob
import numpy as np
from skimage.io import imread
from trs_functions import CropROI, RegImage
from tqdm import tqdm
import sys
import requests
from PIL import Image
from skimage import transform

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

dataloc='/data/keerthi/brainpubdata/'

imgdir = {
    '244':dataloc+'/244/22Aug2024_FB85_CorrectedImages',
    '141':dataloc+'/141/20Aug2024_FB40_CorrectedImages/',
    '213':dataloc+'/213/26Aug2024_FB62_NISSL_CorrectedImages/',
    '222':dataloc+'/222/aligned_nissl_222/', #16Aug2024_FB74_CorrectedImages/',
    '142':dataloc+'/142/20Aug2024_FB34_CorrectedImages/',
}

brainname = {
    '141':'FB40',
    '142': 'FB34',
    '213': 'FB62',
    '222': 'FB74',
    '244': 'FB85',
}

def find_closest_smaller(arr1, target):
    # Filter for elements smaller than the target and find the maximum
    smaller_numbers = [num for num in arr1 if num <= target]
    if smaller_numbers:
        return max(smaller_numbers)
    return None  # In case there is no smaller number

def find_closest_larger(arr1, target):
    # Filter for elements larger than the target and find the minimum
    larger_numbers = [num for num in arr1 if num >= target]
    if larger_numbers:
        return min(larger_numbers)
    return None  # In case there is no larger number

def fillSecCloseOrLarge(present_sections,absent_sections,secno):
    closestSmallSec = find_closest_smaller(present_sections,secno)
    closestLargeSec = find_closest_larger(present_sections,secno)

    if closestSmallSec is None:
        if closestLargeSec is not None:
            return closestLargeSec
        else:
            return None
            
    if closestLargeSec is None:
        if closestSmallSec is not None:
            return closestSmallSec
        else:
            return None
        
    if abs(secno - closestSmallSec) < abs(secno - closestLargeSec):
        closestSec = closestSmallSec
    else:
        closestSec = closestLargeSec

    return closestSec

    
if __name__=="__main__":
    biosampleid=sys.argv[1]

    refdict={}
    for fn in glob.glob(imgdir[biosampleid]+'/*aligned*'):
        bn = os.path.basename(fn)
        secno = bn.split('_')[0]
        if "masked" not in bn and secno not in refdict:            
            refdict[secno]=fn
        

    # dsdict={}
    # for fn in glob.glob('/analyticsdata/%s/NISL/*_downsampled.tif' % biosampleid):
    #     bn = os.path.basename(fn)
    #     secno = bn.split('_')[5]
    #     dsdict[secno]=fn

    minsec=min(refdict.keys(),key=lambda x:int(x))
    maxsec=max(refdict.keys(),key=lambda x:int(x))

    nsl = round(int(maxsec)/3+0.5)

    res=64

    imgsiz=4000
    if biosampleid in ('244','142'):
        imgsiz=5000
        
    nr,nc=imgsiz*16//res,imgsiz*16//res

    

    endpoint='http://apollo2.humanbrain.in:8000'
    authtuple=('admin','admin')
    
    sslist = requests.get(endpoint+'/qc/SeriesSet/',auth=authtuple).json()
    ssdict = {elt['biosample']:(elt['id'],elt['name']) for elt in sslist}

    ssid = ssdict[int(biosampleid)]

    tnail_data=requests.get(endpoint+'/GW/getBrainViewerDetails/IIT/V1/SS-%d?public=0' % ssid[0]).json()
    nisl_data = tnail_data['thumbNail']['NISSL']
    nisl_dict={elt['position_index']:elt for elt in nisl_data}

    print('allocating', nsl,nr,nc,3)
    imgarr = 255*np.ones((nsl,nr,nc,3),np.uint8)
    imgarr_sub = 255*np.ones((nsl,nr//2,nc//2,3),np.uint8)
    
    lastsec = imgarr[0,...]

    # stackdict = {} # secno: 2darr
    
    def workerfunc(secno:int):
    
        if str(secno) in refdict and int(secno) in nisl_dict:
            secim = RegImage(refdict[str(secno)]) #,refroi) #,90) # clockwise 
            arr = secim.arr.copy()
            
            # if 'masked' not in os.path.basename(refdict[str(secno)]):
            #     # dsim = Image.open(dsdict[str(secno)])
            #     # downsampled_size = dsim.size
            #     # sz = secim.arr.shape
                
            #     # secim.contentpercent = downsampled_size[0]/sz[0]*downsampled_size[1]/sz[1]
            #     msk = secim.get_blackmask()>0
                
            #     for ch in range(3):
            #         arr[...,ch][msk]=255
                    
            # lastsec = secim.arr[::res//16,::res//16,:] # 16 to 64 mpp
            img64 = transform.resize(arr,(nr,nc),order=1, preserve_range=True).astype(arr.dtype)
            img128 = transform.resize(arr,(nr//2,nc//2),order=1, preserve_range=True).astype(arr.dtype)
            return img64, img128, secno
            
        return None, None, secno

    present_sections = []
    absent_sections = []

    seq = range(int(minsec),int(maxsec)+1,3)
    worksize = len(seq)
    n_workers = cpu_count()//4
    minwork = 5
    n_rounds = worksize//n_workers//minwork
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        
        for secimg64, secimg128, secno in executor.map(workerfunc, seq, chunksize=n_rounds):
            # stackdict[secno] = secimg
            if secimg64 is not None:
                present_sections.append(secno)
                slno = secno//3
                imgarr[slno,...]=secimg64
                imgarr_sub[slno,...]=secimg128
                print('*',end='',flush=True)
            else:
                # print('-',end='',flush=True)
                absent_sections.append(secno)
                print(secno)
    
    # for secno, secimg in stackdict.items():
    #     slno = secno//3
    #     imgarr[slno,...] = stackdict[secno]

    for secno in absent_sections:

        closestSec = fillSecCloseOrLarge(present_sections,absent_sections,secno)

        
        closestSec = closestSec //3
        slno = secno//3
        imgarr[slno,...] = imgarr[closestSec,...]
        imgarr_sub[slno,...] = imgarr_sub[closestSec,...]
    
    img=sitk.GetImageFromArray(imgarr)
    img.SetSpacing([res/1000,res/1000,0.060])
    
    drnmat=np.array([ [0,0,1], [-1,0,0], [0,-1,0] ]) # psr 

    # sagittal LR
    # rotmat = np.array([[-1,0,0],[0,-1,0],[0,0,1]]) # psr to asl
    
    # sagittal RL
    rotmat = np.eye(3) 
    
    # coronal AP
    if biosampleid=='213':
        rotmat = np.array([[0,-1,0],[1,0,0],[0,0,1]]) # psr to rsa
    
    drn = np.dot(rotmat,drnmat).ravel()

    img.SetDirection(drn.tolist())
    
    # outdir = dataloc+'/'+biosampleid
    outdir = '.'
    
    sitk.WriteImage(img, outdir+'/%s_nisl_%dmpp_rgb.nii.gz' % (brainname[biosampleid],res), useCompression=True)
    del imgarr, img
    
    imgsub = sitk.GetImageFromArray(imgarr_sub)
    imgsub.SetSpacing([res*2/1000,res*2/1000,0.060])
    imgsub.SetDirection(drn.tolist())

    sitk.WriteImage(imgsub, outdir+'/%s_nisl_%dmpp_rgb.nii.gz' % (brainname[biosampleid],res*2), useCompression=True)
    
    imgarr_sub_gray = 255-imgarr_sub[...,1]
    imgsub_gray = sitk.GetImageFromArray(imgarr_sub_gray)
    imgsub_gray.SetSpacing([res*2/1000,res*2/1000,0.060])
    imgsub_gray.SetDirection(drn.tolist())

    sitk.WriteImage(imgsub_gray, outdir+'/%s_nisl_%dmpp_gray.nii.gz' % (brainname[biosampleid],res*2), useCompression=True)
    del imgarr_sub, imgsub, imgarr_sub_gray, imgsub_gray
    