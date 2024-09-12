# from skimage.filters.rank import mean as meanfilt
from skimage.segmentation import morphological_chan_vese
from skimage.transform import resize
# from skimage.filters.rank import subtract_mean
from skimage.filters import unsharp_mask
from skimage.exposure import match_histograms, rescale_intensity
from skimage.morphology import binary_dilation, binary_closing, disk, area_opening
from skimage.io import imread, imsave
import numpy as np
import glob
import sys
import os
from tqdm import tqdm
from scipy.ndimage import shift

def rot270(arr3c):
    if len(arr3c.shape)==3:
        return np.flipud(np.transpose(arr3c,(1,0,2)))
    elif len(arr3c.shape)==2:
        return np.flipud(arr3c.T)
    else:
        raise NotImplementedError
        
def rot90(arr3c):
    if len(arr3c.shape)==3:
        return np.fliplr(np.transpose(arr3c,(1,0,2)))
    elif len(arr3c.shape)==2:
        return np.fliplr(arr3c.T)
    else:
        raise NotImplementedError
        
def rot180(arr3c):
    return np.fliplr(np.flipud(arr3c))

def rotateim(arr3c,rotval):
    if rotval==90:
        return rot90(arr3c)
    if rotval==180:
        return rot180(arr3c)
    if rotval==270:
        return rot270(arr3c)
    return arr3c

def bfi_alter(bfi_crop):
    bfi_crop_fl = bfi_crop.astype(float)

    bfi_crop_fl = np.dstack([
        np.clip(bfi_crop_fl[...,0]*0.95,0,250),
        bfi_crop_fl[...,1]*0.8,
        (np.clip(bfi_crop_fl[...,0]*0.4+bfi_crop_fl[...,1]*0.3+bfi_crop_fl[...,2]*0.3 ,0,255))
     ])

    return bfi_crop_fl.astype(np.uint8)
    
    # return white_balance(bfi_crop_u8)

def bfi_bgboost(bfi_crop, whitergn):
    # whitergn = bfi_crop[-500:-250,500:1000,:]
    bfi_crop_copy = bfi_crop.copy().astype(float)
    
    for ch in range(3):
        chmean = whitergn[...,ch].mean()
        chstd = whitergn[...,ch].std()
        den = chmean+2*chstd
        # print(ch,den)
        bfi_crop_copy[...,ch]=rescale_intensity(bfi_[crop_copy...,ch]/den,out_range=(0,255))

    return bfi_crop_copy.astype(np.uint8)

def match_mu_sd(im, ref_mu, ref_sd):
    out = im.copy()
    for ch in range(3):
        mu = im[...,ch].mean()
        sd = im[...,ch].std()
        
        out[...,ch]=np.clip(((out[...,ch]-mu)/sd*ref_sd[ch])+ref_mu[ch],10,255)
    return out
    
bfi_slices = {
    '213':{'r':slice(500,1700),'c':slice(800,2300)},
    '222':{'r':slice(100,1500),'c':slice(1100,2300)},
    '141':{'r':slice(200,1400),'c':slice(800,2700)},
    '142':{'r':slice(250,1950),'c':slice(300,2300)},
    '244':{'r':slice(250,250+1700),'c':slice(950,950+1700)},
}
bfi_rot = {
    '213': 90,
    '222':90,
    '141':180,
    '142':0,
    '244':90,
}

white_rgn = {
    # '213': {'r':slice(50,150),'c':slice(50,150)},
    '213':{'r':slice(50,150),'c':slice(1300,1400)},
    # '141': {'r':slice(1000,1200),'c':slice(500,700)},
    '141': {'r':slice(950,1100), 'c':slice(1250,1450)},
    '142': {'r':slice(1200,1400),'c':slice(1400,1600)},
    # '244':{'r':slice(100,300),'c':slice(100,300)},
    '244':{'r':slice(600,800),'c':slice(1250,1450)},
    '222':{'r':slice(100,200),'c':slice(850,1000)},
}

secmin = {
    '222': 10,
    '141': 10,
    '142': 10,
    '213': 38,
    '244': 10,
}
    
secmax = {
    '222': 1730,
    '141': 1290,
    '142': 2100,
    '244': 2640,
    '213': 2610,
}


def find_translation(src:np.ndarray,ref:np.ndarray):
    assert len(src.shape)==2 and len(ref.shape)==2

    cross_corr = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(ref) * np.fft.fft2(src).conj()))
    max_corr_idx = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
    return np.array(max_corr_idx) - np.array(ref.shape) / 2
    
def process_bfi(biosampleid, secno, bfiname):
    bfi_im = imread(bfiname)
    
    if biosampleid=='213': # or True:
        bfi_im2 = bfi_alter(bfi_im)
    else:
        bfi_im2 = bfi_im

    bfi_crop0 = bfi_im[bfi_slices[biosampleid]['r'], bfi_slices[biosampleid]['c']] 
    
    bfi_crop1 = bfi_im2[bfi_slices[biosampleid]['r'], bfi_slices[biosampleid]['c']] 
    if biosampleid=='213':
        whitergn = bfi_crop1[white_rgn[biosampleid]['r'], white_rgn[biosampleid]['c']]
        bfi_crop1 = bfi_bgboost(bfi_crop1,whitergn)

    refmu = [160,155,150]
    refsd = [47, 51, 56]
    if biosampleid=='213': 
        refmu = [150,150,150]
        refsd = [50, 55, 65]
    if biosampleid=='222':
        refmu = [210,205,205]
        
    bfi_matched = match_mu_sd(bfi_crop1, refmu, refsd)

    whitergn = bfi_matched[white_rgn[biosampleid]['r'],white_rgn[biosampleid]['c']]

    wm=[whitergn[...,ch].mean() for ch in range(3)]

    bfi_rotated = rotateim(bfi_matched,bfi_rot[biosampleid])

    # shifting
    if biosampleid=='222' and int(secno) < 641:        
        bfi_rotated = shift(bfi_rotated, [90, -25, 0],mode='nearest',prefilter=False)
        
    if biosampleid=='213' and int(secno) < 880:
        bfi_rotated = shift(bfi_rotated, [-80, 0, 0], mode='nearest', prefilter=False)

    if biosampleid=='142' and int(secno) < 608:
        if int(secno)==385:
            tr = -550
        else:
            tr = 40
        bfi_rotated = shift(bfi_rotated, [tr,0,0], mode='nearest', prefilter=False)

    if biosampleid=='244':
        if int(secno)==573:
            bfi_rotated = shift(bfi_rotated, [-500,0,0], mode='nearest', prefilter=False)
    if biosampleid=='141':
        tr=0
        tc=0
        if int(secno) < 557:
            tr=-50
            tc=450
            if int(secno)<144:
                tc=350
                
        if int(secno)==766:
            tr=-100
            tc=0
        if int(secno) in (824,826):
            tr=30
            tc=0
        bfi_rotated = shift(bfi_rotated, [tr,tc,0], mode='nearest', prefilter=False)
        
    return bfi_rotated
    
    niter=20
    if biosampleid=='213':
        niter = 80
    if biosampleid=='142':
        niter = 50
    if biosampleid=='141':
        niter = 100
    if biosampleid=='222':
        niter = 50
    msk_bfi = morphological_chan_vese(bfi_rotated[::4,::4,2],num_iter=niter, init_level_set='disk')

    maximg = np.max(bfi_rotated[::4,::4,:],axis=2)
    if biosampleid=='213':
        maximg = bfi_rotated[::4,::4,1]

    M1=binary_dilation(binary_closing(msk_bfi>0,disk(11)),disk(15))
    M2 = M1 # rectmask(M1)
    M3 = resize(M2,bfi_rotated.shape[:2])>0

    msk_bg = binary_dilation(area_opening(maximg<100,2000),disk(11)) & ~ M1 # & (bfi_rotated[::4,::4,2]<100)# & (bfi_rotated[::4,::4,2]<120)

    msk_bg2 = resize(msk_bg,bfi_rotated.shape[:2])>0
    bfi_masked = bfi_rotated.copy()
    for ch in range(3):
        bfi_masked[...,ch][msk_bg2]=wm[ch]
        
    return bfi_masked

from joblib import Parallel, delayed

if __name__=="__main__":
    biosampleid = sys.argv[1]

    datadir='/data/keerthi/brainpubdata/%s/BFIW' % biosampleid
    outdir = datadir+'_processed'
    os.makedirs(outdir, exist_ok=True)
    
    bfidict={}
    
    for fn in glob.glob(datadir+'/*-SE_*_original.jpg'):
        bn = os.path.basename(fn)
        secno = bn.split('_')[3]
        if int(secno)>=secmin[biosampleid] and int(secno) <= secmax[biosampleid]:
            bfidict[int(secno)]=fn

    secno_list = list(sorted(bfidict.keys()))
    nsec = len(secno_list)
    mid = nsec//2
    
    secno_list_h1 = (secno_list[mid::-1])
    secno_list_h2 = secno_list[mid:]

    print(secno_list_h1[:5])
    print(secno_list_h2[:5])

    refsecno = secno_list_h1[0]
    refsecname = bfidict[refsecno]

    fixedarray = process_bfi(biosampleid, refsecno, refsecname)
    bn = os.path.basename(refsecname)
    newname = bn.replace('_original.jpg','_processed.jpg')
    imsave(outdir+'/'+newname, fixedarray[::4,::4,:])

    def workerfunc(seclist):
        refsecarray = fixedarray.copy()
        for ii, secno in tqdm(enumerate(seclist),total=len(seclist)):
            bfiname = bfidict[secno]
            nextarray = process_bfi(biosampleid, secno, bfiname)
            if biosampleid=='141' or ii < 300 or ii > len(seclist)-300:
                tr = find_translation(nextarray[300:-300,300:-300,2], refsecarray[300:-300,300:-300,2])
            else:
                tr = find_translation(nextarray[...,2], refsecarray[...,2])
            nextarray = shift(nextarray, tr.tolist()+[0], mode='nearest', prefilter=False)
            
            bn = os.path.basename(bfiname)
            newname = bn.replace('_original.jpg','_processed.jpg')
            imsave(outdir+'/'+newname, nextarray[::4,::4,:])
            refsecarray = nextarray

    if True:
        Parallel(n_jobs=2)(
            delayed(workerfunc)(seclist) for seclist in (secno_list_h1[1:], secno_list_h2[1:])
        )
        
    else:
        workerfunc(secno_list_h1[1:])
        workerfunc(secno_list_h2[1:])
        
