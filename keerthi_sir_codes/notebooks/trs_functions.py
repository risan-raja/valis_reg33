from skimage.io import imread
from skimage.feature import match_descriptors, SIFT
from skimage.exposure import rescale_intensity
from skimage.morphology import (
    binary_dilation, diamond, binary_closing, disk, binary_erosion, square,
    area_opening)

from collections import namedtuple
from skimage import transform
import numpy as np
from skimage.measure import ransac
import cv2

class NotEnoughMatchesError(Exception):
    pass

CropROI = namedtuple('CropROI','r1,c1,r2,c2')


def crop_or_pad(arr,croproi:'CropROI'):

    r1 = croproi.r1
    c1 = croproi.c1

    r2 = croproi.r2
    c2 = croproi.c2

    shp = arr.shape
    
    pad_r = [0,0]
    
    if r1<0:
        pad_r[0] = -r1
        r1 = 0
        
    if r2>shp[0]:
        pad_r[1] = r2-shp[0]
        r2 += pad_r[0]
        
    pad_c = [0,0]

    if c1<0:
        pad_c[0] = -c1
        c1 = 0
        
    if c2>shp[1]:
        pad_c[1] = c2-shp[1]
        c2 += pad_c[0]

    padvalues = [pad_r, pad_c]
    if len(shp)>2:
        padvalues+=[[0,0]]
    arr_padded = np.pad(arr,padvalues,constant_values=255)
    
    return arr_padded[r1:r2,c1:c2,...]
    

from scipy import linalg as la
import SimpleITK as sitk
from skimage.transform import SimilarityTransform

def mywarp(img,xfm_mtx, refimg=None, bordercolor=0, interpolator=sitk.sitkLinear):
    
    assert len(img.shape)==2

    ftr,ftc = tuple(xfm_mtx[:2,-1].tolist())
    
    moving_si = sitk.GetImageFromArray(img)
    invmat = la.inv(xfm_mtx)
    
    tr,tc = tuple(invmat[:2,-1].tolist())
    rot = invmat[:2,:2]
    
    sr,sc = la.norm(rot[:,0]),la.norm(rot[:,1])
    rot[:,0]/=sr
    rot[:,1]/=sc
    
    moving_si.SetOrigin((tr,tc))
    moving_si.SetDirection(rot.ravel().tolist())
    
    # print('inside mywarp', tr,tc,ftr,ftc, np.arcsin(rot[1,0]),sr,sc)
    
    if refimg is None:
        refimg = img
    else:
        assert len(refimg.shape)==2

    tmparr = np.zeros_like(refimg)
    ref_si = sitk.GetImageFromArray(tmparr)
    
    if True:
        ref_si.SetOrigin((tr/sr-ftc,tc/sc-ftr)) # ??!! -  because getimagefromarray is not using tmparr.T
    
        
    warpxfm = sitk.Similarity2DTransform()
    warpxfm.SetScale(sr)
    
    moved_si = sitk.Resample(moving_si, ref_si, warpxfm, interpolator, bordercolor)
    
    # for img in moving_si, ref_si, moved_si:
    #     print(img.GetSize(),img.GetOrigin(),img.GetDirection(),img.GetSpacing())
    
    moved = sitk.GetArrayFromImage(moved_si)
    return moved

def warp_rgb(arr,mtx,outsize = [2000,2000], bordercolor=255):
    im_arr = np.zeros((outsize[0],outsize[1],3),np.uint8)
    im_arr[...,0] = mywarp(arr[...,0], mtx, np.zeros(outsize,np.uint8),bordercolor=bordercolor)
    im_arr[...,1] = mywarp(arr[...,1], mtx, np.zeros(outsize,np.uint8),bordercolor=bordercolor)
    im_arr[...,2] = mywarp(arr[...,2], mtx, np.zeros(outsize,np.uint8),bordercolor=bordercolor)
    return im_arr

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
    
def rotatearr(arr3c,deg):
    if deg==90:
        return rot90(arr3c)
    if deg==180:
        return rot180(arr3c)
    if deg==270:
        return rot270(arr3c)
    return arr3c


def get_salient_mask(arr,percent=0.2,sz_divisor=100):
    nr,nc=arr.shape[:2]
    assert percent > 0 and percent < 1, str(percent)
    if sz_divisor <= 0 or sz_divisor > min(nr,nc)//20:
        sz_divisor = min(nr,nc)//20
    # (100-percent)th percentile threshold abs(arr), and select top patches which cover the required percentage of image
    
    mapimg = transform.resize(np.abs(arr),(nr//sz_divisor,nc//sz_divisor),order=1)
    
    th = np.percentile(mapimg,100*(1-percent))
    msk_small = binary_closing(mapimg>th,diamond(5))
    msk_small = binary_closing(msk_small,disk(5))
    msk_salient = transform.resize(msk_small,(nr,nc),order=0)
    return msk_salient.astype(bool)

    
class RegImage:
    
    def __init__(self, imgpath:str, croproi:'CropROI'=None, rot=0):    
        
        
        self.roi = croproi
        self.descriptor_extractor = SIFT(upsampling=1)
        self.arr = imread(imgpath)
        if croproi is not None:
            self.arr = crop_or_pad(self.arr, croproi)

        if rot !=0:
            if rot in (90, 180, 270): # clockwise
                self.arr = rotatearr(self.arr, rot)    
            else:
                arrtype=self.arr.dtype
                self.arr = transform.rotate(self.arr,-rot, mode='constant',cval=255,preserve_range=True).astype(arrtype)

    def get_gray(self):
        channelim = self.arr[...,1].copy()
        in_range_gr = np.percentile(channelim,(2,99.99)).astype(np.uint8)

        return rescale_intensity(channelim,in_range=tuple(in_range_gr),
                                         out_range=(0,255)).astype(np.uint8)

    def custom_gray(self):
        gray = cv2.cvtColor(self.arr,cv2.COLOR_RGB2GRAY)
        check = (gray/255) - np.ones_like(gray, dtype=np.uint8)
        return check
        
    def compute_keypoints(self,factor=2,convert_gray=True):
        if convert_gray:
            gray = self.get_gray()
        else:
            gray = self.custom_gray()
        self.descriptor_extractor.detect_and_extract(gray[::factor,::factor])
        self.keypoints = self.descriptor_extractor.keypoints.copy() * factor
        self.descriptors = self.descriptor_extractor.descriptors.copy()

    def get_blackmask(self):
        nr,nc,d = self.arr.shape
        img = self.arr[::4,::4]
        # blk = (img[:,:,0]<20) & (img[:,:,1]<20) & (img[:,:,2]<20)
        blk0 = (img.mean(axis=2)<150) & (img[...,2]<210) & (img[:,:,0] < 90)
        blk0 = area_opening(blk0,100)

        blk = binary_dilation(blk0,square(29))
        blk = transform.resize(blk,(nr,nc),order=0)
        return blk

    def compute_mask(self):
        
        in_range_gr = np.percentile(self.arr[...,1],(5,99.99)).astype(np.uint8)

        arr1 = rescale_intensity(self.arr[...,1],in_range=tuple(in_range_gr),
                                         out_range=(0,192)).astype(np.uint8)
            
        pc=(arr1<180).sum()/(arr1.shape[0]*arr1.shape[1])
        mask = get_salient_mask(255-arr1,max(0.4,min(0.7,1.1*pc))*self.contentpercent,16)
        mask = binary_dilation(mask,diamond(11))
        return mask

    def compute_mask_2(self):
        gray = cv2.cvtColor(self.arr, cv2.COLOR_RGB2GRAY)
        median: float = np.median(gray) # type: ignore
        t1 = 30
        edges = cv2.Canny(gray, threshold1=t1,threshold2=3*t1)
        kernel = np.ones((3,3), np.uint8)
        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=10)
        erosion = cv2.morphologyEx(closing, cv2.MORPH_ERODE, kernel, iterations=10)
        # plt.imshow(erosion[::4, ::4], cmap='gray')
        contours, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(erosion)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
        return mask.astype("bool")

    

def register_images(movingimg:'RegImage', refimg:'RegImage', factor=3, convert_gray=True):
    try:
        movingimg.compute_keypoints(factor, convert_gray=convert_gray) # FIXME: magic
        refimg.compute_keypoints(factor, convert_gray=convert_gray)
    except Exception as ex:
        print('fault',ex)
        return None

    matches = match_descriptors(refimg.descriptors, movingimg.descriptors, 
                                max_ratio=0.9,
                                cross_check=True)
    if len(matches) < 5:
        print('not enough matches')
        # raise NotEnoughMatchesError("Not enough SIFT matches")
        return None

    print('n.matches',len(matches))
          
    pts_mov = movingimg.keypoints[matches[:,1],:].copy()
    pts_ref = refimg.keypoints[matches[:,0],:].copy()  
    
    xfmtype = transform.EuclideanTransform
    
    npts = pts_mov.shape[0]
    ms = np.max((4,npts//20))
    
    model_robust, inliers = ransac((pts_mov, pts_ref), xfmtype, min_samples=ms,
                                     residual_threshold=40, 
                                    #  is_model_valid=is_model_valid_func,
                                    # stop_sample_num=max(2*npts//3,4),
                                    max_trials=max(200,ms*npts))

    if inliers is not None:
        ninliers = inliers.sum()
        print('n.inliers',ninliers)
        if ninliers>=4:
            return model_robust

        
        
    return None