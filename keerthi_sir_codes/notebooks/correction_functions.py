import json
import numpy as np
# from skimage.io import imread
# import requests
# from matplotlib import pyplot as plt
# import shapely
from skimage.transform import SimilarityTransform
import pandas as pd


def get_trsdata(csvfile, secno):
    df = pd.read_csv(csvfile)
    aa = df[df['position_index'] == int(secno)]
    trsdata=aa.to_dict(orient='records')[0]
    tr16=json.loads(trsdata['tr16'])
    rot = trsdata['rotation']
    return {'tr16':tr16, 'rotation':rot, 'width':trsdata['width'],'height':trsdata['height']}

def apply_correction(pts,rot,jp2_wid,jp2_hei):
    if rot==270:
        theta = -90*np.pi/180
    elif rot==90:
        theta = 90*np.pi/180
    elif rot==180:
        theta = np.pi
    elif rot == 0:
        theta = 0

    c,s = np.cos(theta), np.sin(theta)

    # x' = x c - y s
    # y' = x s + y c

    org = np.array([jp2_wid/2,-jp2_hei/2])

    pts_rot = []
    for pt in pts:
        x,y = pt-org
        x_ = x*c-y*s+org[0]
        y_ = x*s+y*c+org[1]
        pts_rot.append([x_,y_])

    pts_rot = np.array(pts_rot)
    return pts_rot 


def apply_aligned(pts,tr16,rotation):
    
    trsxfm = SimilarityTransform(
        scale=1, 
        translation=np.array(tr16)*2*2*2*2*2, # from 16 micron to 0.5 micron
        rotation=-rotation)
    X = pts[:,0][...,np.newaxis]
    _Y = pts[:,1][...,np.newaxis]
    YX = (trsxfm(np.hstack([-_Y,X])))
    XY = np.fliplr(YX)
    rgn_coords_aligned = np.hstack([XY[:,0][...,np.newaxis], -XY[:,1][...,np.newaxis]]) # x,-y
    return rgn_coords_aligned

def get_final_coords(rgn_coords, jsonrotation, jp2_width, jp2_height, tr16, rotation):
    rgn_coords_corrected = apply_correction(rgn_coords, jsonrotation, jp2_width, jp2_height )
    rgn_coords_out = apply_aligned(rgn_coords_corrected, tr16, rotation)
    return rgn_coords_out
    