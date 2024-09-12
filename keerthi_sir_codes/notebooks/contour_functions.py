import cv2
import shapely
import numpy as np

def get_features(contours, hier):
    polygons = []
    features = []
    outer = np.where(hier[0,:,3]==-1)[0]
    for outerid in outer:
        feat = [contours[outerid].squeeze()]
        innerids = []
        for innerid in np.where(hier[0,:,3]==outerid)[0]:
            innerids.append(innerid)
            feat.append(contours[innerid].squeeze())
            
        polygons.append({'outer':outerid,'inner':innerids})
        features.append(feat)
    return features, polygons

def plot_feature(feat):
# feat = features[0]
# if True:
    # print(len(feat))
    plt.imshow(np.ones((4000,4000),bool))
    if len(feat)>1:
        outer = feat[0]
        for inner in feat[1:]:
            plt.plot(inner[:,0],inner[:,1],'b-')
    else:
        outer = feat
    plt.plot(outer[:,0],outer[:,1],'r-')
    plt.axis('equal')

def make_polyshape(feat):
    if len(feat)>1:
        return shapely.Polygon(shell=feat[0],holes=feat[1:])
    return shapely.Polygon(feat[0])

def mask_to_shapes(msk):
    contours,hier=cv2.findContours( msk.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    features, _ = get_features(contours, hier)

    shapes = []
    for feat in features:
        try:
            shapes.append(make_polyshape(feat))
        except Exception as ex:
            print('make polyshape error',ex)
    return shapes
    
def labelmap_to_shapes(lmap):
    shapes = {}
    for clr in np.unique(lmap):
        if clr==0:
            continue
        shapes[clr] = mask_to_shapes(lmap==clr)
    return shapes
        
def nearest_shape(shp,otherlist):
    distances = [shapely.hausdorff_distance(shp,other) for other in otherlist]
    if len(distances)==0:
        return None, np.inf
    minidx = np.argmin(distances)
    dv = distances[minidx]
    nr = otherlist[minidx]
    minx,miny,maxx,maxy=nr.bounds
    width = max((maxx-minx),(maxy-miny))
    if dv > width:
        return None, np.inf
    return nr, dv
    
    