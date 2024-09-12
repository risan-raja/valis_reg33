import shapely
import numpy as np
import scipy
from skimage.feature import peak_local_max
from skimage import draw as skdraw



def get_longest_side_line(polygon_coordinates, side='right'):
    poly = shapely.Polygon(polygon_coordinates)
    mbr = poly.minimum_rotated_rectangle
    if isinstance(mbr, shapely.Polygon):
        mbr_points = list(zip(*mbr.exterior.coords.xy))
        mbr_lines = [shapely.LineString((mbr_points[i], mbr_points[i + 1])) for i in range(len(mbr_points) - 1)]
        mbr_line_lengths = [line.length for line in mbr_lines]
        lineidx = np.argmax(mbr_line_lengths)

        short_length = mbr_line_lengths[(lineidx + 1) % 4]

        p1, p2 = mbr_points[lineidx], mbr_points[lineidx + 1]
        p1a, p2a = mbr_points[(lineidx + 2) % 4], mbr_points[(lineidx + 3) % 4]
        if (side == 'right' and max(p1a[0], p2a[0]) > max(p1[0], p2[0])) or (side == 'left' and min(p1a[0], p2a[0]) < min(p1[0], p2[0])):
            return p1a, p2a, short_length

        return p1, p2, short_length
    return None, None, 0

def line_orientation(p1, p2):
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi
    
def poly_orientation(poly):
    p1, p2, _ = get_longest_side_line(poly)
    if p1 is not None:
        if p1[0] > p2[0]:
            return line_orientation(p2, p1)
        else:
            return line_orientation(p1, p2)
    return 0

def calculate_representative_point(polygon_coordinates, dbg=False):
    # warning: there is no scaling here - polygon_coordinates should be in reduced form before calling
    xs, ys = zip(*polygon_coordinates)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if max_x == min_x:
        max_x = min_x + 1
    if max_y == min_y:
        max_y = min_y + 1

    scaled_xs = np.array(xs) - min_x
    scaled_ys = np.array(ys) - min_y

    img_size = int(max(max_y - min_y + 1, max_x - min_x + 1))
    
    img = np.zeros((img_size // 2 + 1, img_size // 2 + 1), dtype=bool)
    rr, cc = skdraw.polygon(scaled_ys / 2, scaled_xs / 2)
    # rr = scaled_ys.astype(int)//2
    # cc = scaled_xs.astype(int)//2
    img[rr, cc] = True

    distance = scipy.ndimage.distance_transform_edt(img)
    # if dbg:
    #     plt.figure()
    #     plt.imshow(distance,cmap='gray')
    
    max_points = peak_local_max(distance, num_peaks=1)
    if max_points.shape[0] > 0:
        representative_point = max_points[0]
        
    else:
        representative_point = [scaled_ys.mean(), scaled_xs.mean()]
        
        # return centroid_x + min_x, centroid_y + min_y, 0
    # if dbg:
    #     plt.plot(representative_point[1], representative_point[0])
            
    x_rep = representative_point[1] * 2 + min_x
    y_rep = representative_point[0] * 2 + min_y
    return x_rep, y_rep, np.max(distance) * 2


def get_raster_shape(polygon_coordinates, dbg=False):
    xs, ys = zip(*polygon_coordinates)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if max_x == min_x:
        max_x = min_x + 1
    if max_y == min_y:
        max_y = min_y + 1

    scaled_xs = np.array(xs) - min_x
    scaled_ys = np.array(ys) - min_y

    
    nr = int(max_y - min_y + 1)
    nc = int(max_x - min_x + 1)
    img = np.zeros((nr, nc), dtype=bool)
    rr, cc = skdraw.polygon(scaled_ys, scaled_xs)
    
    # rr = scaled_ys.astype(int)//2
    # cc = scaled_xs.astype(int)//2
    img[rr, cc] = True
    return img, (min_x, min_y), (max_x,max_y)

    
def check_touching(text_tl,textbox_wh,contourimg):
    shp = contourimg.shape
    textimg = np.zeros_like(contourimg)
    r1 = int(text_tl[1])
    r2 = min(shp[0],int(text_tl[1]+textbox_wh[1]+1))
    c1 = int(text_tl[0])
    c2 = min(shp[1],int(text_tl[0]+textbox_wh[0]+1))
    # print('inside check',r1,r2,c1,c2)
    
    if r1<0 or c1 < 0 or r2 > contourimg.shape[0] or c2 > contourimg.shape[1]:
        return True
        
    # print('intersection check')
    textimg[r1:r2,c1:c2]=True
    exceed = textimg & ~contourimg
    
    # plt.figure()
    # plt.subplot(1,3,1)
    # plt.imshow(textimg)
    # plt.subplot(1,3,2)
    # plt.imshow(contourimg)
    # plt.subplot(1,3,3)    
    # plt.imshow(exceed)
    # plt.show()
    return np.sum(exceed)/np.sum(textimg)>0.1
