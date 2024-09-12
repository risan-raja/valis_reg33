from .slide import BFIWSlide
import numpy as np
import cv2
import scipy.ndimage as ndi


def make_slide(slide_path, key=None, is_ref=False):
    slide = BFIWSlide(slide_path, key, is_ref)
    return {key: slide}

def neighbourhood_values(image, coords):
    n_coords = coords - np.array([0, 1])
    s_coords = coords + np.array([0, 1])
    e_coords = coords + np.array([1, 0])
    w_coords = coords - np.array([1, 0])
    n_pixel = image[n_coords[:,0], n_coords[:,1]]
    s_pixel = image[s_coords[:,0], s_coords[:,1]]
    e_pixel = image[e_coords[:,0], e_coords[:,1]]
    w_pixel = image[w_coords[:,0], w_coords[:,1]]
    mean_pixel = np.mean([n_pixel, s_pixel, e_pixel, w_pixel], axis=0)
    var_pixel = np.var([n_pixel, s_pixel, e_pixel, w_pixel], axis=0)
    return np.c_[mean_pixel, var_pixel]


def post_process_mask(mask):
    mask = mask.get()
    # Blur 
    mask = cv2.GaussianBlur(mask, (5, 5), 0.5)  # type: ignore
    kernel = np.ones((3,3),np.uint8)
    cl_kernel = np.ones((3,3),np.uint8)
    # Closing
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, cl_kernel, iterations = 30) # type: ignore
    # Dilation
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations = 3) # type: ignore
    # Erosion
    mask = cv2.erode(mask.astype(np.uint8), kernel, iterations = 10) # type: ignore
    return mask

def predict_mask(slide: BFIWSlide, fil_model):
    """
    Predicts a mask using a given slide and a filter model.
    Args:
        slide (BFIWSlide): The slide object containing the mask and image.
        fil_model: The filter model used for prediction.
    Returns:
        final_mask: The final predicted mask after post-processing.
    """
    try:
        import cupy as cp # type: ignore
    except ImportError:
        raise ImportError("Cupy is required to run this function. Please install it using `pip install cupy`")
    top, bottom, left, right = 1, 1, 1, 1  # Example padding values
    border_type = cv2.BORDER_CONSTANT # type: ignore
    slide_mask = slide.mask
    slide_img = slide.msr_img
    slide_img_p = cv2.copyMakeBorder(slide_img, top, bottom, left, right, border_type)  # type: ignore
    slide_img_center = np.array([slide_img.shape[0]//2, slide_img.shape[1]//2]) # type: ignore
    bfi_coords = np.argwhere(slide_mask) # type: ignore
    bfi_pixels = slide_img[slide_mask==1] # type: ignore
    bfi_neighbourhood = neighbourhood_values(slide_img_p, bfi_coords)
    bfi_coords = bfi_coords-slide_img_center
    X = np.c_[bfi_pixels,bfi_neighbourhood, bfi_coords]
    X = X.astype(np.float32)
    X = cp.array(X)
    y_pred_ = fil_model.predict_proba(X)
    pred_var = y_pred_[:,1].var()
    y_pred = y_pred_[:,1]> 0.93*(1-(((y_pred_==1.0).sum()/len(y_pred_))*pred_var)**5)
    # Make the mask from the prediction
    predicted_mask = cp.zeros_like(slide_mask)
    predicted_mask[slide_mask==1] = y_pred
    final_mask = post_process_mask(predicted_mask)
    return final_mask, predicted_mask

def process_rf_mask(mask):
    mask = mask.astype(np.uint8)
    mask = ndi.binary_fill_holes(mask)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=5) # type: ignore
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=5)
    return mask
