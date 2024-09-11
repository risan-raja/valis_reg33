from skimage.color.colorconv import ahx_from_rgb,hpx_from_rgb,bpx_from_rgb,bro_from_rgb,hax_from_rgb,gdx_from_rgb,rbd_from_rgb,bex_from_rgb,fgx_from_rgb,hdx_from_rgb,hed_from_rgb,separate_stains
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from joblib import Parallel, delayed
from skimage.color import separate_stains



def read_img_mask(image_idx, original=False):
    if original:
        retinex_path = f'rf_trained_pred/original/{str(image_idx).zfill(4)}_original.jpg'
    else:
        retinex_path = f'rf_trained_pred/retinex/{str(image_idx).zfill(4)}_retinex.jpg'
    mask_path = f'rf_trained_pred/processed_mask/{str(image_idx).zfill(4)}_mask.jpg'
    mask2_path = f'rf_trained_pred/raw_mask/{str(image_idx).zfill(4)}_raw_mask.jpg'
    image = cv2.imread(retinex_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask[mask > 0] = 1
    mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)
    mask2[mask2 > 0] = 1
    return image, mask, mask2

def extract_img(image, mask):
    return cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))

def process_channel(channel):
    channel[channel>0]=1
    channel = cv2.dilate(channel, np.ones((3,3), np.uint8), iterations=1)
    channel = ndi.binary_fill_holes(channel)
    # channel = cv2.erode(channel, np.ones((3,3), np.uint8), iterations=5)
    channel = cv2.morphologyEx(channel.astype(np.uint8), cv2.MORPH_HITMISS, np.ones((3,3), np.uint8), iterations=5)
    channel = cv2.morphologyEx(channel.astype(np.uint8), cv2.MORPH_CROSS, np.ones((3,3), np.uint8), iterations=5)
#     channel = cv2.morphologyEx(channel.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8), iterations=5)
    channel = cv2.morphologyEx(channel.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=5)
    channel = ndi.binary_fill_holes(channel)
    channel = cv2.morphologyEx(channel.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=20)
    # channel = cv2.morphologyEx(channel.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=5)
    # channel = cv2.morphologyEx(channel.astype(np.uint8), cv2.MORPH_CROSS, np.ones((3,3), np.uint8), iterations=5)
    return channel

# def plot_channels(image, channels):
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#     for i, (channel, name) in enumerate(zip(cv2.split(channels), ["C1", "C2", "C3"])):
#         channel = process_channel(channel)
#         channel = extract_img(image, channel)
#         axes[i].imshow(channel)
#         axes[i].set_title(f'{name} Channel ({channel.min()}-{channel.max()})')
#         axes[i].axis("off")
#     plt.tight_layout()
#     plt.show()

def process_rf_mask(mask):
    mask = mask.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CROSS, np.ones((3,3), np.uint8), iterations=5)
    mask = ndi.binary_fill_holes(mask)
    # mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=5)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=5)
    return mask.astype(np.uint8)

def plot_current_masks(image, mask, mask2, title_prefix="", save_dir='stain_sep_results' ,image_idx="0"):
    # fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    os.makedirs(f"{save_dir}/{image_idx}", exist_ok=True)
    cv2.imwrite(f"{save_dir}/{image_idx}/raw.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    proc_mask = extract_img(image, mask)
    cv2.imwrite(f"{save_dir}/{image_idx}/proc_mask.jpg", cv2.cvtColor(proc_mask, cv2.COLOR_RGB2BGR))
    # axes[0].imshow(proc_mask)
    # axes[0].set_title("Processed Mask")
    # axes[0].axis("off")
    rf_mask = process_rf_mask(mask2)
    rf_mask = extract_img(image, rf_mask)
    cv2.imwrite(f"{save_dir}/{image_idx}/rf_mask.jpg", cv2.cvtColor(rf_mask, cv2.COLOR_RGB2BGR))
    # axes[1].imshow(rf_mask)
    # axes[1].set_title("Raw Mask")
    # axes[1].axis("off")
    # plt.tight_layout()
    # plt.show()
    
def get_stain_vector(StainingVectorID):
       """
       StainingVectorID: Index
       1  = "H&E"
       2  = "H&E 2"
       3  = "H DAB"
       4  = "H&E DAB"
       5  = "NBT/BCIP Red Counterstain II"
       6  = "H DAB NewFuchsin"
       7  = "H HRP-Green NewFuchsin"
       8  = "Feulgen LightGreen"
       9  = "Giemsa"
       10 = "FastRed FastBlue DAB"
       11 = "Methyl Green DAB"
       12 = "H AEC"
       13 = "Azan-Mallory"
       14 = "Masson Trichrome"
       15 = "Alcian Blue & H"
       16 = "H PAS"
       17 = "Brilliant_Blue"
       18 = "AstraBlue Fuchsin"
       19 = "RGB"
       20 = "ROI Based Extraction"
       """
       if StainingVectorID==1:
              MODx = [0.644211, 0.092789, 0];
              MODy = [0.716556, 0.954111, 0];
              MODz = [0.266844, 0.283111, 0];
       elif StainingVectorID==2:
              MODx = [0.49015734, 0.04615336, 0];
              MODy = [0.76897085, 0.8420684, 0];
              MODz = [0.41040173, 0.5373925, 0];
       elif StainingVectorID==3:
              MODx = [0.650, 0.268, 0];
              MODy = [0.704, 0.570, 0];
              MODz = [0.286, 0.776, 0];
       elif StainingVectorID==4:
              MODx = [0.650, 0.072, 0.268];
              MODy = [0.704, 0.990, 0.570];
              MODz = [0.286, 0.105, 0.776];
       elif StainingVectorID==5:
              MODx = [0.62302786, 0.073615186, 0.7369498];
              MODy = [0.697869, 0.79345673, 0.0010];
              MODz = [0.3532918, 0.6041582, 0.6759475];
       elif StainingVectorID==6:
              MODx = [0.5625407925, 0.26503363, 0.0777851125];
              MODy = [0.70450559, 0.68898016, 0.804293475];
              MODz = [0.4308375625, 0.674584, 0.5886050475];
       elif StainingVectorID==7:
              MODx = [0.8098939567, 0.0777851125, 0.0];
              MODy = [0.4488181033, 0.804293475, 0.0];
              MODz = [0.3714423567, 0.5886050475, 0.0];
       elif StainingVectorID==8:
              MODx = [0.46420921, 0.94705542, 0.0];
              MODy = [0.83008335, 0.25373821, 0.0];
              MODz = [0.30827187, 0.19650764, 0.0];
       elif StainingVectorID==9:
              MODx = [0.834750233, 0.092789, 0.0];
              MODy = [0.513556283, 0.954111, 0.0];
              MODz = [0.196330403, 0.283111, 0.0];
       elif StainingVectorID==10:
              MODx = [0.21393921, 0.74890292, 0.268];
              MODy = [0.85112669, 0.60624161, 0.570];
              MODz = [0.47794022, 0.26731082, 0.776];
       elif StainingVectorID==11:
              MODx = [0.98003, 0.268, 0.0];
              MODy = [0.144316, 0.570, 0.0];
              MODz = [0.133146, 0.776, 0.0];
       elif StainingVectorID==12:
              MODx = [0.650, 0.2743, 0.0];
              MODy = [0.704, 0.6796, 0.0];
              MODz = [0.286, 0.6803, 0.0];
       elif StainingVectorID==13:
              MODx = [0.853033, 0.09289875, 0.10732849];
              MODy = [0.508733, 0.8662008, 0.36765403];
              MODz = [0.112656, 0.49098468, 0.9237484];
       elif StainingVectorID==14:
              MODx = [0.7995107, 0.09997159, 0.0];
              MODy = [0.5913521, 0.73738605, 0.0];
              MODz = [0.10528667, 0.6680326, 0.0];
       elif StainingVectorID==15:
              MODx = [0.874622, 0.552556, 0.0];
              MODy = [0.457711, 0.7544, 0.0];
              MODz = [0.158256, 0.353744, 0.0];
       elif StainingVectorID==16:
              MODx = [0.644211, 0.175411, 0.0];
              MODy = [0.716556, 0.972178, 0.0];
              MODz = [0.266844, 0.154589, 0.0];
       elif StainingVectorID==17:
              MODx = [0.31465548, 0.383573, 0.7433543];
              MODy = [0.6602395, 0.5271141, 0.51731443];
              MODz = [0.68196464, 0.7583024, 0.4240403];
       elif StainingVectorID==18:
              MODx = [0.92045766, 0.13336428, 0.0];
              MODy = [0.35425216, 0.8301452, 0.0];
              MODz = [0.16511545, 0.5413621, 0.0];
       elif StainingVectorID==19:
              MODx = [0.001, 1.0, 1.0];
              MODy = [1.0, 0.001, 1.0];
              MODz = [1.0, 1.0, 0.001];
       elif StainingVectorID==20:
              MODx =[ 0.22777562, 0.55556387, 0.7996668 ]
              MODy =[0.035662863,  0.08113831, 0.9960646]
              MODz =[ 0.7617831, 0.6478321 ,  0.001]
       else:
              MODx = [1.0, 0.0, 0.0];
              MODy = [0.0, 1.0, 0.0];
              MODz = [0.0, 0.0, 1.0];
       eps = 1e-5
       stain_vectors = np.array([MODx, MODy, MODz]).T
       # Check 2nd Channel is missing
       if all([x==0.0 for x in stain_vectors[1,:]]):
              stain_vectors[1,:] = stain_vectors[0,:][[2,0,1]]
       # Check 3rd Channel is missing
       if all([x==0.0 for x in stain_vectors[2,:]]):
              stain_vectors[2,:] = np.cross(stain_vectors[0,:] , stain_vectors[1,:])
       return np.linalg.inv(stain_vectors)

def background_removal_dog(brightfield_img, kernel1=15, kernel2=50):
    """Remove background using the Difference of Gaussian (DoG) method."""
    # Apply Gaussian blur with two different kernel sizes
    blurred1 = cv2.GaussianBlur(brightfield_img, (kernel1, kernel1), 0)
    blurred2 = cv2.GaussianBlur(brightfield_img, (kernel2, kernel2), 0)
    
    # Subtract the two blurred images (Difference of Gaussians)
    dog_result = blurred1 - blurred2
    
    # Threshold to isolate the foreground
    _, binarized_foreground = cv2.threshold(dog_result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    
    return binarized_foreground


def plot_channels(image, channels, axes, title_prefix="", save_dir='stain_sep_results', image_idx=str(0)):
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    import os
    save_dir = os.path.join(save_dir, image_idx)
    os.makedirs(save_dir, exist_ok=True)
    for i, (channel, name) in enumerate(zip(cv2.split(channels), ["C1", "C2", "C3"])):
        channel = process_channel(channel)
        channel = extract_img(image, channel)
        fname = f"{title_prefix}_{name}.png"
        cv2.imwrite(os.path.join(save_dir, fname), cv2.cvtColor(channel, cv2.COLOR_RGB2BGR))
        # axes[i].imshow(channel)
        # axes[i].set_title(f'{title_prefix} {name} ({channel.min()}-{channel.max()})')
        # axes[i].axis("off")



def save_stain_sep(image_idx):
    try:
        plt.ioff()
        image, mask, mask2 = read_img_mask(image_idx, original=False)
        stain_refs = [ahx_from_rgb,hpx_from_rgb,bpx_from_rgb,bro_from_rgb,hax_from_rgb,gdx_from_rgb,rbd_from_rgb,bex_from_rgb,fgx_from_rgb,hdx_from_rgb,hed_from_rgb]
        stain_ref_names = ["ahx_from_rgb","hpx_from_rgb","bpx_from_rgb","bro_from_rgb","hax_from_rgb","gdx_from_rgb","rbd_from_rgb","bex_from_rgb","fgx_from_rgb","hdx_from_rgb","hed_from_rgb"]
        stain_refs+=[get_stain_vector(i) for i in range(1,21)]
        stain_ref_names+=["StainingVectorID_"+str(i) for i in range(1,21)]
        # fig, ax = plt.subplots(31,3, figsize=(15, 90))
        plot_current_masks(image, mask, mask2, title_prefix="Processed", save_dir='stain_sep_results' ,image_idx=str(image_idx).zfill(4))
        for i,stain_ref in enumerate(stain_refs):
            stain = separate_stains(image, stain_ref)
            plot_channels(image, stain, None, title_prefix=stain_ref_names[i], save_dir='stain_sep_results' ,image_idx=str(image_idx).zfill(4))
        # plt.tight_layout()
        # plt.savefig(f"stain_separation/{image_idx}.png")
        # plt.close('all')
    except Exception as e:
          print(f"Error in {image_idx}: {e}")



if __name__ == "__main__":
      import os
      from tqdm import tqdm
      img_idxx = os.listdir('rf_trained_pred/retinex/')
      img_idxx = [int(i.split('_')[0]) for i in img_idxx]
      img_idxx = sorted(img_idxx)
    #   img_idxx = img_idxx[::10]
      _ = Parallel(n_jobs=48)(delayed(save_stain_sep)(i) for i in tqdm(img_idxx))