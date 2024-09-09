import os
from tqdm import tqdm
from .utils import predict_mask


def extract_tissue(reg, fil_model, save_dir):
    """
    Extracts tissue from all slides in the registration object.
    Args:
        reg (BFIWReg): The registration object containing all slides.
        fil_model: The filter model used for prediction.
    Returns:
        final_masks: A dictionary containing the final masks for each slide.
    """
    final_masks = {}
    new_masks = {}
    os.makedirs(save_dir, exist_ok=True)
    retinex_dir = os.path.join(save_dir, "retinex")
    original_dir = os.path.join(save_dir, "original")
    mask_dir = os.path.join(save_dir, "processed_mask")
    raw_mask_dir = os.path.join(save_dir, "raw_mask")
    os.makedirs(retinex_dir, exist_ok=True)
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(raw_mask_dir, exist_ok=True)
    for key, slide in tqdm(reg.slides.items()):
        final_mask, new_mask = predict_mask(slide, fil_model)
        final_masks[key] = final_mask
        new_masks[key] = new_mask
        retinex_img = cv2.bitwise_and(slide.msr_img, slide.msr_img, mask=final_mask) # type: ignore
        retinex_img[final_mask==0] = [255, 255, 255]
        retinex_img = cv2.cvtColor(retinex_img, cv2.COLOR_RGB2BGR) # type: ignore
        original_img = cv2.bitwise_and(slide.img, slide.img, mask=final_mask) # type: ignore
        original_img[final_mask==0] = [255, 255, 255]
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR) # type: ignore
        cv2.imwrite( # type: ignore
            os.path.join(retinex_dir, f"{key}_retinex.jpg"), 
            retinex_img,
        )
        cv2.imwrite( # type: ignore
            os.path.join(original_dir, f"{key}_original.jpg"), 
            original_img,
        )
        cv2.imwrite( # type: ignore
            os.path.join(mask_dir, f"{key}_mask.jpg"), 
            final_mask,
        )
        cv2.imwrite( # type: ignore
            os.path.join(raw_mask_dir, f"{key}_raw_mask.jpg"), 
            new_mask.get(),
        )
    return{'final': final_masks, 'raw': new_masks}