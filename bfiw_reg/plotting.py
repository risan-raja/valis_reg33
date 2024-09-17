from .slide import BFIWSlide
from .utils import predict_mask
import matplotlib.pyplot as plt
import cv2

def visualize_model_prediction(slide: BFIWSlide , model):
    final_mask, new_mask = predict_mask(slide, model)
    fig, ax = plt.subplots(1, 5, figsize=(20, 10))
    ax[0].imshow(slide.msr_bfiw_img)
    ax[0].set_title("Slide Image")
    ax[0].axis('off')
    ax[1].imshow(slide.mask)
    ax[1].set_title("Slide Mask")
    ax[1].axis('off')
    ax[2].imshow(new_mask.get())
    ax[2].set_title("Predicted Mask")
    ax[2].axis('off')
    ax[3].imshow(final_mask)
    ax[3].set_title("Post Processed Mask")
    ax[3].axis('off')
    cropped_img = cv2.bitwise_and(slide.msr_bfiw_img, slide.msr_bfiw_img, mask=final_mask) # type: ignore
    cropped_img[final_mask==0] = [255, 255, 255]
    ax[4].imshow(cropped_img)
    ax[4].set_title("Final Masked Image")
    plt.show()