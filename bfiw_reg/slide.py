import cv2
import numpy as np
import ants
from .retinex import msrcr

class BFIWSlide:
    def __init__(self, slide_path, key=None, is_ref=False):
        self.slide_path = slide_path
        self.key = key
        self.img = cv2.imread(slide_path)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.msr_img = None
        self.msr_img_gray = None
        self.mask = None
        self.is_ref = is_ref
        self.apply_msrcr()
        self.get_mask()
        self.apply_mask(self.mask)

    def apply_msrcr(self):
        self.msr_img = msrcr(self.img)
        self.msr_img_gray = cv2.cvtColor(self.msr_img, cv2.COLOR_RGB2GRAY) # type: ignore

    def apply_mask(self, mask):
        self.temp_img = (np.ones_like(self.img) * 255).astype(np.uint8) # type: ignore
        self.temp_img[mask == 1] = self.img[mask == 1]
        self.img = self.temp_img
        self.temp_img = (np.ones_like(self.img) * 255).astype(np.uint8)
        self.temp_img[mask == 1] = self.msr_img[mask == 1] # type: ignore
        self.msr_img = self.temp_img

    def get_mask(self):
        if self.msr_img_gray is None:
            self.apply_msrcr()
        sample_ants = ants.from_numpy(self.msr_img_gray)
        self.mask = sample_ants.get_mask(cleanup=4).numpy().astype(np.uint8) # type: ignore
        # return self.mask
    
    def apply_crop(self, crop):
        self.img = self.img[crop[0]:crop[1], crop[2]:crop[3]]
        self.msr_img = self.msr_img[crop[0]:crop[1], crop[2]:crop[3]] # type: ignore
        self.mask = self.mask[crop[0]:crop[1], crop[2]:crop[3]] # type: ignore
    
    def get_block_contours(self):
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # type: ignore
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        self.block_contour=contours[0]
        self.block_bbox =  cv2.boundingRect(self.block_contour)
        self.block_crop = (self.block_bbox[1], self.block_bbox[1]+self.block_bbox[3], self.block_bbox[0], self.block_bbox[0]+self.block_bbox[2])

    def apply_block_crop(self):
        self.apply_crop(self.block_crop)