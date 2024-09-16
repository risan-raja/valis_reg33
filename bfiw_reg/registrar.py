import re
import os
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm
from .utils import make_slide

class BFIWReg:
    def __init__(self, src_dir, dest_dir, ref_idx, regex) -> None:
        self.src_dir = src_dir
        self.dest_dir = dest_dir
        self.ref_idx = ref_idx
        imgs = os.listdir(self.src_dir)
        imgs = sorted(imgs, key=lambda x: int(regex.match(x).group(1))) # type: ignore
        imgs_ordered = {regex.match(x).group(1).zfill(4): os.path.join(self.src_dir, x) for x in imgs} # type: ignore
        # for img in imgs:
        #     section_num = int(regex.match(img).group(1)) # type: ignore
        #     section_id = str(section_num)
        #     section_id_digits = len(section_id)
        #     if section_id_digits < 4:
        #         section_id = "0" * (4 - section_id_digits) + str(section_num)
        #     imgs_ordered[section_id] = os.path.join(self.src_dir, img)
        self.imgs = imgs_ordered
        if ref_idx not in self.imgs:
            raise ValueError("Reference index not found in the image list")
        self.img_items = list(self.imgs.items())
        if ref_idx not in dict(self.img_items):
            self.img_items.append((ref_idx, self.imgs[ref_idx]))
        

    def register(self):
        self.slides_ = Parallel(n_jobs=32)(
            delayed(make_slide)(img, key) for key, img in tqdm(self.img_items)
        )
        self.slides = {}
        for slide in self.slides_:
            self.slides.update(slide) # type: ignore
        self.ref_slide = self.slides[self.ref_idx]
        self.ref_slide.is_ref = True
        print("Applying Reference Slide Mask to all slides")
        for key, slide in tqdm(self.slides.items()):
            if key == self.ref_idx:
                continue
            slide.apply_mask(self.ref_slide.mask)
        self.ref_slide.get_block_contours()
        self.ref_crop = self.ref_slide.block_crop
        print("Applying Reference Slide Crop to all slides")
        for key, slide in tqdm(self.slides.items()):
            slide.apply_crop(self.ref_crop)
        print("Applying Own Slide Block Crop to all slides")
        for key, slide in tqdm(self.slides.items()):
            slide.get_block_contours()
            slide.apply_block_crop()


    def save_output(self):
        print("Saving `img` and `msr_img` of all slides")
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir, exist_ok=True)
        if not os.path.exists(os.path.join(self.dest_dir, "msrcr")):
            os.makedirs(os.path.join(self.dest_dir, "msrcr"), exist_ok=True)
        if not os.path.exists(os.path.join(self.dest_dir, "original")):
            os.makedirs(os.path.join(self.dest_dir, "original"), exist_ok=True)
        for key, slide in tqdm(self.slides.items()):
            cv2.imwrite(
                os.path.join(self.dest_dir, f"original/{key}.jpg"),
                cv2.cvtColor(slide.img, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(self.dest_dir, f"msrcr/{key}_msrcr.jpg"),
                cv2.cvtColor(slide.msr_img, cv2.COLOR_RGB2BGR),
            )
