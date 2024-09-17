import re
import os
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm
from .utils import make_slide
from .slide import BFIWSlide

class BFIWReg:
    def __init__(self, src_bfiw_dir, src_bfi_dir, dest_dir, ref_idx, bfiw_regex, bfi_regex) -> None:
        self.src_bfiw_dir = src_bfiw_dir
        self.dest_dir = dest_dir
        self.ref_idx = ref_idx
        bfiw_imgs = os.listdir(self.src_bfiw_dir)
        bfiw_imgs = sorted(bfiw_imgs, key=lambda x: int(bfiw_regex.match(x).group(1))) # type: ignore
        bfiw_imgs_ordered = {bfiw_regex.match(x).group(1).zfill(4): os.path.join(self.src_bfiw_dir, x) for x in bfiw_imgs} # type: ignore
        self.src_bfi_dir = src_bfi_dir
        bfi_imgs = os.listdir(self.src_bfi_dir)
        bfi_imgs = sorted(bfi_imgs, key=lambda x: int(bfi_regex.match(x).group(1)))
        bfi_imgs_ordered = {bfi_regex.match(x).group(1).zfill(4): os.path.join(self.src_bfi_dir, x) for x in bfi_imgs}
        # for img in imgs:
        #     section_num = int(regex.match(img).group(1)) # type: ignore
        #     section_id = str(section_num)
        #     section_id_digits = len(section_id)
        #     if section_id_digits < 4:
        #         section_id = "0" * (4 - section_id_digits) + str(section_num)
        #     imgs_ordered[section_id] = os.path.join(self.src_dir, img)
        self.bfiw_imgs = bfiw_imgs_ordered
        if ref_idx not in self.bfiw_imgs:
            raise ValueError("Reference index not found in the image list")
        self.img_items = list(self.bfiw_imgs.items())
        self.bfi_img_items = list(bfi_imgs_ordered.items())
        self.img_items = [(key, img, bfi_imgs_ordered[key]) for key, img in self.img_items if key in bfi_imgs_ordered]

        # if ref_idx not in dict(self.img_items):
        #     self.img_items.append((ref_idx, self.imgs[ref_idx]))
        

    def register(self):
        self.slides_ = Parallel(n_jobs=32)(
            delayed(make_slide)(bfiw_img, bfi_img, key) for key, bfiw_img, bfi_img in tqdm(self.img_items)
        )
        self.slides: dict[str, BFIWSlide] = {}
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
        if not os.path.exists(os.path.join(self.dest_dir, "bfiw_original")):
            os.makedirs(os.path.join(self.dest_dir, "bfiw_original"), exist_ok=True)
        if not os.path.exists(os.path.join(self.dest_dir, "bfiw_msr")):
            os.makedirs(os.path.join(self.dest_dir, "bfiw_msr"), exist_ok=True)
        if not os.path.exists(os.path.join(self.dest_dir, "bfi_original")):
            os.makedirs(os.path.join(self.dest_dir, "bfi_original"), exist_ok=True)
        if not os.path.exists(os.path.join(self.dest_dir, "bfi_msr")):
            os.makedirs(os.path.join(self.dest_dir, "bfi_msr"), exist_ok=True)
        for key, slide in tqdm(self.slides.items()):
            cv2.imwrite(
                os.path.join(self.dest_dir, f"bfiw_original/{key}.jpg"),
                cv2.cvtColor(slide.bfiw_img, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(self.dest_dir, f"bfiw_msr/{key}_msr.jpg"),
                cv2.cvtColor(slide.msr_bfiw_img, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(self.dest_dir, f"bfi_original/{key}.jpg"),
                cv2.cvtColor(slide.bfi_img, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(self.dest_dir, f"bfi_msr/{key}_msr.jpg"),
                cv2.cvtColor(slide.msr_bfi_img, cv2.COLOR_RGB2BGR),
            )
