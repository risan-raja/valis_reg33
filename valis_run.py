import cv2
import os
import re
import matplotlib.pyplot as plt
import ants
import valis
from valis import registration, feature_detectors, non_rigid_registrars, affine_optimizer
from valis import registration

non_rigid_registrar_cls = non_rigid_registrars.SimpleElastixWarper
img_dir = '/storage/keerthi_data/BFIW_processed/'
imgs = os.listdir(img_dir)
regex = re.compile(r".*-SE_(\d+)_processed.jpg")
imgs = sorted(imgs, key=lambda x: int(regex.match(x).group(1)))
imgs_ordered = {}
for img in imgs:
    section_num = int(regex.match(img).group(1))
    section_id = str(section_num)
    section_id_digits = len(section_id)
    if section_id_digits <4:
        section_id = '0'*(4-section_id_digits) + str(section_num)
    imgs_ordered[ os.path.join(img_dir, img)] = section_id


slide_src_dir = '/storage/keerthi_data/BFIW_processed/'
results_dir = '/storage/valis_reg/valis_process'
os.makedirs(results_dir, exist_ok=True)
reference_slide = "/storage/keerthi_data/BFIW_processed/B_213-ST_BFIW-SE_1606_processed.jpg"
registrar = registration.Valis(slide_src_dir, results_dir, reference_img_f=reference_slide)


registrar.name_dict = imgs_ordered
registrar.imgs_ordered=True
registrar.crop='overlap'
rigid_registrar, non_rigid_registrar, error_df = registrar.register()