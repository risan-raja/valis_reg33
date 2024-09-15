from valis import registration

src_dir = 'registered_imgs'
dst_dir = 'registered_imgs_valis'
register_all = registration.Valis(src_dir, dst_dir,non_rigid_registrar_cls=None)
import os
file_names = {f'{src_dir}/{f}':str((1575-int(os.path.splitext(f)[0]))).zfill(4) for f in os.listdir(src_dir) }
ref_slide_idx = 'registered_imgs/0.jpg'

file_names = dict(sorted(file_names.items(), key=lambda item: item[1]))

register_all.imgs_ordered=True

register_all.name_dict = file_names

register_all.register()