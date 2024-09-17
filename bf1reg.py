from bfiw_reg.registrar import BFIWReg
import re

reg = BFIWReg(src_dir='244_BFIW/BFI', dest_dir='244_BFI_reg/BFI',ref_idx='1608', regex= re.compile(r'B_244-ST_BFI-SE_(\d+).jpg'))
reg.register()
reg.save_output()