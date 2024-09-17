from bfiw_reg.registrar import BFIWReg
import re

reg = BFIWReg(src_bfiw_dir='244_BFIW/BFIW',src_bfi_dir='244_BFIW/BFI', dest_dir='244_BFI_reg_multi/BFI',ref_idx='1608', bfiw_regex= re.compile(r'B_244-ST_BFIW-SE_(\d+).jpg'), bfi_regex= re.compile(r'B_244-ST_BFI-SE_(\d+).jpg'))
reg.register()
reg.save_output()