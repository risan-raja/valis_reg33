import json
from skimage.morphology import binary_dilation, area_closing, disk

import glob
import requests
import os
from skimage.io import imread

import numpy as np
from skimage import draw as skdraw

import sys
# sys.path.append('/workspace/bb/ap7_datareleasework/')

from correction_functions import apply_correction

from atlas_functions import (
    calculate_representative_point, poly_orientation,
)

from trs_functions import warp_rgb, crop_or_pad, transform

import io
from PIL import Image
from tqdm import tqdm
import pandas as pd
import shapely

def get_image_from_url(imgurl):
    try:
        resp = requests.get(imgurl)
        if resp.status_code==200:
            imgbytes = io.BytesIO(resp.content)
            return np.array(Image.open(imgbytes)) #,mode='r',formats=('jpeg','png')))
    except:
        print('iip get fail')
        return None

def get_reduced_coords(coords,mpp=32):
    # coords should be x,-y (ol coords)
    native_mpp=0.5
    coords_small = coords*native_mpp/mpp
    coords_small[:,1]*=-1
    return coords_small # x,y

class APIfunctions:
    def __init__(self,bsid:str, trscsvfile, ontoid=189, localdir='/code/dev/gjcache'):
        
        self.bsid=bsid
        self.ontoid=ontoid
        self.localdir=localdir+'/'+bsid
        os.makedirs(self.localdir, exist_ok=True)
        self.endpoint='http://apollo2.humanbrain.in:8000'
        self.authtuple=('admin','admin')
        sslist = requests.get(self.endpoint+'/qc/SeriesSet/',auth=self.authtuple).json()
        ssdict = {elt['biosample']:(elt['id'],elt['name']) for elt in sslist}
        self.ssid,self.bsname = ssdict[int(bsid)]

        tnail_list=requests.get(self.endpoint+'/GW/getBrainViewerDetails/IIT/V1/SS-%d?public=1' % self.ssid).json()
        nisl_list = tnail_list['thumbNail']['NISSL']
        self.nisl_dict={elt['position_index']:elt for elt in nisl_list}
        self.atlasurllist = []
        self.secdata = {}
        self.trsdata = pd.read_csv(trscsvfile)

    def _get_trsdata(self,secno:str):
        rec = self.trsdata[self.trsdata['position_index'] == int(secno)].to_dict(orient='records')[0]
        # tr16=json.loads(trsdata['tr16'])
        for key in ('tr16',): #'tr32':
            val = rec[key].replace('[','').replace(']','').strip().split(' ')
            rec[key]=[float(val[0]), float(val[-1])]
        tr16 = rec['tr16']
        # tr32 = rec['tr32']
        rot = rec['rotation']
        return {'tr16':tr16, 'rotation':rot } #, 'width':trsdata['width'],'height':trsdata['height']}
    
    def _setup_atlasurllist(self):
        print('setting up atlas url list')
        localfname = self.localdir+'/atlasurllist_%s_%d.json' % (self.bsid,self.ontoid)
        if not os.path.exists(localfname):
            self.atlasurllist = requests.get(self.endpoint+'/BR/getAtlasDataUrlsAll/%d/%s' % (self.ontoid,self.ssid),auth=self.authtuple).json() # ontoid,ssid
            
            os.makedirs(os.path.dirname(localfname),exist_ok=True)
            
            json.dump(self.atlasurllist,open(localfname,'wt'))
        else:
            self.atlasurllist = json.load(open(localfname))

    def _setup_gjdata(self):
        os.makedirs(self.localdir+'/'+str(self.ontoid),exist_ok=True)

        urldictpath = self.localdir+'/atlasurldict_%s_%d.json' % (self.bsid, self.ontoid)
        urllistpath = self.localdir+'/atlasurllist_%s_%d.json' % (self.bsid, self.ontoid)

        print('setting up secdata')
        if  os.path.exists(urldictpath):
            
            atlasurldict = json.load(open(urldictpath))
            for secno, elturl in tqdm(atlasurldict.items()):
                gjpath=self.localdir+'/'+str(self.ontoid)+'/'+self.bsid+'_'+str(secno)+'.geojson'
                if not os.path.exists(gjpath):
                    data = requests.get(elturl,auth=self.authtuple).json()
                    sel.secdata[secno]={
                        'atlasgeojson':json.loads(data['atlasgeojson']),
                        'width':ad['width'],
                        'height':ad['height'],
                        'rigidrotation':ad['rigidrotation'],
                    }
                    json.dump(self.secdata[secno]['atlasgeojson'],
                              open(gjpath,'wt'))
                else:
                    data = json.load(open(gjpath))
                    self.secdata[secno]={
                        'atlasgeojson':data
                    }

        else:
            atlasurldict = {}
        
            for elt in tqdm(self.atlasurllist):
                
                data = requests.get(elt,auth=self.authtuple).json()
                ad=json.loads(data['accessdetails'])[0]
                secno = ad['url'].split('/')[-1].split('&')[0].split('_')[5]
                
                atlasurldict[secno]=elt
                
                self.secdata[secno]={
                    'atlasgeojson':json.loads(data['atlasgeojson']),
                    'width':ad['width'],
                    'height':ad['height'],
                    'rigidrotation':ad['rigidrotation']
                }
                
                json.dump(self.secdata[secno]['atlasgeojson'],open(self.localdir+'/'+str(self.ontoid)+'/'+self.bsid+'_'+str(secno)+'.geojson','wt'))
    
            print('writing url dict')
            json.dump(atlasurldict,open(self.localdir+'/atlasurldict_%s_%d.json' % (self.bsid, self.ontoid),'wt'))
                                  
    def get_nisslimg_api(self,secno:str,mpp=32):
        jp2path = self.nisl_dict[int(secno)]['jp2Path']
        native_mpp=0.5
        downsample = int(mpp/native_mpp)
        reduce = int(np.log2(downsample))
        pct = 100/(2**reduce)
        imgurl = "/".join(['https://apollo2.humanbrain.in/iipsrv/fcgi-bin/iipsrv.fcgi?IIIF=%s'%jp2path,'full','pct:%f'%pct, '0', 'default.jpg'])
        print(imgurl)
        img = get_image_from_url(imgurl)

        msksize = int(2000*32/mpp)
        if str(self.bsid)=="244" or str(self.bsid)=="142":
            msksize = int(2500*32/mpp)
        imshape = [msksize,msksize]
        
        trsdata = self._get_trsdata(secno)
        
        trsxfm = transform.SimilarityTransform(
            scale=1, 
            translation=np.array(trsdata['tr16'])*16/mpp, # from 16 micron to mpp micron
            rotation=-trsdata['rotation'])

        warped = warp_rgb(img, trsxfm.params, imshape)
        return warped

    def get_mask(self,secno:str, mpp=16, skiptrs=False):
        msksize = int(2000*32/mpp)
        if str(self.bsid)=="244" or str(self.bsid)=="142":
            msksize = int(2500*32/mpp)
        
        D ,_ = self.get_ontoid_to_featurelist(secno)
        # print("get_ontoid_to_featurelist")
        mskshape = [msksize,msksize]
        msk_whole = np.zeros(mskshape,bool)
        # print("msk_whole done")
        
        for rgnname in D:
            rgn = D[rgnname]
            msk_i = []
            for rid,obj in rgn.items():
                coords = obj['coordinates']
                coords_aligned = self.get_aligned_coords(secno, coords, skiptrs)
                coords_reduced = get_reduced_coords(coords_aligned, mpp)
                msk_i.append(skdraw.polygon2mask(mskshape, np.fliplr(coords_reduced))>0)
            msk_whole = msk_whole | np.logical_or.reduce(msk_i)
            # print("rgn done")

        return msk_whole

    def get_mask_stack(self,secno:str, mpp=16, skiptrs=False):
        msksize = int(2000*32/mpp)
        if str(self.bsid)=="244" or str(self.bsid)=="142":
            msksize = int(2500*32/mpp)
        
        D ,_ = self.get_ontoid_to_featurelist(secno)
        # print("get_ontoid_to_featurelist")
        mskshape = [msksize,msksize]
        msk_whole = np.zeros(mskshape,np.uint16)
        # print("msk_whole done")
        
        for rgnname in D:
            rgn = D[rgnname]
            # msk_i = []
            for rid,obj in rgn.items():
                coords = obj['coordinates']
                coords_aligned = self.get_aligned_coords(secno, coords, skiptrs)
                coords_reduced = get_reduced_coords(coords_aligned, mpp)
                msk_i = skdraw.polygon2mask(mskshape, np.fliplr(coords_reduced))>0
                undrawn = (msk_whole==0) & msk_i
                msk_whole[undrawn]=int(obj['ontoid'])
                # msk_whole = np.where(msk_i, int(obj['ontoid']), msk_whole)
            # print("rgn done")

        return msk_whole

    
        
    def get_gjdata(self, secno:str):
        gjpath=self.localdir+'/'+str(self.ontoid)+'/'+self.bsid+'_'+secno+'.geojson'
        urldictpath = self.localdir+'/atlasurldict_%s_%d.json' % (self.bsid, self.ontoid)
        urllistpath = self.localdir+'/atlasurllist_%s_%d.json' % (self.bsid, self.ontoid)
        
        if not os.path.exists(gjpath):
            if not os.path.exists(urldictpath):
                if not os.path.exists(urllistpath):
                    self._setup_atlasurllist()
                else:
                    self.atlasurllist = json.load(open(urllistpath))
                self._setup_gjdata()
                return self.secdata[secno]['atlasgeojson']
            else:
                elturl = json.load(open(urldictpath))[secno]
                data = requests.get(elturl,auth=self.authtuple).json()        
                return json.loads(data['atlasgeojson'])
        else:
            if secno not in self.secdata:
                gjdata = json.load(open(gjpath))
                self.secdata[secno]={'atlasgeojson':gjdata}
            else:
                gjdata = self.secdata[secno]['atlasgeojson']
                
            return gjdata
            
    def get_ontoid_to_featurelist(self,secno:str):
        # mpp=0.5 in this function
        
            
        data = self.get_gjdata(secno)

        ontoid_to_featurelist = {} 
        min_left_x = None
        max_right_x = None
        for feature in data["features"]:
            properties = feature.get("properties", {})
            data_props = properties.get("data", {})
            acronym = data_props.get("acronym", "No Acronym")
            polygon_coordinates = feature["geometry"]["coordinates"][0]
            ontoid = data_props['id']
            fid = feature.get("id", -1)
            color_hex_triplet = data_props.get("color_hex_triplet", "0000FF").lower()
            
            if not isinstance(polygon_coordinates, list) or len(polygon_coordinates) < 4:
                print(f"Skipping feature {fid}: invalid polygon coordinates")
                continue
        
            
            try:
                poly = shapely.Polygon(polygon_coordinates)
            except Exception as e:
                print(f"Skipping feature {fid} : {e}")
                continue
        
            details = {
                # 'length': len(polygon_coordinates),
                'area': poly.area,
                'perimeter':poly.length,
                'orientation': poly_orientation(poly),
                'centroid': poly.centroid.coords[0],
                'left':poly.bounds[0], 
                'right':poly.bounds[2], 
                'color_hex': color_hex_triplet,
                'coordinates':np.array(polygon_coordinates),
                'text':data_props.get('text',''),
                'acronym':acronym,
                'ontoid':ontoid
                # 'selected':0,
            }
            if ontoid in ontoid_to_featurelist:
                ontoid_to_featurelist[ontoid][fid] = details
            else:
                ontoid_to_featurelist[ontoid] = {fid: details}
    
            if min_left_x is None:
                min_left_x = details['left']
            else:
                min_left_x = min(min_left_x,details['left'])
                
            if max_right_x is None:
                max_right_x = details['right']
            else:
                max_right_x = max(max_right_x,details['right'])
    
        mid_x = (min_left_x + max_right_x)/2
        # print(min_left_x/64, max_right_x/64, mid_x/64)
                        
        return ontoid_to_featurelist, mid_x
    
    def get_aligned_coords(self, secno:str, rgn_coords, skiptrs=False):
        jsonrotation = self.secdata[secno]['atlasgeojson']['rotation']
        jp2_width = self.nisl_dict[int(secno)]['width']
        jp2_height = self.nisl_dict[int(secno)]['height']
        
        rgn_coords_corrected = apply_correction(rgn_coords, jsonrotation, jp2_width, jp2_height )

        trsdata = self._get_trsdata(secno)
        native_mpp = 0.5
        trsxfm = transform.SimilarityTransform(
            scale=1, 
            translation=np.array(trsdata['tr16'])*16/native_mpp, # from 16 micron to 0.5 micron
            rotation=-trsdata['rotation'])

        pts = np.array(rgn_coords_corrected)

        if not skiptrs:
            X = pts[:,0][...,np.newaxis]
            _Y = pts[:,1][...,np.newaxis]
            YX = (trsxfm(np.hstack([-_Y,X])))
            XY = np.fliplr(YX)
            rgn_coords_aligned = np.hstack([XY[:,0][...,np.newaxis], -XY[:,1][...,np.newaxis]]) # x,-y
        
            return rgn_coords_aligned
        else:
            return pts
            

class Nomenclature:
    def __init__(self):
        self.treenom = json.load(open('/data/keerthi/brainpubdata/sgbc-brainnomenclaturev01r06.json'))['msg'][0]['children']
        self.flatnom = json.load(open('/data/keerthi/brainpubdata/flatnom_189.json'))['msg'][0]['children']
        
        self.groups = {
            'HPF':['HPF'], # hippocampal formation
            'AMY_BN':['AMY','BN'], # amygdala + basal nucleus
            'Mig':['Lms','Rms','GE'], # migratory areas
            'HY':['HY'], # hypothalamus
            'TH':['TH'], # thalamus
            'MB':['MB'], # midbrain
            'HB':['HB'], # hind brain 
            'CB':['CB'], # cerebellum
            'dev':['dev'], # developmental
            'ft':['ft'], # fiber tracts
            'Vs':['Vs'], # ventricles
            'Ctx':['Ctx'] # fallback cortex 
        }

        self.subtrees = {k:[] for k in self.groups}

        self.onto_lookup = {} # id:(acronym,name,level,parentid)

        for elt in self.treenom:
            self._find_subtrees(elt)
            self.onto_lookup[elt['id']]=(elt['acronym'],elt['name'],0,0)
            if 'children' in elt:
                for child in elt['children']:
                    self.dft(child,1,elt['id'])

        for elt in self.flatnom:
            if elt['id'] not in self.onto_lookup:
                self.onto_lookup[elt['id']]=(elt['acronym'],elt['name'],-1,-1)
    
    def dft(self, elt, level, parentid):
        self.onto_lookup[elt['id']]=(elt['acronym'],elt['name'],level,parentid)
        if 'children' in elt:
            for child in elt['children']:
                self.dft(child,level+1,elt['id'])
        

    def _find_subtrees(self,elt):
        self._check_node(elt)
        if 'children' in elt:
            for child in elt['children']:
                self._find_subtrees(child)

    def _check_node(self,elt):
        for grpname, grpparents in self.groups.items():
            if elt['acronym'] in grpparents:
                self.subtrees[grpname].append(elt)
                    
    def get_group_by_name(self, rgnname):
        for grpname in self.subtrees:
            for subtr in self.subtrees[grpname]:
                if self._find_node_by_name(subtr,rgnname):
                    return grpname
        return None

    def _find_node_by_name(self,seednode, rgnname):
        if seednode['acronym']==rgnname:
            return True
        if 'children' in seednode:
            for child in seednode['children']:
                if self._find_node_by_name(child, rgnname):
                    return True
        return False

    def get_group_by_ontoid(self, ontoid):
        for grpname in self.subtrees:
            for subtr in self.subtrees[grpname]:
                if self._find_node_by_id(subtr, ontoid):
                    return grpname

    def _find_node_by_id(self, seednode, ontoid):
        if seednode['id']==ontoid:
            return True
        if 'children' in seednode:
            for child in seednode['children']:
                if self._find_node_by_id(child, ontoid):
                    return True
        return False
              
        
    def print_tree(self):
        for toplevel in self.treenom:
            # print(toplevel['id'],toplevel['name'])
            self.show_children(toplevel)
        
    def show_children(self,elt,level=0):
        lvlstr='[%d]'%level
        if '-' in elt['acronym']:
            parts = elt['acronym'].split('-')
            if parts[0] in ('SGL','MZ','CP','SP','IZ','SVZ','VZ'):
                lvlstr = '[*]'
                
        print(''.join(['\t']*level),lvlstr, elt['id'],elt['acronym'],elt['name'])
        if 'children' not in elt:
            pass
            # print('@no children')
        else:
            if len(elt['children'])>0:
                for child in elt['children']:
                    show_children(child,level+1)



import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from atlas_functions import get_raster_shape, check_touching

class Plotter:
    
    def __init__(self, apiobj,secno):
        self.apiobj = apiobj
        self.secno = secno
        self.bgsize = 2000 
        if self.apiobj.bsid == '244' or self.apiobj.bsid == '142':
            self.bgsize = 2500
        
        self.D,_ = self.apiobj.get_ontoid_to_featurelist(secno)
        
        self.area_suffix = ['FCTx','ORB', 'PAR', 'OCC','TEMP','INS','CING', 'ENT']
        # not included: 'PRESUB','PARA','POST','SUB','HPF','HIP','CTX'
        
        self.cmap_ctx = mpl.colormaps['viridis'].resampled(len(self.area_suffix)+1)
        
        self.layer_prefix = ["VZ", "SVZ", "IZ", "SP", "CP", "MZ", "SGL"]
        
        self.cortex_regions = {suf:[] for suf in self.area_suffix}
        self.layer_regions = {pre:[] for pre in self.layer_prefix}
        
        self.layer_colors = {}
        
        for ontoid in self.D:
            
            rgn = self.D[ontoid]
            for idv,obj in rgn.items():
                acro = obj['acronym']
                clr = obj['color_hex']
                coords = obj['coordinates']
                
                if self.name_is_cortex(acro):
                    layername,ctxname = acro.split('-')
                    
                    if ctxname in self.area_suffix:
                        self.cortex_regions[ctxname].append(coords)
        
                    if layername in self.layer_prefix:
                        self.layer_colors[layername]='#'+clr
                        self.layer_regions[layername].append(coords)

            

    def minimap_areas(self):
        fig=plt.figure(figsize=(1,1),dpi=300)

        
        plt.imshow(np.ones((self.bgsize//10,self.bgsize//10,3),np.uint8)*255) # 1/10 of thumbnail size (1/10 of 32 mpp = 320 mpp)
        
        legend_handles = []
        for ctxname,coordslist in self.cortex_regions.items():
            ctxid = self.area_suffix.index(ctxname)
            if len(coordslist)==0:
                continue
            coords_by_ctx = []
            for coords in coordslist:
                coords_aligned = self.apiobj.get_aligned_coords(self.secno,coords)
                coords_reduced = get_reduced_coords(coords_aligned, mpp=320)
                coords_by_ctx.append(coords_reduced) 
        
            clr = self.cmap_ctx(ctxid)
            for coords in coords_by_ctx:
                plt.fill(coords[:,0],coords[:,1],color=clr)
                
            legend_handles.append(mpatches.Patch(color=clr, label=ctxname))
        
        plt.axis('off')
        plt.legend(handles=legend_handles,ncols=1,loc='upper left',bbox_to_anchor=(1.1,0.9), fontsize=3, borderaxespad=0.)
        plt.title('Cortical areas',y=0.9,fontsize=5)
        # mmname = '/workspace/forref/fb_62_new_out/minimap_areas_%s_%s.png' % (biosampleid, secno)
        # plt.savefig(mmname,pad_inches=0,bbox_inches="tight",dpi=75)
        return plt.gcf()
        
        # plt.savefig(mmname,pad_inches=0,bbox_inches="tight",dpi=75)
        # return imread(mmname)[...,:3]

    def minimap_layers(self):
        fig = plt.figure(figsize=(1.5, 1.2), dpi=300)  # Increased width to 1.5
        plt.imshow(np.ones((self.bgsize // 10, self.bgsize // 10, 3), np.uint8) * 255)  # 1/10 of thumbnail size (1/10 of 32 mpp = 320 mpp)
        
        legend_handles = []
        
        for layername, coordslist in self.layer_regions.items():
            layerid = self.layer_prefix.index(layername)
            if len(coordslist) == 0:
                continue
            coords_by_layer = []
            for coords in coordslist:
                coords_aligned = self.apiobj.get_aligned_coords(self.secno, coords)
                coords_reduced = get_reduced_coords(coords_aligned, mpp=320)
                coords_by_layer.append(coords_reduced)
        
            clr = self.layer_colors[layername]
            for coords in coords_by_layer:
                plt.fill(coords[:, 0], coords[:, 1], color=clr)
                
            legend_handles.append(mpatches.Patch(color=clr, label=layername))
        
        plt.axis('off')
        plt.legend(handles=legend_handles, ncols=1, loc='upper left', bbox_to_anchor=(1.1, 0.9), fontsize=3, borderaxespad=0.)
        plt.title('Developmental cortical layers', y=0.9, fontsize=4)  # Reduced font size to 4
        return plt.gcf()


    def get_poly_map(self,coords_reduced):
        # coords_reduced should be 32mpp
        poly_map = np.zeros([self.bgsize,self.bgsize],bool) 
        rr,cc = skdraw.polygon(coords_reduced[:,1],coords_reduced[:,0])    
        poly_map[rr,cc]=True
        return poly_map

    def name_is_cortex(self, rgnname):
        if '-' in rgnname:
            parts = rgnname.split('-')
            if parts[0] in self.layer_prefix:
                return True
        return False

    