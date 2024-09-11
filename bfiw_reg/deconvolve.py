import numpy as np

def get_stain_vector(StainingVectorID):
       """
       StainingVectorID: Index
       1  = "H&E"
       2  = "H&E 2"
       3  = "H DAB"
       4  = "H&E DAB"
       5  = "NBT/BCIP Red Counterstain II"
       6  = "H DAB NewFuchsin"
       7  = "H HRP-Green NewFuchsin"
       8  = "Feulgen LightGreen"
       9  = "Giemsa"
       10 = "FastRed FastBlue DAB"
       11 = "Methyl Green DAB"
       12 = "H AEC"
       13 = "Azan-Mallory"
       14 = "Masson Trichrome"
       15 = "Alcian Blue & H"
       16 = "H PAS"
       17 = "Brilliant_Blue"
       18 = "AstraBlue Fuchsin"
       19 = "RGB"
       20 = "ROI Based Extraction"
       """
       if StainingVectorID==1:
              MODx = [0.644211, 0.092789, 0];
              MODy = [0.716556, 0.954111, 0];
              MODz = [0.266844, 0.283111, 0];
       elif StainingVectorID==2:
              MODx = [0.49015734, 0.04615336, 0];
              MODy = [0.76897085, 0.8420684, 0];
              MODz = [0.41040173, 0.5373925, 0];
       elif StainingVectorID==3:
              MODx = [0.650, 0.268, 0];
              MODy = [0.704, 0.570, 0];
              MODz = [0.286, 0.776, 0];
       elif StainingVectorID==4:
              MODx = [0.650, 0.072, 0.268];
              MODy = [0.704, 0.990, 0.570];
              MODz = [0.286, 0.105, 0.776];
       elif StainingVectorID==5:
              MODx = [0.62302786, 0.073615186, 0.7369498];
              MODy = [0.697869, 0.79345673, 0.0010];
              MODz = [0.3532918, 0.6041582, 0.6759475];
       elif StainingVectorID==6:
              MODx = [0.5625407925, 0.26503363, 0.0777851125];
              MODy = [0.70450559, 0.68898016, 0.804293475];
              MODz = [0.4308375625, 0.674584, 0.5886050475];
       elif StainingVectorID==7:
              MODx = [0.8098939567, 0.0777851125, 0.0];
              MODy = [0.4488181033, 0.804293475, 0.0];
              MODz = [0.3714423567, 0.5886050475, 0.0];
       elif StainingVectorID==8:
              MODx = [0.46420921, 0.94705542, 0.0];
              MODy = [0.83008335, 0.25373821, 0.0];
              MODz = [0.30827187, 0.19650764, 0.0];
       elif StainingVectorID==9:
              MODx = [0.834750233, 0.092789, 0.0];
              MODy = [0.513556283, 0.954111, 0.0];
              MODz = [0.196330403, 0.283111, 0.0];
       elif StainingVectorID==10:
              MODx = [0.21393921, 0.74890292, 0.268];
              MODy = [0.85112669, 0.60624161, 0.570];
              MODz = [0.47794022, 0.26731082, 0.776];
       elif StainingVectorID==11:
              MODx = [0.98003, 0.268, 0.0];
              MODy = [0.144316, 0.570, 0.0];
              MODz = [0.133146, 0.776, 0.0];
       elif StainingVectorID==12:
              MODx = [0.650, 0.2743, 0.0];
              MODy = [0.704, 0.6796, 0.0];
              MODz = [0.286, 0.6803, 0.0];
       elif StainingVectorID==13:
              MODx = [0.853033, 0.09289875, 0.10732849];
              MODy = [0.508733, 0.8662008, 0.36765403];
              MODz = [0.112656, 0.49098468, 0.9237484];
       elif StainingVectorID==14:
              MODx = [0.7995107, 0.09997159, 0.0];
              MODy = [0.5913521, 0.73738605, 0.0];
              MODz = [0.10528667, 0.6680326, 0.0];
       elif StainingVectorID==15:
              MODx = [0.874622, 0.552556, 0.0];
              MODy = [0.457711, 0.7544, 0.0];
              MODz = [0.158256, 0.353744, 0.0];
       elif StainingVectorID==16:
              MODx = [0.644211, 0.175411, 0.0];
              MODy = [0.716556, 0.972178, 0.0];
              MODz = [0.266844, 0.154589, 0.0];
       elif StainingVectorID==17:
              MODx = [0.31465548, 0.383573, 0.7433543];
              MODy = [0.6602395, 0.5271141, 0.51731443];
              MODz = [0.68196464, 0.7583024, 0.4240403];
       elif StainingVectorID==18:
              MODx = [0.92045766, 0.13336428, 0.0];
              MODy = [0.35425216, 0.8301452, 0.0];
              MODz = [0.16511545, 0.5413621, 0.0];
       elif StainingVectorID==19:
              MODx = [0.001, 1.0, 1.0];
              MODy = [1.0, 0.001, 1.0];
              MODz = [1.0, 1.0, 0.001];
       elif StainingVectorID==20:
              MODx =[ 0.22777562, 0.55556387, 0.7996668 ]
              MODy =[0.035662863,  0.08113831, 0.9960646]
              MODz =[ 0.7617831, 0.6478321 ,  0.001]
       else:
              MODx = [1.0, 0.0, 0.0];
              MODy = [0.0, 1.0, 0.0];
              MODz = [0.0, 0.0, 1.0];
       eps = 1e-5
       stain_vectors = np.array([MODx, MODy, MODz]).T
       # Check 2nd Channel is missing
       if all([x==0.0 for x in stain_vectors[1,:]]):
              stain_vectors[1,:] = stain_vectors[0,:][[2,0,1]]
       # Check 3rd Channel is missing
       if all([x==0.0 for x in stain_vectors[2,:]]):
              stain_vectors[2,:] = np.cross(stain_vectors[0,:] , stain_vectors[1,:])
       return np.linalg.inv(stain_vectors)