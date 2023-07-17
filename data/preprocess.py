"""
Preprocessing of the input cortical surfaces and brain MRI volumes.
The goal is to establish correspondence between the input surfaces and volumes.
To reduce the GPU memory cost, we clip the input MRI volume and
translate the surfaces accordingly.
For a new dataset, if the MRI is aligned to MNI-152 space 
and the ground truth is generated by FreeSurfer,
we recommend to set data_name='adni' with slight modification on the image clipping.
Note: it is tedious to tune the transformation of the surface to match the volume
for a new dataset, but the matching is important to make everything work.
"""
#from eval import seg2surf
#160, 208, 208
#208,208,164

#153x180x150

#150,180,153




#(224, 251, 203)
#203 251 224 
#101 125 110
import numpy as np

def process_volume(x, data_name='fetal'):
    if data_name == 'fetal':
        x = np.pad(x, ((2,2),(0,0),(0,0)), 'constant', constant_values=0)
        print("x de preprocess",x.shape)
        return x[None].copy()
    else:
        raise ValueError("data_name should be in ['fetal']")
        
        
        

def process_surface(v, f, data_name='fetal'):
    f = f.astype(np.float32)

    if data_name == 'fetal':
        v = v[:,[2,1,0]].copy()
        f = f[:,[2,1,0]].copy()
        # normalize to [-1, 1] par division par 2
        v = (v - [101, 125, 110]) / 125
        print("Size of v:", v.shape)
        print("Size of f:", f.shape)
        print("Data has been normalized. Min value:", np.min(v), "Max value:", np.max(v))

    
        
    else:
        raise ValueError("data_name should be in ['fetal']")

    return v, f



def process_surface_inverse(v, f, data_name='fetal'):
    if data_name == 'fetal':
        v = v * 125 + [101, 1125, 110]
        v = v[:,[2,1,0]].copy()
        f = f[:,[2,1,0]].copy()
        
    else:
        raise ValueError("data_name should be in ['fetal']")

    return v, f

#M ercredi 10h
#n ormalisation avec D


"""

    def seg_surface(self, seg):
        if self.data_name == 'fetal':
            seg = seg * 104 + [104, 104, 78]
            seg= seg[:, [2, 1, 0]].copy()
            #f = f[:, [2, 1, 0]].copy()
        else:
            raise ValueError("data_name should be in ['fetal']")

        return seg

    def inverse_preprocess_segmentation(self, v):
        if self.data_name == 'fetal':
            seg = seg * 104 + [104, 104, 78]
            seg = seg[:, [2, 1, 0]].copy()
            #f = f[:, [2, 1, 0]].copy()
        else:
            raise ValueError("data_name should be in ['fetal']")

        return seg 

"""