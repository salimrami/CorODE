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
import numpy as np

def process_volume(x, data_name='fetal'):
    if data_name == 'fetal':
        x = np.pad(x, ((2,2),(0,0),(0,0)), 'constant', constant_values=0)
        return x[None].copy()
    else:
        raise ValueError("data_name should be in ['fetal']")

def process_surface(v, f, data_name='fetal'):
    f = f.astype(np.float32)

    if data_name == 'fetal':
        v = v[:,[2,1,0]].copy()
        f = f[:,[2,1,0]].copy()
        # normalize to [-1, 1]
        v = (v - [104, 104, 78]) / 104
    else:
        raise ValueError("data_name should be in ['fetal']")

    return v, f

def process_surface_inverse(v, f, data_name='fetal'):
    if data_name == 'fetal':
        v = v * 104 + [104, 104, 78]
        v = v[:,[2,1,0]].copy()
        f = f[:,[2,1,0]].copy()
    else:
        raise ValueError("data_name should be in ['fetal']")

    return v, f




 
 

class SegmentationPreprocessor:
    def __init__(self, data_name='fetal'):
        self.data_name = data_name

    def process_segmentation(self, seg):
        if self.data_name == 'fetal':
            seg = np.pad(seg, ((2, 2), (0, 0), (0, 0)), 'constant', constant_values=0)
            return seg[None].copy()
        else:
            raise ValueError("data_name should be in ['fetal']")

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


"""