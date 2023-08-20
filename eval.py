#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 21:09:46 2023

@author: salimrami
"""

"""
Created on Fri Jun 16 14:25:30 2023

@author: salimrami
"""
#save write mesh gifti
#pred surface not aligned
# warning axes surface surface inverse 
#le 5 /07 plan 15min de presentation 
#10 repetition 14h
# 10 slice almost

import os
import nibabel as nib
import trimesh
import numpy as np
from tqdm import tqdm
from scipy.ndimage import distance_transform_cdt as cdt
from skimage.measure import marching_cubes
from skimage.measure import label as compute_cc
from skimage.filters import gaussian

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

from data.preprocess import process_volume, process_surface, process_surface_inverse
from util.mesh import laplacian_smooth, compute_normal, compute_mesh_distance
from util.tca import topology
from model.net import CortexODE
from config import load_config
import time
from nibabel import gifti





# initialize topology correction
topo_correct = topology()


def seg2surf(seg,
             data_name='fetal',
             sigma=0.5,
             alpha=16,
             level=0.5,
             n_smooth=2):
    """
    Extract the surface based on the segmentation.
    
    seg: input segmentation
    sigma: standard deviation of guassian blurring
    alpha: threshold for obtaining boundary of topology correction
    level: extracted surface level for Marching Cubes
    n_smooth: iteration of Laplacian smoothing
    """
    
    # ------ connected components checking ------ 
    #cc, nc = compute_cc(seg, connectivity=2, return_num=True)
    cc, nc = compute_cc(seg, connectivity=2, return_num=True)

    #cc_id = 1 + np.argmax(np.array([np.count_nonzero(cc == i) for i in range(1, nc+1)]))
    
    

    cc_id = 1 + np.argmax(np.array([np.count_nonzero(cc == i)\
                                    for i in range(1, nc+1)]))

    #cc_id = 1 + np.argmax(np.array([np.count_nonzero(cc == i)\
                                    #for i in range(1, nc+1)]))
    seg = (cc==cc_id).astype(np.float64)

    # ------ generate signed distance function ------ 
    sdf = -cdt(seg) + cdt(1-seg)
    sdf = sdf.astype(float)
    sdf = gaussian(sdf, sigma=sigma)

     # ------ topology correction ------
    sdf_topo= topo_correct.apply(sdf, threshold=alpha)

    # ------ marching cubes ------
    v_mc, f_mc, _, _ = marching_cubes(-sdf_topo, level=-level, method='lorensen')
    v_mc = v_mc[:,[2,1,0]].copy()
    #v_mc = v_mc[:,[0,1,2]].copy()

    f_mc = f_mc.copy()
    D1,D2,D3 = sdf_topo.shape
    #print(D1,D2,D3)
    D = max(D1,D2,D3)
    #jai decommenté ca pour la normalisation des surface pial et white
    #v_mc = (2*v_mc - [D3, D2, D1]) / D   # rescale to [-1,1]
    #v_mc = (2*v_mc - [D1, D2, D3]) / D   # rescale to [-1,1]
    #print("v_mc apres normalisation",v_mc)

    #inverser ca !
    #sauvegarder freesurfer
    
    
    # ------ bias correction ------
    # Note that this bias is introduced by FreeSurfer.
    # FreeSurfer changed the size of the input MRI, 
    # but the affine matrix of the MRI was not changed.
    # So this bias is caused by the different between 
    # the original and new affine matrix.
    
        
    # ------ mesh smoothing ------
    v_mc = torch.Tensor(v_mc).unsqueeze(0).to(device)
    f_mc = torch.LongTensor(f_mc).unsqueeze(0).to(device)
    for j in range(n_smooth):    # smooth and inflate the mesh
        v_mc = laplacian_smooth(v_mc, f_mc, 'uniform', lambd=1)
    v_mc = v_mc[0].cpu().numpy()
    #v_mc = v_mc[:,[0,1,2]].copy()
    f_mc = f_mc[0].cpu().numpy()
    
    return v_mc, f_mc


if __name__ == '__main__':
    
    # ------ load configuration ------
    config = load_config()
    test_type = config.test_type  # initial surface / prediction / evaluation
    data_dir = config.data_dir  # directory of datasets
    model_dir = config.model_dir  # directory of pretrained models
    init_dir = config.init_dir  # directory for saving the initial surfaces
    result_dir = config.result_dir  # directory for saving the predicted surfaces
    data_name = config.data_name  # hcp, adni, dhcp
    surf_hemi = config.surf_hemi  # lh, rh
    device = config.device
    tag = config.tag  # identity of the experiment

    C = config.dim_h     # hidden dimension of features
    K = config.kernel_size    # kernel / cube size
    Q = config.n_scale    # multi-scale input
    
    step_size = config.step_size    # step size of integration
    solver = config.solver    # ODE solver
    n_inflate = config.n_inflate  # inflation iterations
    rho = config.rho # inflation scale

    # ------ load models ------
    
    seg_file = "/scratch/saiterrami/seg/lh_seg.nii.gz"
    seg_file_rh = "/scratch/saiterrami/seg/lh_segmentation1.nii.gz"
    seg_data_rh = nib.load(seg_file_rh).get_fdata()


    #print(seg_file)

# Load the segmentation data
    seg_data = nib.load(seg_file).get_fdata()
    

    
    
    
    print("taille de la seg",seg_data.shape)
    

    if test_type == 'pred' or test_type == 'eval':
        T = torch.Tensor([0,1]).to(device)
        cortexode_wm = CortexODE(dim_in=3, dim_h=C, kernel_size=K, n_scale=Q).to(device)
        cortexode_gm = CortexODE(dim_in=3, dim_h=C, kernel_size=K, n_scale=Q).to(device)
        cortexode_wm.load_state_dict(torch.load(model_dir+'model_wm_'+data_name+'_'+surf_hemi+'_'+tag+'.pt', map_location=device))
        cortexode_gm.load_state_dict(torch.load(model_dir+'model_gm_'+data_name+'_'+surf_hemi+'_'+tag+'.pt', map_location=device))
        cortexode_wm.eval()
        cortexode_gm.eval()


    # ------ start testing ------
    subject_list = sorted(os.listdir(data_dir))

    if test_type == 'eval':
        assd_wm_all = []
        assd_gm_all = []
        hd_wm_all = []
        hd_gm_all = []
        sif_wm_all = []
        sif_gm_all = []

    for i in tqdm(range(len(subject_list))):
        subid = subject_list[i]

        ## ------- load brain MRI ------- 
        
            
            
            
            
        if data_name == 'fetal':
            brain = nib.load(data_dir+subid+'/'+subid+'_T2w.nii.gz')
            brain_arr = brain.get_fdata()
            brain_arr = (brain_arr / 1200).astype(np.float16)
           # brain_arr = brain_arr[2:-2, :, :]  # Remove padding
        brain_arr = process_volume(brain_arr, data_name)
        volume_in = torch.Tensor(brain_arr).unsqueeze(0).to(device)
        
            

        # ------- predict segmentation ------- 
            


            

        with torch.no_grad():
            
            seg_pred = seg_data_rh #seg_data
            #seg_pred = np.transpose(seg_data, (2, 1, 0))  # Permute les dimensions selon l'ordre (1, 2, 0)
            seg_pred = process_volume(seg_pred, data_name='fetal')  # Prétraitement de seg_pred
            seg_pred = seg_pred.squeeze(0)  # Supprime la dimension du batch
            #seg_pred = np.transpose(seg_pred, (2, 0, 1))  # Ajuste les dimensions
            #seg_pred = seg_pred[2:-2, :, :]  # Remove padding
            segnet = torch.from_numpy((seg_pred)).to(device)
            seg_pred = segnet  # Assuming the segmentation data is stored in the first channel
            seg_pred = torch.squeeze(seg_pred, dim=0)
            print("seg_pred shape:", seg_pred.shape)
            
            
            
            
            
           
            
            counter = 1  # Initialize the counter
            if surf_hemi == 'lh':
                seg = (seg_pred==1).cpu().numpy()  # lh
                
                
                print("seg shape:", seg.shape)
                
                    
                #seg_img = nib.Nifti1Image(seg.astype(np.uint8), brain.affine)
        
        # Generate the file name with counter
               # file_name = f'lh_segmentation{counter}.nii.gz'
        
                #nib.save(seg_img, file_name)  # Save predicted segmentation
        
                #counter += 1  # Increment the counter for the next segmentation
               
                
            
                
            elif surf_hemi == 'rh':
                
                seg = (seg_data_rh==2).cpu().numpy()  # rh
                #seg = seg[2:-2, :, :]  # Remove padding
              #  seg_img = nib.Nifti1Image(seg.astype(np.uint8), brain.affine)
               # print(seg_img.shape)

                #nib.save(seg_img, 'rh_segmentation.nii.gz') #save predicted segmentation

      


        """      with torch.no_grad():
                    seg_out = segnet(volume_in)
                    seg_pred = torch.argmax(seg_out, dim=1)[0]
                    if surf_hemi == 'lh':
                        seg = (seg_pred == 1).cpu().numpy()  # lh
                        seg = seg[2:-2, :, :]  # Remove padding
                        seg_img = nib.Nifti1Image(seg.astype(np.uint8), np.eye(4))
                        print(seg_img.shape)

                        ##nib.save(seg_img, 'lh_segmentation.nii.gz')#save predicted segmentation
                        # Generate a unique file name using timestamp
        """   
        
                
                

        # ------- extract initial surface ------- 
        v_in, f_in = seg2surf(seg, data_name, sigma=0.5,
                              alpha=16, level=0.5, n_smooth=2)
        v_in = v_in[:,[2,1,0]]
        mesh_init = trimesh.Trimesh(v_in, f_in)
        
        v_in, f_in = process_surface(v_in, f_in, data_name)
        v_in, f_in = process_surface_inverse(v_in, f_in, data_name)
        mesh_init = trimesh.Trimesh(v_in, f_in)
        mesh_init.export('/scratch/saiterrami/init/init.obj')
        nib.freesurfer.io.write_geometry(result_dir+data_name+'init''_''.white',
                                         v_in, f_in)
        gii = nib.gifti.GiftiImage()
        gii.add_gifti_data_array(nib.gifti.GiftiDataArray(v_in, intent='NIFTI_INTENT_POINTSET'))
        gii.add_gifti_data_array(nib.gifti.GiftiDataArray(f_in, intent='NIFTI_INTENT_TRIANGLE'))
        nib.save(gii, result_dir + data_name + 'init' + '_init.gii')

        # ------- save initial surface ------- 
        if test_type == 'init':
            mesh_init = trimesh.Trimesh(v_in, f_in)
            #v_in, f_in = process_surface_inverse(v_in, f_in, data_name)
            mesh_init.export(init_dir+'init_'+data_name+'_'+surf_hemi+'_'+subid+'.obj')
            #v_in, f_in = process_surface(v_in, f_in, data_name)
            
            #mesh_in = trimesh.Trimesh(v_in, f_in)
            #mesh_in.export(result_dir+'in_'+data_name+'_'+surf_hemi+'_'+subid+'.obj')

        # ------- predict cortical surfaces ------- 
        if test_type == 'pred' or test_type == 'eval':
            with torch.no_grad():
                v_in = torch.Tensor(v_in).unsqueeze(0).to(device)
                print("v_in",v_in.shape)
                f_in = torch.LongTensor(f_in).unsqueeze(0).to(device)
                
                # wm surface
                cortexode_wm.set_data(v_in, volume_in)
                v_wm_pred = odeint(cortexode_wm, v_in, t=T, method=solver,
                                   options=dict(step_size=step_size))[-1]
                v_gm_in = v_wm_pred.clone()
                

                # inflate and smooth
                
                for i in range(2):
                    v_gm_in = laplacian_smooth(v_gm_in, f_in, lambd=1.0)
                    n_in = compute_normal(v_gm_in, f_in)
                    v_gm_in += 0.002 * n_in

                # pial surface
                cortexode_gm.set_data(v_gm_in, volume_in)
                v_gm_pred = odeint(cortexode_gm, v_gm_in, t=T, method=solver,
                                   options=dict(step_size=step_size/2))[-1]  # divided by 2 to reduce SIFs
                #v_gm_pred = v_gm_pred[:,[2,1,0]]

            v_wm_pred = v_wm_pred[0].cpu().numpy()
            f_wm_pred = f_in[0].cpu().numpy()
            v_gm_pred = v_gm_pred[0].cpu().numpy()
            f_gm_pred = f_in[0].cpu().numpy()
            # map the surface coordinate from [-1,1] to its original space
            
            #ces lignes en dessous je l'ai ajouté pour corriger le probleme d'axes
            v_wm_pred = v_wm_pred[:,[0,1,2]]
            
            v_gm_pred = v_gm_pred[:,[0,1,2]]
            #v_wm_pred, f_wm_pred = process_surface_inverse(v_wm_pred, f_wm_pred, data_name)
            #v_gm_pred, f_gm_pred = process_surface_inverse(v_gm_pred, f_gm_pred, data_name)
           


        # ------- save predictde surfaces ------- 
        if test_type == 'pred':
            ### save mesh to .obj or .stl format by Trimesh
            #mesh_wm = trimesh.Trimesh(v_wm_pred, f_wm_pred)
            #mesh_gm = trimesh.Trimesh(v_gm_pred, f_gm_pred)
            #mesh_wm.export(result_dir+'wm_'+data_name+'_'+surf_hemi+'_'+subid+'.stl')
            #mesh_gm.export(result_dir+'gm_'+data_name+'_'+surf_hemi+'_'+subid+'.obj')
            #mesh_wm.export(result_dir+'wm_'+data_name+'_'+surf_hemi+'_'+subid+'.obj')

            # save the surfaces in FreeSurfer format
            nib.freesurfer.io.write_geometry(result_dir+data_name+'_'+surf_hemi+'_'+subid+'.white',
                                             v_wm_pred, f_wm_pred)
            nib.freesurfer.io.write_geometry(result_dir+data_name+'_'+surf_hemi+'_'+subid+'.pial',
                                             v_gm_pred, f_gm_pred)
            white_gii = gifti.GiftiImage()
            white_data = gifti.GiftiDataArray(v_wm_pred, intent='NIFTI_INTENT_POINTSET')
            white_faces = gifti.GiftiDataArray(f_wm_pred, intent='NIFTI_INTENT_TRIANGLE')
            white_gii.add_gifti_data_array(white_data)
            white_gii.add_gifti_data_array(white_faces)

            # Save the white geometry to a GIfTI file
            white_file = result_dir + data_name + '_' + surf_hemi + '_' + subid + '.white.gii'
            nib.save(white_gii, white_file)
            # Create a GiftiImage object for the pial geometry
            pial_gii = gifti.GiftiImage()
            pial_data = gifti.GiftiDataArray(v_gm_pred, intent='NIFTI_INTENT_POINTSET')
            pial_faces = gifti.GiftiDataArray(f_gm_pred, intent='NIFTI_INTENT_TRIANGLE')
            pial_gii.add_gifti_data_array(pial_data)
            pial_gii.add_gifti_data_array(pial_faces)

    # Save the pial geometry to a GIfTI file
            pial_file = result_dir + data_name + '_' + surf_hemi + '_' + subid + '.pial.gii'
            nib.save(pial_gii, pial_file)

            
        # ------- load ground truth surfaces ------- 
        if test_type == 'eval':
            
            if data_name == 'fetal':
                if surf_hemi == 'lh':
                    surf_wm_gt = nib.load(data_dir+subid+'/'+subid+'_left_wm.surf.gii')
                    surf_gm_gt = nib.load(data_dir+subid+'/'+subid+'_left_pial.surf.gii')
                    v_wm_gt, f_wm_gt = surf_wm_gt.agg_data('pointset'), surf_wm_gt.agg_data('triangle')
                    v_gm_gt, f_gm_gt = surf_gm_gt.agg_data('pointset'), surf_gm_gt.agg_data('triangle')
                elif surf_hemi == 'rh':
                    surf_wm_gt = nib.load(data_dir+subid+'/'+subid+'_right_wm.surf.gii')
                    surf_gm_gt = nib.load(data_dir+subid+'/'+subid+'_right_pial.surf.gii')
                    v_wm_gt, f_wm_gt = surf_wm_gt.agg_data('pointset'), surf_wm_gt.agg_data('triangle')
                    v_gm_gt, f_gm_gt = surf_gm_gt.agg_data('pointset'), surf_gm_gt.agg_data('triangle')

                # apply the affine transformation provided by brain MRI nifti
                v_tmp = np.ones([v_wm_gt.shape[0],4])
                v_tmp[:,:3] = v_wm_gt
                v_wm_gt = v_tmp.dot(np.linalg.inv(brain.affine).T)[:,:3]
                v_tmp = np.ones([v_gm_gt.shape[0],4])
                v_tmp[:,:3] = v_gm_gt
                v_gm_gt = v_tmp.dot(np.linalg.inv(brain.affine).T)[:,:3]

        # ------- evaluation -------
        if test_type == 'eval':
            v_wm_pred = torch.Tensor(v_wm_pred).unsqueeze(0).to(device)
            f_wm_pred = torch.LongTensor(f_wm_pred).unsqueeze(0).to(device)
            v_gm_pred = torch.Tensor(v_gm_pred).unsqueeze(0).to(device)
            f_gm_pred = torch.LongTensor(f_gm_pred).unsqueeze(0).to(device)
#j'ai changéles v et f de la GT pour mesurer les distances HD ASSD

            v_wm_gt = torch.Tensor(v_wm_gt).unsqueeze(0).to(device)
            f_wm_gt = torch.LongTensor(f_wm_gt.astype(np.float32)).unsqueeze(0).to(device)
            v_gm_gt = torch.Tensor(v_gm_gt).unsqueeze(0).to(device)
            f_gm_gt = torch.LongTensor(f_gm_gt.astype(np.float32)).unsqueeze(0).to(device)




            # compute ASSD and HD
            assd_wm, hd_wm = compute_mesh_distance(v_wm_pred, v_wm_gt, f_wm_pred, f_wm_gt)
            assd_gm, hd_gm = compute_mesh_distance(v_gm_pred, v_gm_gt, f_gm_pred, f_gm_gt)
            if data_name == 'fetal':  # the resolution is 0.7
                assd_wm = 0.5*assd_wm
                assd_gm = 0.5*assd_gm
                hd_wm = 0.5*hd_wm
                hd_gm = 0.5*hd_gm
            assd_wm_all.append(assd_wm)
            assd_gm_all.append(assd_gm)
            hd_wm_all.append(hd_wm)
            hd_gm_all.append(hd_gm)

            ### compute percentage of self-intersecting faces
            ### uncomment below if you have installed torch-mesh-isect
            ### https://github.com/vchoutas/torch-mesh-isect
            # sif_wm_all.append(check_self_intersect(v_wm_pred, f_wm_pred, collisions=20))
            # sif_gm_all.append(check_self_intersect(v_gm_pred, f_gm_pred, collisions=20))
            sif_wm_all.append(0)
            sif_gm_all.append(0)



# ------- report the final results ------- 
    if test_type == 'eval':
        print('======== wm ========')
        print('assd mean:', np.mean(assd_wm_all))
        print('assd std:', np.std(assd_wm_all))
        print('hd mean:', np.mean(hd_wm_all))
        print('hd std:', np.std(hd_wm_all))
        print('sif mean:', np.mean(sif_wm_all))
        print('sif std:', np.std(sif_wm_all))
        print('======== gm ========')
        print('assd mean:', np.mean(assd_gm_all))
        print('assd std:', np.std(assd_gm_all))
        print('hd mean:', np.mean(hd_gm_all))
        print('hd std:', np.std(hd_gm_all))
        print('sif mean:', np.mean(sif_gm_all))
        print('sif std:', np.std(sif_gm_all))
        
        
        
        
       
        
       




import torch
import numpy as np
import nibabel as nib
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss.point_mesh_distance import _PointFaceDistance

def load_gifti_data(mesh_path):
    """ Load mesh data from a GIFTI file and return vertices and faces. """
    mesh = nib.load(mesh_path)
    vertices = torch.tensor(mesh.darrays[0].data, dtype=torch.float32)
    faces = torch.tensor(mesh.darrays[1].data, dtype=torch.int64)
    return vertices, faces

def compute_surface_distance(mesh_pred_path, mesh_gt_path, n_pts=100000):
    """ Compute average symmetric surface distance (ASSD) and Hausdorff distance (HD) 
    between two meshes given their file paths. """
    
    v_sur1, f_sur1 = load_gifti_data(mesh_pred_path)
    v_sur2, f_sur2 = load_gifti_data(mesh_gt_path)
    
    mesh_pred = Meshes(verts=[v_sur1], faces=[f_sur1])
    mesh_gt = Meshes(verts=[v_sur2], faces=[f_sur2])
    
    # Sample points from meshes
    pts_pred = sample_points_from_meshes(mesh_pred, num_samples=n_pts)
    pts_gt = sample_points_from_meshes(mesh_gt, num_samples=n_pts)
    pcl_pred = Pointclouds(pts_pred)
    pcl_gt = Pointclouds(pts_gt)

    # Compute point-to-mesh distances using _PointFaceDistance
    point_face_distance = _PointFaceDistance()
    x_dist = point_face_distance(pcl_pred, mesh_gt)
    y_dist = point_face_distance(pcl_gt, mesh_pred)

    # Calculate ASSD and HD
    assd = (x_dist.mean().item() + y_dist.mean().item()) / 2

    x_quantile = torch.quantile(x_dist, 0.9).item()
    y_quantile = torch.quantile(y_dist, 0.9).item()
    hd = max(x_quantile, y_quantile)
    
    return assd, hd

# Chemins des fichiers GIFTI pour les deux surfaces
mesh_pred_path = "/scratch/saiterrami/results/fetalinit_init.gii"
mesh_gt_path = "/scratch/saiterrami/results/fetal_lh_fetus_data.white.gii"

# Calcul des distances
assd, hd = compute_surface_distance(mesh_pred_path, mesh_gt_path)

print('ASSD:', assd)
print('HD:', hd)

       
        
       
        
       
        
       
        