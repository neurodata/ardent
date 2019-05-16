# A wrapper around torch_lddmm.py to enable standard access to its registration capabilities.
# Both are to be replaced, respectively with the minimal core of torch_lddmm.py and similar wrappers, 
# or a totally fresh implementation.

import numpy as np
import torch
from torch_lddmm import LDDMM

# This proto lddmm function should provide similar IO to the ultimately desired version.
def torch_register(template, target, sigmaR, eV, eL=0, eT=0, **kwargs):
    """Perform a registration between <template> and <target>.
    Supported kwargs [default value]:
    a [2]-> smoothing kernel, in # of voxels
    niter -> total iteration limit
    eT [0] -> translation step size
    eL [0] -> linear transformation step size
    eV -> deformative step size
    sigmaR -> deformation allowance
    do_affine [0]-> enable affine transformation (0 or 1)
    outdir -> ['.'] output directory path
   """

    # Set defaults.
    arguments = {
        'template':template, # atlas file name
        'target':target, # target file name
        'a':2, # smoothing kernel, scaled to pixel size
        'p':2,
        'niter':200,
        'epsilon':eV, # eV
        'epsilonL':eL, # not needed
        'epsilonT':eT, # not needed
        'minbeta':1e-15, # make close to 0
        'sigma':1.0, # sigmaM
        'sigmaR':sigmaR, # sigmaR
        'nt':5, # number of time steps in velocity field
        'do_affine':0, # 0 = False
        'checkaffinestep':0, # 0 = False
        'im_norm_ms':0, # 0 = False
        'gpu_number':0, # index of CUDA_VISIBLE_DEVICES to use, or None to use CPU
        'dtype':'float', # vs double
        'energy_fraction':0.000, # fraction of initial energy at which to stop, make small
        'cc':1, # contrast correction, 1=True
        'cc_channels':[], # image channels to run contrast correction, 0-indexed ([] means all channels)
        'costmask':None, # costmask file name
        'outdir':'.', # output directory name
        'optimizer':'gd', # gradient descent
   }
    # Update parameters with kwargs.
    arguments.update(kwargs)

    # Instantiate LDDMM object and run registration.
    lddmm = torch_lddmm.LDDMM(arguments)
    lddmm.run()

    # Assemble outputs.

    # lddmm.computeThisDisplacement() returns phiinvAinvs - (X0, X1, X2).
    X012 = np.array(list(map(lambda t: t.numpy(), [lddmm.X0, lddmm.X1, lddmm.X2])))
    phiinvAinvs = np.array(lddmm.computeThisDisplacement()) + X012 # Elment-wise addition.

    affineA = lddmm.affineA # Preserve lddmm.affineA.
    lddmm.affineA = torch.tensor(np.eye(4)).type(lddmm.params['dtype']).to(device=lddmm.params['cuda']) # Set lddmm.affineA to identity.
    phiinvs = lddmm.computeThisDisplacement() + X012 # Element-wise addition.
    # Reverse the order and take the negative of the vt012 tensor-lists.
    lddmm.vt0 = [-t for t in lddmm.vt0[::-1]]
    lddmm.vt1 = [-t for t in lddmm.vt1[::-1]]
    lddmm.vt2 = [-t for t in lddmm.vt2[::-1]]
    phi0, phi1, phi2 = lddmm.computeThisDisplacement() + X012 # Element-wise addition.
    lddmm.affineA = affineA # Restore lddmm.affineA.

    Aphi0 = lddmm.affineA[0,0]*phi0 + lddmm.affineA[0,1]*phi1 + lddmm.affineA[0,2]*phi2 + lddmm.affineA[0,3]
    Aphi1 = lddmm.affineA[1,0]*phi0 + lddmm.affineA[1,1]*phi1 + lddmm.affineA[1,2]*phi2 + lddmm.affineA[1,3]
    Aphi2 = lddmm.affineA[2,0]*phi0 + lddmm.affineA[2,1]*phi1 + lddmm.affineA[2,2]*phi2 + lddmm.affineA[2,3]

    # Not presently included: Ainv.
    return {
        'Aphis':np.array([Aphi0, Aphi1, Aphi2]), 
        'phis':np.array([phi0, phi1, phi2]), 
        'phiinvs':phiinvs, 
        'phiinvAinvs':phiinvAinvs, 
        'A':lddmm.affineA, 
        
        'lddmm':lddmm}


def torch_apply_transform(image:np.ndarray, deform_to='template', Aphis=None, phiinvAinvs=None, lddmm=None):
    """Apply the transformation stored in Aphis (for deforming to the template) and phiinvAinvs (for deforming to the target).
    If deform_to='template', Aphis must be provided.
    If deform_to='target', phiinvAinvs must be provided."""

    # Presently must be given lddmm.
    if lddmm is None:
        raise RuntimeError("lddmm must be provided with present implementation.")

    # Convert image from np.ndarray to torch.Tensor.
    image = torch.tensor(image).type(dtype).to(device=lddmm.params['cuda']) # Side-effect: breaks alias.

    # Set transformArray012 to Aphis for deforming to the template or to phiinvAinvs for deforming to the target.
    if deform_to == 'template':
        transformArray0, transformArray1, transformArray2 = [torch.tensor(Aphi).type(lddmm.params['dtype']).to(device=lddmm.params['cuda']) for Aphi in Aphis]
    elif deform_to == 'target':
        transformArray0, transformArray1, transformArray2 = [torch.tensor(phiinvAinv).type(lddmm.params['dtype']).to(device=lddmm.params['cuda']) for phiinvAinv in phiinvAinvs]
    else:
        raise ValueError(f"deform_to must be either 'template' or 'target'.\ndeform_to: {deform_to}.")

    # Deform image accordingly.
    # Convert sample points to [-1, 1], reshape image so first 2 dimensions are 1, perform grid_resample
    # line copied from torch_lddmm.py, last line before return of def applyThisTransformation().
    deformed_image = torch.squeeze(torch.nn.functional.grid_sample(image.unsqueeze(0).unsqueeze(0),torch.stack((transformArray2.type(dtype).to(device=lddmm.params['cuda'])/(lddmm.nx[2]*lddmm.dx[2]-lddmm.dx[2])*2-1,transformArray1.type(dtype).to(device=lddmm.params['cuda'])/(lddmm.nx[1]*lddmm.dx[1]-lddmm.dx[1])*2-1,transformArray0.type(dtype).to(device=lddmm.params['cuda'])/(lddmm.nx[0]*lddmm.dx[0]-lddmm.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border',mode='bilinear'))

    return deformed_image.numpy()
