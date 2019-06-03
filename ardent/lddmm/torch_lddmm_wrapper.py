# A wrapper around torch_lddmm.py to enable standard access to its registration capabilities.
# Both are to be replaced, respectively with the minimal core of torch_lddmm.py and similar wrappers, 
# or a totally fresh implementation.

import numpy as np
import torch
from .torch_lddmm_stable import LDDMM

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

    # Adjust template and target to match in shape.
    template, target = _pad_to_same_shape(template, target)

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
        'do_affine':1, # 0 = False
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
    lddmm = LDDMM(**arguments)
    lddmm.run()

    # Assemble outputs.

    # lddmm.computeThisDisplacement() returns phiinvAinvs - (X0, X1, X2).
    X012 = np.array(list(map(lambda t: t.cpu().numpy(), [lddmm.X0, lddmm.X1, lddmm.X2])))
    phiinvAinvs = np.array(lddmm.computeThisDisplacement()) + X012 # Elment-wise addition.

    # Instantiate lddmm.affineA to identity Tensor if do_affine == 0.
    if not hasattr(lddmm, 'affineA'):
        lddmm.affineA = torch.tensor(np.eye(4)).type(lddmm.params['dtype']).to(device=lddmm.params['cuda'])

    affineA = lddmm.affineA # Preserve lddmm.affineA.
    lddmm.affineA = torch.tensor(np.eye(4)).type(lddmm.params['dtype']).to(device=lddmm.params['cuda']) # Set lddmm.affineA to identity.
    phiinvs = lddmm.computeThisDisplacement() + X012 # Element-wise addition.
    # Reverse the order and take the negative of the vt012 tensor-lists.
    lddmm.vt0 = [-t for t in lddmm.vt0[::-1]]
    lddmm.vt1 = [-t for t in lddmm.vt1[::-1]]
    lddmm.vt2 = [-t for t in lddmm.vt2[::-1]]
    phi0, phi1, phi2 = lddmm.computeThisDisplacement() + X012 # Element-wise addition.
    # Restore the order and parity of the vt012 tensor-lists.
    lddmm.vt0 = [-t for t in lddmm.vt0[::-1]]
    lddmm.vt1 = [-t for t in lddmm.vt1[::-1]]
    lddmm.vt2 = [-t for t in lddmm.vt2[::-1]]
    lddmm.affineA = affineA # Restore lddmm.affineA.

    Aphi0 = lddmm.affineA.cpu().numpy()[0,0]*phi0 + lddmm.affineA.cpu().numpy()[0,1]*phi1 + lddmm.affineA.cpu().numpy()[0,2]*phi2 + lddmm.affineA.cpu().numpy()[0,3]
    Aphi1 = lddmm.affineA.cpu().numpy()[1,0]*phi0 + lddmm.affineA.cpu().numpy()[1,1]*phi1 + lddmm.affineA.cpu().numpy()[1,2]*phi2 + lddmm.affineA.cpu().numpy()[1,3]
    Aphi2 = lddmm.affineA.cpu().numpy()[2,0]*phi0 + lddmm.affineA.cpu().numpy()[2,1]*phi1 + lddmm.affineA.cpu().numpy()[2,2]*phi2 + lddmm.affineA.cpu().numpy()[2,3]

    # Not presently included: Ainv.
    return {
        'Aphis':np.array([Aphi0, Aphi1, Aphi2]), 
        'phis':np.array([phi0, phi1, phi2]), 
        'phiinvs':phiinvs, 
        'phiinvAinvs':phiinvAinvs, 
        'A':lddmm.affineA.cpu().numpy(), 
        
        'lddmm':lddmm}


def torch_apply_transform(image:np.ndarray, deform_to='template', Aphis=None, phiinvAinvs=None, lddmm=None):
    """Apply the transformation stored in Aphis (for deforming to the template) and phiinvAinvs (for deforming to the target).
    If deform_to='template', Aphis must be provided.
    If deform_to='target', phiinvAinvs must be provided."""

    # Presently must be given lddmm.
    if lddmm is None:
        raise RuntimeError("lddmm must be provided with present implementation.")

    dtype = 'torch.FloatTensor'

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

    return deformed_image.cpu().numpy()

def _pad_to_same_shape(template, target):
    """Return a tuple containing copies of template and target 
    that are padded to match the larger of the two shapes in each dimension."""

    # Break alias.
    template = np.copy(template)
    target = np.copy(target)

    # Verify template and target have the same number of dimensions.
    if template.ndim != target.ndim:
        raise ValueError((f"template and target must have the same number of dimensions.\n"
            f"template.ndim: {template.ndim}, target.ndim: {target.ndim}."))

    # Pad.
    for dim, (template_dim_length, target_dim_length) in enumerate(zip(template.shape, target.shape)):
        # Apply pad to the array with the lesser length along each dimension.

        if template_dim_length == target_dim_length:
            continue
        # template and target have different lengths along this dimension.
        
        # Apply half the padding on each side.
        # If the difference is odd, apply an extra layer on the far side of that dimension.
        dim_length_difference = abs(template_dim_length - target_dim_length)
        pad_size = dim_length_difference // 2
        odd_difference_correction = dim_length_difference % 2

        pad_shape = np.full([template.ndim, 2], 0)
        pad_shape[dim] = (pad_size, pad_size + odd_difference_correction) # Additional padding at end of dimension.

        if template_dim_length < target_dim_length:
            # template is shorter along this dimension and should be padded.
            template = np.pad(template, pad_width=pad_shape, mode='edge')
        else:
            # target is shorter along this dimension and should be padded.
            target = np.pad(target, pad_width=pad_shape, mode='edge')
    
    return template, target