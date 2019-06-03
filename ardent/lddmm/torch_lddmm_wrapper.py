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
        'A':lddmm.affineA, 
        
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

################################################################################
# for Daniel's demo, let's use this slightly different version

class Transformer:
    def __init__(self,I,J,
                 xI=None,xJ=None,
                 nt=5,a=2.0,p=2.0,
                 sigmaM=1.0,sigmaR=1.0,
                 order=2,
                 sigmaA=None,
                 **kwargs):
        '''
        Specify polynomial intensity mapping order with order parameters
        2 corresponds to linear, nothing less than 2 is supported
        input sigmaA for weights
        
        assume input images are and gridpoints are torch tensors already
        '''
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        self.dtype = torch.float64
        
        self.I = I
        if xI is None:
            xI = [torch.arange(I.shape[i],dtype=self.dtype,device=self.device) for i in range(3)]
            xI = [x - torch.mean(x) for x in xI]
        self.xI = xI
        self.nxI = I.shape
        self.dxI = torch.tensor([xI[0][1]-xI[0][0], xI[1][1]-xI[1][0], xI[2][1]-xI[2][0]],
                                dtype=self.dtype,device=self.device)
        self.XI = torch.stack(torch.meshgrid(xI))
        
        self.J = J
        if xJ is None:
            xJ = [torch.arange(J.shape[i],dtype=self.dtype,device=self.device) for i in range(3)]
            xJ = [x - torch.mean(x) for x in xJ]
        self.xJ = xJ
        self.nxJ = J.shape
        self.dxJ = torch.tensor([xJ[0][1]-xJ[0][0], xJ[1][1]-xJ[1][0], xJ[2][1]-xJ[2][0]],
                                dtype=self.dtype,device=self.device)
        self.XJ = torch.stack(torch.meshgrid(xJ))
        
        # a weight, may be updated via EM
        self.WM = torch.ones(self.nxJ,dtype=self.dtype,device=self.device)
        if sigmaA is not None:
            self.WM *= 0.9
            self.WA = torch.ones(self.nxJ,dtype=self.dtype,device=self.device)*0.1
            self.CA = torch.max(J) # constant value for artifact
        
        self.nt = nt
        self.dt = 1.0/nt
        
        self.sigmaM = sigmaM
        self.sigmaR = sigmaR
        self.sigmaA = sigmaA
        
        self.order = order
        
        self.EMsave = []
        self.ERsave = []
        self.Esave = []
        
        usegrad = False # typically way too much memory
        self.v = torch.zeros((self.nt,3,self.nxI[0],self.nxI[1],self.nxI[2]),
                             dtype=self.dtype,device=self.device, requires_grad=usegrad)
        self.vhat = torch.rfft(self.v,3,onesided=False)
        self.A = torch.eye(4,dtype=self.dtype,device=self.device, requires_grad=usegrad)
        
        # smoothing
        f0I = torch.arange(self.nxI[0],dtype=self.dtype,device=self.device)/self.dxI[0]/self.nxI[0]
        f1I = torch.arange(self.nxI[1],dtype=self.dtype,device=self.device)/self.dxI[1]/self.nxI[1]
        f2I = torch.arange(self.nxI[2],dtype=self.dtype,device=self.device)/self.dxI[2]/self.nxI[2]
        F0I,F1I,F2I = torch.meshgrid(f0I, f1I, f2I)
        self.a = a
        self.p = p
        Lhat = (1.0 - self.a**2*( (-2.0 + 2.0*torch.cos(2.0*np.pi*self.dxI[0]*F0I))/self.dxI[0]**2 
                + (-2.0 + 2.0*torch.cos(2.0*np.pi*self.dxI[1]*F1I))/self.dxI[1]**2
                + (-2.0 + 2.0*torch.cos(2.0*np.pi*self.dxI[2]*F2I))/self.dxI[2]**2 ) )**self.p
        self.Lhat = Lhat
        self.LLhat = self.Lhat**2
        self.Khat = 1.0/self.LLhat
        
    def forward(self):        
        ################################################################################
        # flow forwards
        self.phii = self.XI.clone().detach() # recommended way to copy construct from  a tensor
        self.It = torch.zeros((self.nt,self.nxI[0],self.nxI[1],self.nxI[2]),dtype=self.dtype,device=self.device)
        self.It[0] = self.I
        for t in range(self.nt):
            # apply the tform to I0    
            if t > 0: self.It[t] = self.interp3(self.xI,self.I,self.phii)
            Xs = self.XI - self.dt*self.v[t]
            self.phii = self.interp3(self.xI,self.phii-self.XI,Xs) + Xs
        # apply deformation including affine
        self.Ai = torch.inverse(self.A)
        X0s = self.Ai[0,0]*self.XJ[0] + self.Ai[0,1]*self.XJ[1] + self.Ai[0,2]*self.XJ[2] + self.Ai[0,3]
        X1s = self.Ai[1,0]*self.XJ[0] + self.Ai[1,1]*self.XJ[1] + self.Ai[1,2]*self.XJ[2] + self.Ai[1,3]
        X2s = self.Ai[2,0]*self.XJ[0] + self.Ai[2,1]*self.XJ[1] + self.Ai[2,2]*self.XJ[2] + self.Ai[2,3]
        self.AiX = torch.stack([X0s,X1s,X2s])
        self.phiiAi = self.interp3(self.xI,self.phii-self.XI,self.AiX) + self.AiX
        self.AphiI = self.interp3(self.xI,self.I,self.phiiAi)
        ################################################################################
        # calculate and apply intensity transform        
        AphiIflat = torch.flatten(self.AphiI)
        Jflat = torch.flatten(self.J)
        WMflat = torch.flatten(self.WM)
        # format data into a Nxorder matrix
        B = torch.zeros((self.AphiI.numel(),self.order), device=self.device, dtype=self.dtype) # B for basis functions
        for o in range(self.order):
            B[:,o] = AphiIflat**o
        BT = torch.transpose(B,0,1)
        BTB = torch.matmul( BT*WMflat, B)        
        BTJ = torch.matmul( BT*WMflat, Jflat )              
        self.coeffs,_ = torch.solve(BTJ[:,None],BTB) 
        self.CA = torch.mean(self.J*(1.0-self.WM))
        # torch.solve(B,A) solves AX=B (note order is opposite what I'd expect)        
        self.fAphiI = torch.matmul(B,self.coeffs).reshape(self.nxJ)
        # for convenience set this error to a member
        self.err = self.fAphiI - self.J
        
    def weights(self):
        '''Calculate image matching and artifact weights in simple Gaussian mixture model'''
        fM = torch.exp( (self.fAphiI - self.J)**2*(-1.0/2.0/self.sigmaM**2) ) / np.sqrt(2.0*np.pi*self.sigmaM**2)
        fA = torch.exp( (self.CA     - self.J)**2*(-1.0/2.0/self.sigmaA**2) ) / np.sqrt(2.0*np.pi*self.sigmaA**2)
        fsum = fM + fA
        self.WM = fM/fsum        
        
    def cost(self):                
        # get matching cost
        EM = torch.sum((self.fAphiI - self.J)**2*self.WM)/2.0/self.sigmaM**2*torch.prod(self.dxJ)
        # note the complex number is just an extra dimension at the end
        # note divide by numel(I) to conserve power when summing in fourier domain
        ER = torch.sum(torch.sum(torch.sum(torch.abs(self.vhat)**2,dim=(-1,1,0))*self.LLhat))\
            *(self.dt*torch.prod(self.dxI)/2.0/self.sigmaR**2/torch.numel(self.I))        
        E = ER + EM     
        # append these outputs for plotting
        self.EMsave.append(EM.cpu().numpy())
        self.ERsave.append(ER.cpu().numpy())
        self.Esave.append(E.cpu().numpy())
        
    def step_v(self, eV=0.0):
        ''' One step of gradient descent for velocity field v'''
        # get error
        err = (self.fAphiI - self.J)*self.WM
        # propagate error through poly
        Df = torch.zeros(self.nxJ, device=self.device, dtype=self.dtype)            
        for o in range(1,self.order):
            Df +=  o * self.AphiI**(o-1)
        errDf = err * Df
        # deform back through flow
        self.phi = self.XI.clone().detach() # torch recommended way to make a copy
        for t in range(self.nt-1,-1,-1):
            Xs = self.XI + self.dt*self.v[t]
            self.phi = self.interp3(self.xI,self.phi-self.XI,Xs) + Xs
            Aphi0 = self.A[0,0]*self.phi[0] + self.A[0,1]*self.phi[1] + self.A[0,2]*self.phi[2] + self.A[0,3]
            Aphi1 = self.A[1,0]*self.phi[0] + self.A[1,1]*self.phi[1] + self.A[1,2]*self.phi[2] + self.A[1,3]
            Aphi2 = self.A[2,0]*self.phi[0] + self.A[2,1]*self.phi[1] + self.A[2,2]*self.phi[2] + self.A[2,3]
            self.Aphi = torch.stack([Aphi0,Aphi1,Aphi2])
            # gradient
            Dphi = self.gradient(self.phi,self.dxI)
            detDphi = Dphi[0][0]*(Dphi[1][1]*Dphi[2][2]-Dphi[1][2]*Dphi[2][1]) \
                - Dphi[0][1]*(Dphi[1][0]*Dphi[2][2] - Dphi[1][2]*Dphi[2][0]) \
                + Dphi[0][2]*(Dphi[1][0]*Dphi[2][1] - Dphi[1][1]*Dphi[2][0])
            # pull back error
            errDft = self.interp3(self.xJ,errDf,self.Aphi)
            # gradient of image
            DI = self.gradient(self.It[t],self.dxI)
            # the gradient, error, times, determinant, times image grad
            grad = (errDft*detDphi)[None]*DI*(-1.0/self.sigmaM**2)*torch.det(self.A)
            # smooth it (add extra dimension for complex)
            gradhats = torch.rfft(grad,3,onesided=False)*self.Khat[...,None]
            # add reg
            gradhats = gradhats + self.vhat[t]/self.sigmaR**2
            # get final gradient
            grad = torch.irfft(gradhats,3,onesided=False)
            # update
            self.v[t] -= grad*eV
        # fourier transform for later
        self.vhat = torch.rfft(self.v,3,onesided=False)
           
    def step_A(self,eL=0.0,eT=0.0):        
        # get error
        err = (self.fAphiI - self.J)*self.WM
        # energy gradient with respect to affine transform
        DfAphiI = self.gradient(self.AphiI,dx=self.dxJ)
        DfAphiI0 = torch.cat((DfAphiI,torch.zeros(self.nxJ,dtype=self.dtype,device=self.device)[None]))
        # gradient should go down a row, X across a column
        AiXo = torch.cat((self.AiX,torch.ones(self.nxJ,dtype=self.dtype,device=self.device)[None]))
        gradA = torch.sum(DfAphiI0[:,None,...]*AiXo[None,:,...]*err[None,None],(-1,-2,-3))\
            *(-1.0/self.sigmaM**2*torch.prod(self.dxI))            
        gradA = torch.matmul(torch.matmul(self.Ai.t(),gradA),self.Ai.t())
        
        # update A
        EL = torch.tensor([[1,1,1,0],[1,1,1,0],[1,1,1,0],[0,0,0,0]],dtype=self.dtype,device=self.device)
        ET = torch.tensor([[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,0]],dtype=self.dtype,device=self.device)
        e = EL*eL + ET*eT            
        stepA = e*gradA       
        self.A = self.A - stepA
        self.Ai = torch.inverse(self.A)
        
    # to interpolate, use this
    # https://pytorch.org/docs/0.3.0/nn.html#torch.nn.functional.grid_sample
    def interp3(self,x,I,phii):
        '''Interpolate image I,
        sampled at points x (1d array), 
        at the new points phii (dense grid)     

        Note that
        grid[n, d, h, w] specifies the x, y, z pixel locations 
        for interpolating output[n, :, d, h, w]
        '''
        # unpack arguments
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]
        phii0 = phii[0]
        phii1 = phii[1]
        phii2 = phii[2]
        # the input image needs to be reshaped so the first two dimensions are 1
        if I.dim() == 3:
            # for grayscale images
            Ireshape = I[None,None,...]
        elif I.dim() == 4:
            # for vector fields, multi channel images, etc.
            Ireshape = I[None,...]
        else:
            raise ValueError('Tensor to interpolate must be dim 3 or 4')
        # the coordinates need to be rescaled from -1 to 1    
        grid0 = (phii0 - x0[0])/(x0[-1] - x0[0])*2.0 - 1.0
        grid1 = (phii1 - x1[0])/(x1[-1] - x1[0])*2.0 - 1.0
        grid2 = (phii2 - x2[0])/(x2[-1] - x2[0])*2.0 - 1.0
        grid = torch.stack([grid2,grid1,grid0], dim=-1)
        # and the grid also needs to be reshaped to have a 1 as the first index
        grid = grid[None]
        # do the resampling
        out = torch.nn.functional.grid_sample(Ireshape, grid, padding_mode='border')
        # squeeze out the first dimensions
        if I.dim()==3:
            out = out[0,0,...]
        elif I.dim() == 4:
            out = out[0,...]
        # return the output
        return out
    
    # now we need gradient
    def gradient(self,I,dx=[1,1,1]):
        ''' Gradient of an image in each direction
        We want centered difference in the middle, 
        and forward or backward difference at the ends
        image I can have as many leading dimensions as you want, 
        gradient will apply to the last three
        '''
        I_0_list = [ (I[...,1,:,:]-I[...,0,:,:])[...,None,:,:]/dx[0],
                (I[...,2:,:,:]-I[...,:-2,:,:])/(2.0*dx[0]),
                (I[...,-1,:,:]-I[...,-2,:,:])[...,None,:,:]/dx[0] ]
        I_0 = torch.cat(I_0_list, dim=-3)

        I_1_list = [ (I[...,:,1,:]-I[...,:,0,:])[...,:,None,:]/dx[1],
            (I[...,:,2:,:]-I[...,:,:-2,:])/(2.0*dx[1]),
            (I[...,:,-1,:]-I[...,:,-2,:])[...,:,None,:]/dx[1] ]
        I_1 = torch.cat(I_1_list, dim=-2)

        I_2_list = [ (I[...,:,:,1]-I[...,:,:,0])[...,:,:,None]/dx[2],
            (I[...,:,:,2:]-I[...,:,:,:-2])/(2.0*dx[2]),
            (I[...,:,:,-1]-I[...,:,:,-2])[...,:,:,None]/dx[2] ]
        I_2 = torch.cat(I_2_list, dim=-1)
        # note we insert the new dimension at position -4!
        return torch.stack([I_0,I_1,I_2],-4)
    
    def show_image(self,I,x=None,n=None,fig=None,clim=None):
        if n is None:
            n = 6
        if x is None:        
            x = [np.arange(n) - np.mean(np.arange(n)) for n in I.shape]
        slices = np.linspace(0,I.shape[2],n+2)
        slices = np.round(slices[1:-1]).astype(int)
        if fig is None:
            fig,ax = plt.subplots(1,n)
        else:
            fig.clf()
            ax = fig.subplots(1,n)

        if clim is None:
            clim = [np.min(I),np.max(I)]
            m = np.median(I)
            sad = np.mean(np.abs(I - m))
            nsad = 4.0
            clim = [m-sad*nsad, m+sad*nsad]

        for s in range(n):        
            ax[s].imshow(I[:,:,slices[s]], 
                         extent=(x[1][0],x[1][-1],x[0][0],x[0][-1]), 
                         origin='lower',vmin=clim[0],vmax=clim[1],cmap='gray')
            if s > 0:
                ax[s].set_yticklabels([])
            ax[s].set_title(f'z={slices[s]}')
        return fig,ax


def torch_apply_transform_daniel(image:np.ndarray, deform_to='template', Aphis=None, phiinvAinvs=None, lddmm=None):
    """daniel's version for demo to be replaced
    Apply the transformation stored in Aphis (for deforming to the template) and phiinvAinvs (for deforming to the target).
    If deform_to='template', Aphis must be provided.
    If deform_to='target', phiinvAinvs must be provided."""
    # Presently must be given lddmm.
    if lddmm is None:
        raise RuntimeError("lddmm must be provided with present implementation.")
        
    if deform_to == 'template':
        out = lddmm.interp3(lddmm.xJ,torch.tensor(image,dtype=lddmm.dtype,device=lddmm.device),lddmm.Aphi)
    elif deform_to == 'target':
        out = lddmm.interp3(lddmm.xI,torch.tensor(image,dtype=lddmm.dtype,device=lddmm.device),lddmm.phiiAi)
    elif deform_to == 'template-identity': # deform to template with identity
        out = lddmm.interp3(lddmm.xJ,torch.tensor(image,dtype=lddmm.dtype,device=lddmm.device),lddmm.XI)
    elif deform_to == 'target-identity':
        out = lddmm.interp3(lddmm.xI,torch.tensor(image,dtype=lddmm.dtype,device=lddmm.device),lddmm.XJ)
    return out.cpu().numpy()
    
import matplotlib.pyplot as plt
plt.ion()
def torch_register_daniel(template, target, sigmaR, eV, eL=0, eT=0, **kwargs):
    """daniel's version for demo to be replaced
    Perform a registration between <template> and <target>.
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
        'a':2, # smoothing kernel, scaled to pixel size
        'p':2,
        'niter':200,
        'eV':eV, # velocity
        'eL':eL, # linear
        'eT':eT, # translation
        'sigmaM':1.0, # sigmaM
        'sigmaR':sigmaR, # sigmaR
        'sigmaA':None, # for EM algorithm
        'nt':3, # number of time steps in velocity field           
        'order':2, # polynomial order
        'draw':False,
        'xI':None,
        'xJ':None,
    }
    # Update parameters with kwargs.
    arguments.update(kwargs)
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    dtype = torch.float64
        
    lddmm = Transformer(torch.tensor(template,dtype=dtype,device=device),
                        torch.tensor(target,dtype=dtype,device=device),**arguments)
    if arguments['draw']:
        f1 = plt.figure()
        f2 = plt.figure()
        if arguments['sigmaA'] is not None:
            f3 = plt.figure()
    for it in range(arguments['niter']):
        lddmm.forward()
        lddmm.cost()
        if arguments['sigmaA'] is not None:
            lddmm.weights()
        if it >=0 and arguments['eV']>-1.0:
            lddmm.step_v(eV=arguments['eV'])
        lddmm.step_A(eT=arguments['eT'],eL=arguments['eL'])
        
        if arguments['draw'] and not it%5:        
            plt.close(f1)
            f1 = plt.figure()
            ax = f1.add_subplot(1,1,1)    
            ERsave = lddmm.ERsave    
            EMsave = lddmm.EMsave
            Esave = lddmm.Esave
            ax.plot(ERsave)
            ax.plot(EMsave)
            ax.plot(Esave)
            ax.legend(['ER','EM','E'])
            f1.canvas.draw()

            
            plt.close(f2)
            f2 = plt.figure()
            #lddmm.show_image(lddmm.fAphiI.cpu().numpy(),fig=f2) # or show err?
            lddmm.show_image(lddmm.err.cpu().numpy(),fig=f2) # or show err?
            f2.canvas.draw()
            
            if arguments['sigmaA'] is not None:
                plt.close(f3)
                f3 = plt.figure()
                lddmm.show_image(lddmm.WM.cpu().numpy(),fig=f3,clim=[0,1])            
                f3.canvas.draw()
                
            plt.pause(0.0001)
        print(f'Completed iteration {it}, E={lddmm.Esave[-1]}, EM={lddmm.EMsave[-1]}, ER={lddmm.ERsave[-1]}')
            
    
    return {
        'Aphis':lddmm.Aphi, 
        'phis':lddmm.phi, 
        'phiinvs':lddmm.phii, 
        'phiinvAinvs':lddmm.phiiAi, 
        'A':lddmm.A, 
        'lddmm':lddmm}
    
torch_apply_transform = torch_apply_transform_daniel
torch_register = torch_register_daniel
