import numpy as np

class ImageMetaData():
    """Container class for metadata about an image. Stores:
        - nxyz (np.ndarray): image shape - computed from provided image.
        - dxyz (np.ndarray or scalar): image resolution - if dxyz is a scalar, it is upcast to the length of nxyz.
        - xyz (np.ndarray): image coordinates - computed, not provided.
        - origin (np.ndarray): origin for xyz coordinates - default is center.
            Options:
            - 'center'
            - 'zero'
            - np.ndarray of the same length as nxyz
            - scalar, upcast to np.ndarray of same length as nxyz"""
    
    def __init__(self, dxyz, nxyz=None, image=None, origin="center", name=None):
        """If nxyz is provided, use it as nxyz, the image's shape.
        If image is provided, use its shape as nxyz.
        If both nxyz and image are provided and indicate different shapes, raise error.
        If neither nxyz nor image are provided, raise error.
        If origin is provided, it is used to compute xyz.
        If origin is not provided, xyz is centered by default."""

        # Instantiate attributes.
        self.dxyz = None
        self.nxyz = None
        self.xyz = None
        self.name = None

        # Populate attributes.

        # nxyz attribute.

        # Validate agreement between nxyz and image.
        if nxyz is not None and image is not None:
            if not all(nxyz == np.array(image).shape):
                raise ValueError(f"nxyz and image were both provided, but nxyz does not match the shape of image.\n\
nxyz: {nxyz}, image.shape: {image.shape}.")

        # If nxyz is provided but image is not.
        if nxyz is not None:
            # Cast as np.ndarray.
            if not isinstance(nxyz, np.ndarray):
                nxyz = np.array(nxyz) # Side effect: breaks alias.
            # If nxyz is multidimensional, raise error.
            if nxyz.ndim > 1:
                raise ValueError(f"nxyz cannot be multidimensional.\nnxyz.ndim: {nxyz.ndim}.")
            # If nxyz is 0-dimensional, upcast to 1-dimensional. A perverse case.
            if nxyz.ndim == 0:
                nxyz = np.array([nxyz])
            # If nxyz is 1-dimensional, set nxyz attribute.
            if nxyz.ndim == 1:
                self.nxyz = nxyz
        # If image is provided but nxyz is not.
        # Will fail if image cannot be cast as a np.ndarray.
        elif image is not None:
            # Cast as np.ndarray.
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            # If image is 0-dimensional, upcast to 1-dimensional. A perverse case.
            if image.ndim == 0:
                image = np.array([image])
            # image is a non-zero-dimensional np.ndarray. Set nxyz attribute.
            self.nxyz = np.array(image.shape)
        else:
            raise RuntimeError(f"At least one of nxyz and image must be provided. Both were received as their default value: None.")
        
        # dxyz attribute.

        # Cast as np.ndarray.
        if not isinstance(dxyz, np.ndarray):
            dxyz = np.array(dxyz) # Side effect: breaks alias.
        # if dxyz is multidimensional, raise error.
        if dxyz.ndim > 1:
            raise ValueError(f"dyxz must be 1-dimensional. The value provided is {dxyz.ndim}-dimensional.")
        # If dxyz is 0-dimensional, upcast to match the length of self.nxyz.
        if dxyz.ndim == 0:
            dxyz = np.array([dxyz]*len(self.nxyz))
        # dxyz is 1-dimensional.
        # If dxyz is 1-dimensional and matches the length of self.nxyz, set dxyz attribute.
        if len(dxyz) == len(self.nxyz):
            self.dxyz = dxyz
        else:
            raise ValueError(f"dyxz must be either 0-dimensional or 1-dimensional and matching the length of nxyz or the shape of image.\n\
len(dxyz): {len(dxyz)}.")
        
        # xyz attribute.

        # Upcast self.dxyz with nxyz if possible.
        if self.dxyz.ndim == 0:
            self.dxyz = np.array([self.dxyz]*len(self.nxyz))
        
        # Verify that the lengths of self.dxyz and self.nxyz match.
        if len(self.dxyz) != len(self.nxyz):
            raise ValueError(f"dxyz must either be a scalar or match the length of nxyz or the shape of image.\n\
len(dxyz): {len(dxyz)}.")

        # Instantiate self.xyz.
        # self.xyz is a list of np.ndarray objects of type float and represents the physical coordinates in each dimension.
        self.xyz = [np.arange(nxyz_i).astype(float)*dxyz_i for nxyz_i, dxyz_i in zip(self.nxyz, self.dxyz)]
        
        # Shift self.xyz.
        # The default choice is to center self.xyz.
        if isinstance(origin, str):
            if origin == "center": # Default.
                # Offset by the mean along each dimension.
                for coords in self.xyz:
                    coords -= np.mean(coords)
            elif origin == "zero":
                pass
            else:
                raise ValueError(f"Unsupported value for origin. Supported string values include ['center', 'zero'].\n\
origin: {origin}.")
        elif isinstance(origin, (int, float, list, np.ndarray)):
            if isinstance(origin, (int, float, list)):
                # Cast to np.ndarray.
                origin = np.array(origin)
            # origin is a np.ndarray.
            # If origin has length 1, broadcast to match self.nxyz.
            if origin.ndim == 0:
                origin = np.array(list(origin) * len(self.nxyz))
            # If the length of origin matches the length of self.nxyz, perform offset in each dimension.
            if len(origin) == len(self.nxyz):
                for dim, coords in enumerate(self.xyz):
                    coords -= origin[dim]
            else:
                raise ValueError(f"origin must either be a scalar, have length 1, or match the length of nxyz or the shape of image.\n\
len(origin): {len(origin)}.")
        else:
            raise ValueError(f"must be one of the following types: [str, int, float, list, np.ndarray].\n\
type(origin): {type(origin)}.")

        # name attribute.

        self.name = name


class Image(ImageMetaData):
    """Subclass of ImageMetaData that also stores the image data itself.
    Attributes:
        - image, the image data
        - dxyz, the resolution in each dimension
        - nxyz, the shape of the image
        - xyz, the coordinates of the image
        - name, optional name"""

    def __init__(self, image, dxyz, origin="center", name=None):
        """Validation is handled by ImageMetaData.__init__."""

        super().__init__(dxyz=dxyz, image=image, origin=origin, name=name)

        # Redo validation on image to set image attribute.
        image = np.array(image) # Side effect: breaks alias
        # If image is 0-dimensional, upcast to 1-dimensional. A perverse case.
        if image.ndim == 0:
            image = np.array([image])
        # Set image attribute.
        self.image = image