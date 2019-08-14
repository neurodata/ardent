from ardent import imageData
import numpy as np

class Test_ImageMetaData:
    """Test class for ImageMetaData."""

    def test_validate_nxyz(self):

        # Test valid runs.

        valid_inputs = [
            # Scalar nxyz.
            {'nxyz':0, 'image':None}, 
            {'nxyz':1, 'image':None}, 
            {'nxyz':9, 'image':None}, 
            # nxyz from 1-D iterables.
            {'nxyz':[1, 2, 3], 'image':None}, 
            {'nxyz':np.array([1, 2, 3]), 'image':None}, 
            # Varying-dimensional image of types: scalar, list, np.ndarray.
            {'nxyz':None, 'image':5}, 
            {'nxyz':None, 'image':[1, 2]}, 
            {'nxyz':None, 'image':[[1, 2], [3, 4], [5, 6]]}, 
            {'nxyz':None, 'image':np.arange(3*4*5).reshape(3,4,5)}, 
            # nxyz and image agreement.
            {'nxyz':1, 'image':5}, 
            {'nxyz':[2], 'image':[1, 2]}, 
            {'nxyz':[3,2], 'image':[[1, 2], [3, 4], [5, 6]]}, 
            {'nxyz':np.array([3,4,5]), 'image':np.arange(3*4*5).reshape(3,4,5)}
        ]
        valid_outputs = [
            # Scalar nxyz.
            np.array([0]), 
            np.array([1]), 
            np.array([9]), 
            # nxyz from 1-D iterables.
            np.array([1, 2, 3]), 
            np.array([1, 2, 3]), 
            # Varying-dimensional image.
            np.array([1]), 
            np.array([2]), 
            np.array([3, 2]), 
            np.array([3, 4, 5]), 
            # nxyz and image agreement.
            np.array([1]), 
            np.array([2]), 
            np.array([3, 2]), 
            np.array([3, 4, 5])
        ]

        for inputs, output in zip(valid_inputs, valid_outputs):
            assert np.all(imageData.ImageMetaData._validate_nxyz(**inputs) == output)
        
        # Test invalid runs.

