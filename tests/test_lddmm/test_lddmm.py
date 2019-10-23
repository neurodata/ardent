import pytest

import numpy as np
from scipy.linalg import inv
from scipy.ndimage import rotate
import ardent

from ardent.lddmm import register
from ardent.lddmm.lddmm import _generate_position_field
from ardent.lddmm.lddmm import _apply_position_field
from ardent.lddmm import apply_transform

@pytest.fixture
def template_and_target():
    '''
    Creates and returns a template and target as [template, target]
    respectively a sphere and an ellipsoid of 1's on a background of 0's, 
    then throwin into ardent.basic_preprocessing.
    '''

    template = np.zeros([12]*3, dtype=float)
    r = 5
    for i in range(template.shape[0]):
        for j in range(template.shape[1]):
            for k in range(template.shape[2]):
                if np.sqrt((i-6)**2 + (j-6)**2 + (k-6)**2) <= r:
                    template[i,j,k] = 1

    target = np.zeros([18, 18, 12], dtype=float)
    a, b, c = 8, 8, 5
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            for k in range(target.shape[2]):
                if (i-9)**2 / a**2 + (j-9)**2 / b**2 + (k-6)**2 / c**2 <= 1:
                    target[i,j,k] = 1

    return ardent.basic_preprocessing([template, target])

def test_register():
    pass


@pytest.mark.parametrize('deform_to', ['template', 'target'])
def test__generate_position_field(deform_to):

    # Test identity affine and velocity_fields.

    num_timesteps = 10

    template_shape = (3,4,5)
    template_resolution = 1
    target_shape = (2,4,6)
    target_resolution = 1
    velocity_fields = np.zeros((*template_shape, num_timesteps, len(template_shape)))
    velocity_field_resolution = 1
    affine = np.eye(4)

    if deform_to == 'template':
        expected_output = ardent.utilities._compute_coords(template_shape, template_resolution)
    elif deform_to == 'target':
        expected_output = ardent.utilities._compute_coords(target_shape, target_resolution)

    position_field = _generate_position_field(affine=affine, velocity_fields=velocity_fields, velocity_field_resolution=velocity_field_resolution, 
        template_shape=template_shape, template_resolution=template_resolution, target_shape=target_shape, target_resolution=target_resolution, deform_to=deform_to)

    assert np.array_equal(position_field, expected_output)

    # Test identity affine and constant shift velocity_fields.

    num_timesteps = 10

    template_shape = (3,4,5)
    template_resolution = 1
    target_shape = (2,4,6)
    target_resolution = 1
    velocity_fields = np.ones((*template_shape, num_timesteps, len(template_shape)))
    velocity_field_resolution = 1
    affine = np.eye(4)

    if deform_to == 'template':
        expected_output = ardent.utilities._compute_coords(template_shape, template_resolution) + 1
    elif deform_to == 'target':
        expected_output = ardent.utilities._compute_coords(target_shape, target_resolution) - 1

    position_field = _generate_position_field(affine=affine, velocity_fields=velocity_fields, velocity_field_resolution=velocity_field_resolution, 
        template_shape=template_shape, template_resolution=template_resolution, target_shape=target_shape, target_resolution=target_resolution, deform_to=deform_to)

    assert np.allclose(position_field, expected_output)

    # Test rotational affine and identity velocity_fields.

    num_timesteps = 10

    template_shape = (3,4,5)
    template_resolution = 1
    target_shape = (2,4,6)
    target_resolution = 1
    velocity_fields = np.zeros((*template_shape, num_timesteps, len(template_shape)))
    velocity_field_resolution = 1
    # Indicates a 90 degree rotation to the right.
    affine = np.array([
        [0,1,0,0],
        [-1,0,0,0],
        [0,0,1,0],
        [0,0,0,1],
    ])

    if deform_to == 'template':
        expected_output = ardent.utilities._multiply_by_affine(
            ardent.utilities._compute_coords(template_shape, template_resolution), 
            affine, 3
        )
    elif deform_to == 'target':
        expected_output = ardent.utilities._multiply_by_affine(
            ardent.utilities._compute_coords(target_shape, target_resolution), 
            inv(affine), 3
        )

    position_field = _generate_position_field(affine=affine, velocity_fields=velocity_fields, velocity_field_resolution=velocity_field_resolution, 
        template_shape=template_shape, template_resolution=template_resolution, target_shape=target_shape, target_resolution=target_resolution, deform_to=deform_to)
    
    assert np.allclose(position_field, expected_output)


def test__apply_position_field():
    
    # Test simplest identity position_field.

    subject = np.arange(3*4).reshape(3,4)
    subject_resolution = 1
    output_resolution = 1
    position_field_resolution = subject_resolution
    position_field = ardent.utilities._compute_coords(subject.shape, position_field_resolution)

    deformed_subject = _apply_position_field(
        subject=subject,
        subject_resolution=subject_resolution,
        output_resolution=output_resolution,
        position_field=position_field,
        position_field_resolution=position_field_resolution,
    )
    expected_output = subject
    assert np.allclose(deformed_subject, expected_output)

    # Test identity position_field with different output_resolution.

    subject = np.array([
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,0,0],
        [0,0,1,1,1,1,0,0],
        [0,0,1,1,1,1,0,0],
        [0,0,1,1,1,1,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
    ])
    subject_resolution = 1
    output_resolution = 2
    position_field_resolution = subject_resolution
    position_field = ardent.utilities._compute_coords(subject.shape, position_field_resolution)

    deformed_subject = _apply_position_field(
        subject=subject,
        subject_resolution=subject_resolution,
        output_resolution=output_resolution,
        position_field=position_field,
        position_field_resolution=position_field_resolution,
    )
    expected_output = np.array([
        [0,0,0,0],
        [0,1,1,0],
        [0,1,1,0],
        [0,0,0,0],
    ])
    assert np.allclose(deformed_subject, expected_output)
    
    # Test constant shifting position_field with simple extrapolation case.

    # Note: applying a leftward shift to the position_field is done by subtracting 1 from the appropriate dimension.
    # The corresponding effect on the deformed_subject is a shift to the right.

    subject = np.array([
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,0,0],
        [0,0,1,1,1,1,0,0],
        [0,0,1,1,1,1,0,0],
        [0,0,1,1,1,1,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
    ])
    subject_resolution = 1
    output_resolution = 1
    position_field_resolution = subject_resolution
    position_field = ardent.utilities._compute_coords(subject.shape, position_field_resolution) + [0, -1] # Shift to the left by 1.

    deformed_subject = _apply_position_field(
        subject=subject,
        subject_resolution=subject_resolution,
        output_resolution=output_resolution,
        position_field=position_field,
        position_field_resolution=position_field_resolution,
    )
    expected_output = np.array([
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,1,1,1,1,0],
        [0,0,0,1,1,1,1,0],
        [0,0,0,1,1,1,1,0],
        [0,0,0,1,1,1,1,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
    ])

    assert np.allclose(deformed_subject, expected_output)

    # Test constant shifting position_field, demonstrating idiosyncratic extrapolation behavior.

    subject = np.array([
        [0,0,0,0],
        [0,1,1,0],
        [0,1,1,0],
        [0,0,0,0],
    ])
    subject_resolution = 1
    output_resolution = 1
    position_field_resolution = subject_resolution
    position_field = ardent.utilities._compute_coords(subject.shape, position_field_resolution) + [0, -1] # Shift to the left by 1.

    deformed_subject = _apply_position_field(
        subject=subject,
        subject_resolution=subject_resolution,
        output_resolution=output_resolution,
        position_field=position_field,
        position_field_resolution=position_field_resolution,
    )
    expected_output = np.array([
        [0,0,0,0],
        [-1,0,1,1],
        [-1,0,1,1],
        [0,0,0,0],
    ])

    assert np.allclose(deformed_subject, expected_output)

    # Test rotational position_field.

    # Note: applying an affine indicating a clockwise-rotation to a position_field produces a position _ield rotated counter-clockwise.
    # The corresponding effect on the deformed_subject is a counter-clockwise rotation.

    subject = np.array([
        [0,1,0,0],
        [0,1,0,0],
        [0,1,0,0],
        [0,1,1,1],
    ])
    subject_resolution = 1
    output_resolution = 1
    position_field_resolution = subject_resolution
    # Indicates a 90 degree rotation to the right.
    affine = np.array([
        [0,1,0],
        [-1,0,0],
        [0,0,1],
    ])
    position_field = ardent.utilities._multiply_by_affine(
        ardent.utilities._compute_coords(subject.shape, position_field_resolution), 
        affine, 2
    )

    deformed_subject = _apply_position_field(
        subject=subject,
        subject_resolution=subject_resolution,
        output_resolution=output_resolution,
        position_field=position_field,
        position_field_resolution=position_field_resolution,
    )
    expected_output = np.array([
        [0,0,0,1],
        [0,0,0,1],
        [1,1,1,1],
        [0,0,0,0],
    ])
    
    assert np.allclose(deformed_subject, expected_output)


@pytest.mark.parametrize('deform_to', ['template', 'target'])
def test_apply_transform(deform_to):

    # Test identity position fields.
    
    subject = np.array([
        [0,0,0,0],
        [0,1,1,0],
        [0,1,1,0],
        [0,0,0,0],
    ])
    subject_resolution = 1
    template_shape = (3,4)
    template_resolution = 1
    target_shape = (2,5)
    target_resolution = 1

    affine_phi = ardent.utilities._compute_coords(template_shape, template_resolution)
    phi_inv_affine_inv = ardent.utilities._compute_coords(target_shape, target_resolution)

    if deform_to == 'template':
        output_resolution = np.copy(template_resolution)
    elif deform_to == 'target':
        output_resolution = np.copy(target_resolution)

    expected_output = _apply_position_field(subject, subject_resolution, output_resolution, 
        position_field=affine_phi if deform_to == 'template' else phi_inv_affine_inv, 
        position_field_resolution=template_resolution if deform_to == 'template' else target_resolution)

    deformed_subject = apply_transform(
        subject=subject, subject_resolution=subject_resolution, 
        affine_phi=affine_phi, phi_inv_affine_inv=phi_inv_affine_inv, 
        template_resolution=template_resolution, target_resolution=target_resolution, 
        output_resolution=output_resolution, deform_to=deform_to)

    assert np.allclose(deformed_subject, expected_output)
    assert np.array_equal(deformed_subject, expected_output)


def test_it_runs(): # TODO: delete test

    # Test identity quasi-two-dimensional sphere to sphere registration.

    template = np.array([[[(col-6)**2 + (row-6)**2 <= 4**2 for col in range(13)] for row in range(13)]]*3, int)
    template_resolution = 1
    target = np.array([[[(col-6)**2 + (row-6)**2 <= 4**2 for col in range(13)] for row in range(13)]]*3, int)
    target_resolution = 1
    translational_stepsize = 0.5
    linear_stepsize = 0.5
    deformative_stepsize = 0.5
    sigmaR = 2
    num_iterations = 200
    num_affine_only_iterations = 50
    initial_affine = np.eye(4)
    initial_velocity_fields = None
    num_timesteps = 5
    contrast_order = 3
    sigmaM=None
    smooth_length = None

    reg_output = register(
        template=template,
        template_resolution=template_resolution,
        target=target,
        target_resolution=target_resolution,
        translational_stepsize=translational_stepsize,
        linear_stepsize=linear_stepsize,
        deformative_stepsize=deformative_stepsize,
        sigmaR=sigmaR,
        num_iterations=num_iterations,
        num_affine_only_iterations=num_affine_only_iterations,
        initial_affine=initial_affine,
        initial_velocity_fields=initial_velocity_fields,
        num_timesteps=num_timesteps,
        contrast_order=contrast_order,
        sigmaM=sigmaM,
        smooth_length=smooth_length,
    )

    deformed_target = apply_transform(
        subject=target, 
        subject_resolution=target_resolution, 
        deform_to='template', 
        **reg_output,
    )

    deformed_template = apply_transform(
        subject=template, 
        subject_resolution=template_resolution, 
        deform_to='target', 
        **reg_output,
    )

    assert True # Just 'cuz.


@pytest.mark.skip # TODO: don't skip.
def test_register():
    
    # Test identity quasi-two-dimensional sphere to sphere registration.

    template = np.array([[[(col-6)**2 + (row-6)**2 <= 4**2 for col in range(13)] for row in range(13)]]*3, int)
    template_resolution = 1
    target = np.array([[[(col-6)**2 + (row-6)**2 <= 4**2 for col in range(13)] for row in range(13)]]*3, int)
    target_resolution = 1
    translational_stepsize = 0.5
    linear_stepsize = 0.5
    deformative_stepsize = 0.5
    sigmaR = 2
    num_iterations = 200
    num_affine_only_iterations = 50
    initial_affine = np.eye(4)
    initial_velocity_fields = None
    num_timesteps = 5
    contrast_order = 3
    sigmaM=None
    smooth_length = None

    reg_output = register(
        template=template,
        template_resolution=template_resolution,
        target=target,
        target_resolution=target_resolution,
        translational_stepsize=translational_stepsize,
        linear_stepsize=linear_stepsize,
        deformative_stepsize=deformative_stepsize,
        sigmaR=sigmaR,
        num_iterations=num_iterations,
        num_affine_only_iterations=num_affine_only_iterations,
        initial_affine=initial_affine,
        initial_velocity_fields=initial_velocity_fields,
        num_timesteps=num_timesteps,
        contrast_order=contrast_order,
        sigmaM=sigmaM,
        smooth_length=smooth_length,
    )

    deformed_target = apply_transform(
        subject=target, 
        subject_resolution=target_resolution, 
        deform_to='template', 
        **reg_output,
    )

    deformed_template = apply_transform(
        subject=template, 
        subject_resolution=template_resolution, 
        deform_to='target', 
        **reg_output,
    )

    assert np.allclose(deformed_template, target, rtol=1e-5, atol=1e-8)
    assert np.allclose(deformed_target, template, rtol=1e-5, atol=1e-8)
    
    # Test quasi-two-dimensional affine-only ellipse to ellipse registration.

    # template hhas shape (3, 9, 17) and semi-radii 2 and 6.
    template = np.array([[[(col-8)**2/6**2 + (row-4)**2/2**2 <= 1 for col in range(17)] for row in range(9)]]*3, int)
    template_resolution = 1
    # target is a rotation of template.
    target = rotate(template, 30, (0,1))
    target_resolution = 1
    translational_stepsize = 0.5
    linear_stepsize = 0.5
    deformative_stepsize = 0.5
    sigmaR = 2
    num_iterations = 100
    num_affine_only_iterations = 100
    initial_affine = np.eye(4)
    initial_velocity_fields = None
    num_timesteps = 5
    contrast_order = 3
    sigmaM=None
    smooth_length = None

    reg_output = register(
        template=template,
        template_resolution=template_resolution,
        target=target,
        target_resolution=target_resolution,
        translational_stepsize=translational_stepsize,
        linear_stepsize=linear_stepsize,
        deformative_stepsize=deformative_stepsize,
        sigmaR=sigmaR,
        num_iterations=num_iterations,
        num_affine_only_iterations=num_affine_only_iterations,
        initial_affine=initial_affine,
        initial_velocity_fields=initial_velocity_fields,
        num_timesteps=num_timesteps,
        contrast_order=contrast_order,
        sigmaM=sigmaM,
        smooth_length=smooth_length,
    )

    deformed_target = apply_transform(
        subject=target, 
        subject_resolution=target_resolution, 
        deform_to='template', 
        **reg_output,
    )

    deformed_template = apply_transform(
        subject=template, 
        subject_resolution=template_resolution, 
        deform_to='target', 
        **reg_output,
    )

    assert np.allclose(deformed_template, target, rtol=1e-5, atol=1e-8)
    assert np.allclose(deformed_target, template, rtol=1e-5, atol=1e-8)

    # Test quasi-two-dimensional sphere to ellipsoid registration.

    # template has shape (3, 13, 13) and radius 4.
    template = np.array([[[(col-6)**2 + (row-6)**2 <= 4**2 for col in range(13)] for row in range(13)]]*3, int)
    template_resolution = 1
    # target has shape (3, 9, 17) and semi-radii 2 and 6.
    target = np.array([[[(col-8)**2/6**2 + (row-4)**2/2**2 <= 1 for col in range(17)] for row in range(9)]]*3, int)
    target_resolution = 1
    translational_stepsize = 0.5
    linear_stepsize = 0.5
    deformative_stepsize = 0.5
    sigmaR = 2
    num_iterations = 200
    num_affine_only_iterations = 50
    initial_affine = np.eye(4)
    initial_velocity_fields = None
    num_timesteps = 5
    contrast_order = 3
    sigmaM=None
    smooth_length = None

    reg_output = register(
        template=template,
        template_resolution=template_resolution,
        target=target,
        target_resolution=target_resolution,
        translational_stepsize=translational_stepsize,
        linear_stepsize=linear_stepsize,
        deformative_stepsize=deformative_stepsize,
        sigmaR=sigmaR,
        num_iterations=num_iterations,
        num_affine_only_iterations=num_affine_only_iterations,
        initial_affine=initial_affine,
        initial_velocity_fields=initial_velocity_fields,
        num_timesteps=num_timesteps,
        contrast_order=contrast_order,
        sigmaM=sigmaM,
        smooth_length=smooth_length,
    )

    deformed_target = apply_transform(
        subject=target, 
        subject_resolution=target_resolution, 
        deform_to='template', 
        **reg_output,
    )

    deformed_template = apply_transform(
        subject=template, 
        subject_resolution=template_resolution, 
        deform_to='target', 
        **reg_output,
    )

    assert np.allclose(deformed_template, target, rtol=1e-5, atol=1e-8)
    assert np.allclose(deformed_target, template, rtol=1e-5, atol=1e-8)
