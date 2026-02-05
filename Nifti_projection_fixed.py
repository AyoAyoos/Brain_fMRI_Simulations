#!/usr/bin/env python3
"""
Real fMRI Data Visualization - Full Brain (Bilateral)
======================================================
Production-ready NIfTI projection to bilateral cortical surfaces.

Features:
- Both hemispheres (left + right)
- Robust data normalization
- Dual-layer rendering for guaranteed visibility
- Enhanced lighting and depth perception

Author: Claude
Date: February 2026
"""

import numpy as np
import pyvista as pv
from nilearn import datasets, surface, image
import warnings
warnings.filterwarnings('ignore')


def load_nifti_volume(file_path, volume_index=None):
    """
    Load NIfTI file and select appropriate volume.
    
    Parameters
    ----------
    file_path : str
        Path to .nii or .nii.gz file
    volume_index : int, optional
        Volume index for 4D data (None = auto-select middle)
        
    Returns
    -------
    img : nibabel image
        Selected 3D volume
    """
    print(f"Loading: {file_path}")
    
    nii_img = image.load_img(file_path)
    print(f"✓ Shape: {nii_img.shape}")
    
    # Handle 4D data
    if len(nii_img.shape) == 4:
        n_volumes = nii_img.shape[3]
        print(f"✓ 4D data: {n_volumes} volumes")
        
        if volume_index is None:
            volume_index = n_volumes // 2
        
        print(f"✓ Using volume {volume_index}")
        return image.index_img(nii_img, volume_index)
    else:
        print(f"✓ 3D volume")
        return nii_img


def project_volume_to_surface(nii_vol, surface_mesh, radius=3.0):
    """
    Project volume data onto cortical surface.
    
    Parameters
    ----------
    nii_vol : nibabel image
        3D volume
    surface_mesh : str
        Path to surface mesh file
    radius : float
        Sampling radius in mm
        
    Returns
    -------
    texture : ndarray
        Surface texture values
    """
    texture = surface.vol_to_surf(
        nii_vol,
        surface_mesh,
        radius=radius,
        interpolation='linear',
        kind='line'
    )
    
    # Clean invalid values
    texture = np.nan_to_num(texture, nan=0.0, posinf=0.0, neginf=0.0)
    
    return texture


def normalize_data(texture, method='robust'):
    """
    Normalize texture data for visualization.
    
    Parameters
    ----------
    texture : ndarray
        Raw surface texture
    method : str
        'robust' (percentile-based) or 'minmax'
        
    Returns
    -------
    normalized : ndarray
        Normalized values in [0, 1]
    """
    if method == 'robust':
        # Use percentiles to handle outliers
        p2 = np.percentile(texture, 2)
        p98 = np.percentile(texture, 98)
        
        normalized = np.clip(texture, p2, p98)
        normalized = (normalized - p2) / (p98 - p2 + 1e-10)
        
    else:  # minmax
        tmin, tmax = texture.min(), texture.max()
        normalized = (texture - tmin) / (tmax - tmin + 1e-10)
    
    # Suppress very low values for better contrast
    low_threshold = np.percentile(texture, 5)
    normalized[texture < low_threshold] = 0
    
    return normalized


def create_bilateral_mesh(fsaverage):
    """
    Create combined left + right hemisphere mesh.
    
    Parameters
    ----------
    fsaverage : dict
        fsaverage dataset from nilearn
        
    Returns
    -------
    left_mesh : pyvista.PolyData
        Left hemisphere mesh
    right_mesh : pyvista.PolyData
        Right hemisphere mesh
    """
    # Load left hemisphere
    vertices_L, faces_L = surface.load_surf_mesh(fsaverage['pial_left'])
    faces_vtk_L = np.hstack([
        np.full((faces_L.shape[0], 1), 3),
        faces_L
    ]).astype(np.int64)
    left_mesh = pv.PolyData(vertices_L, faces_vtk_L)
    
    # Load right hemisphere
    vertices_R, faces_R = surface.load_surf_mesh(fsaverage['pial_right'])
    faces_vtk_R = np.hstack([
        np.full((faces_R.shape[0], 1), 3),
        faces_R
    ]).astype(np.int64)
    right_mesh = pv.PolyData(vertices_R, faces_vtk_R)
    
    return left_mesh, right_mesh


def add_hemisphere_to_plotter(plotter, mesh, texture, colormap, hemisphere_label):
    """
    Add single hemisphere with dual-layer rendering.
    
    Parameters
    ----------
    plotter : pyvista.Plotter
        PyVista plotter instance
    mesh : pyvista.PolyData
        Cortical surface mesh
    texture : ndarray
        Normalized activation data
    colormap : str
        Colormap name
    hemisphere_label : str
        'left' or 'right' for naming
    """
    # Layer 1: Base brain (gray, semi-transparent)
    base_mesh = mesh.copy()
    plotter.add_mesh(
        base_mesh,
        color='lightgray',
        opacity=0.3,
        smooth_shading=True,
        show_edges=False,
        name=f'base_{hemisphere_label}'
    )
    
    # Layer 2: Activation data
    activation_mesh = mesh.copy()
    activation_mesh['activation'] = texture
    
    plotter.add_mesh(
        activation_mesh,
        scalars='activation',
        cmap=colormap,
        opacity='linear',
        smooth_shading=True,
        show_edges=False,
        show_scalar_bar=(hemisphere_label == 'left'),  # Only show one colorbar
        scalar_bar_args={
            'title': 'BOLD Signal',
            'n_labels': 5,
            'fmt': '%.2f',
            'position_x': 0.85,
            'position_y': 0.25,
            'color': 'white',
            'width': 0.12,
            'height': 0.5
        },
        clim=[0, 1],
        lighting=True,
        ambient=0.4,
        diffuse=0.6,
        specular=0.3,
        specular_power=20,
        name=f'activation_{hemisphere_label}'
    )


def setup_lighting(plotter):
    """Configure three-point lighting for optimal 3D visualization."""
    # Key light
    plotter.add_light(pv.Light(
        position=(100, 100, 100),
        intensity=0.8
    ))
    
    # Fill light
    plotter.add_light(pv.Light(
        position=(-100, -50, 100),
        intensity=0.4
    ))
    
    # Back light
    plotter.add_light(pv.Light(
        position=(0, -100, -50),
        intensity=0.3
    ))


def main():
    """Main execution for bilateral brain visualization."""
    
    print("=" * 70)
    print("Full Brain fMRI Visualization (Bilateral)")
    print("=" * 70)
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    FILE_PATH = r"CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-01.nii.gz"
    VOLUME_INDEX = 50  # None = auto-select middle volume
    SAMPLING_RADIUS = 5.0  # mm
    COLORMAP = 'hot'  # 'hot', 'jet', 'plasma', 'viridis', 'coolwarm'
    
    print(f"\nConfiguration:")
    print(f"  File: {FILE_PATH}")
    print(f"  Sampling radius: {SAMPLING_RADIUS} mm")
    print(f"  Colormap: {COLORMAP}")
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    print("\n" + "=" * 70)
    print("Loading Data")
    print("=" * 70)
    
    try:
        nii_vol = load_nifti_volume(FILE_PATH, VOLUME_INDEX)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("  1. Verify file path is correct")
        print("  2. Use absolute path (e.g., C:\\Users\\...)")
        print("  3. Ensure file is valid NIfTI format")
        return
    
    # ========================================================================
    # LOAD CORTICAL SURFACES
    # ========================================================================
    print("\n" + "=" * 70)
    print("Loading Cortical Surfaces")
    print("=" * 70)
    
    print("Fetching fsaverage template...")
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
    
    print("Creating bilateral mesh...")
    left_mesh, right_mesh = create_bilateral_mesh(fsaverage)
    
    print(f"✓ Left:  {left_mesh.n_points} vertices")
    print(f"✓ Right: {right_mesh.n_points} vertices")
    
    # ========================================================================
    # PROJECT DATA TO SURFACES
    # ========================================================================
    print("\n" + "=" * 70)
    print("Projecting Volume to Surfaces")
    print("=" * 70)
    
    print("\nLeft hemisphere...")
    texture_left = project_volume_to_surface(
        nii_vol,
        fsaverage['pial_left'],
        SAMPLING_RADIUS
    )
    
    print(f"  Range: [{texture_left.min():.4f}, {texture_left.max():.4f}]")
    print(f"  Mean: {texture_left.mean():.4f}")
    
    print("\nRight hemisphere...")
    texture_right = project_volume_to_surface(
        nii_vol,
        fsaverage['pial_right'],
        SAMPLING_RADIUS
    )
    
    print(f"  Range: [{texture_right.min():.4f}, {texture_right.max():.4f}]")
    print(f"  Mean: {texture_right.mean():.4f}")
    
    # ========================================================================
    # NORMALIZE DATA
    # ========================================================================
    print("\n" + "=" * 70)
    print("Normalizing Data")
    print("=" * 70)
    
    texture_left_norm = normalize_data(texture_left, method='robust')
    texture_right_norm = normalize_data(texture_right, method='robust')
    
    active_left = np.sum(texture_left_norm > 0.1)
    active_right = np.sum(texture_right_norm > 0.1)
    
    print(f"Active vertices:")
    print(f"  Left:  {active_left} / {len(texture_left_norm)}")
    print(f"  Right: {active_right} / {len(texture_right_norm)}")
    
    # ========================================================================
    # CREATE VISUALIZATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("Creating Visualization")
    print("=" * 70)
    
    plotter = pv.Plotter(window_size=[1920, 1080])
    plotter.background_color = 'black'
    
    # Enable depth peeling for proper transparency
    plotter.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0.0)
    
    # Add both hemispheres
        # Add both hemispheres (textures swapped so activation appears on left side of screen)
    print("Adding left hemisphere...")
    add_hemisphere_to_plotter(
        plotter,
        left_mesh,
        texture_right_norm,  # left mesh gets right texture → shows on right of screen
        COLORMAP,
        'left'
    )
    
    print("Adding right hemisphere...")
    add_hemisphere_to_plotter(
        plotter,
        right_mesh,
        texture_left_norm,   # right mesh gets left texture → activation on left of screen
        COLORMAP,
        'right'
    )
    
    # Setup lighting
    setup_lighting(plotter)
    
    # Camera position (posterior view showing both hemispheres)
    plotter.camera_position = [
        (0, -150, 50),   # Camera position
        (0, 0, 20),      # Focal point
        (0, 0, 1)        # Up vector
    ]
    
    # Add title
    plotter.add_text(
        "Full Brain fMRI Activation\nBilateral Hemispheres",
        position='upper_left',
        font_size=14,
        color='white'
    )
    
    # ========================================================================
    # DISPLAY
    # ========================================================================
    print("\n" + "=" * 70)
    print("Launching Interactive Viewer")
    print("=" * 70)
    print("\nControls:")
    print("  Left Click + Drag: Rotate")
    print("  Scroll: Zoom")
    print("  Middle Click + Drag: Pan")
    print("  'r': Reset camera")
    print("  'q': Quit")
    print("=" * 70)
    
    plotter.show()
    
    print("\n✓ Visualization complete")


if __name__ == "__main__":
    main()