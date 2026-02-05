# Real-Time 3D fMRI BOLD Signal Simulation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

A high-performance, GPU-accelerated visualization system for simulating Blood-Oxygen-Level-Dependent (BOLD) signals in functional MRI using Python, PyVista, and Nilearn.

## Features

- 🧠 **Anatomically Accurate**: Uses standard `fsaverage` cortical surface meshes
- 🎨 **Glass Brain Rendering**: Multi-layer transparency with depth peeling
- ⚡ **GPU-Accelerated**: Real-time updates at 60+ FPS on modern hardware
- 🔬 **Neuroscience-Based**: Simulates hemodynamic response function (HRF)
- 🎮 **Interactive**: Fully rotatable, zoomable 3D visualization
- 📊 **Customizable**: Adjustable ROI locations, activation patterns, and visual parameters

## Demo

The simulation creates a "Glass Brain" where:

- The cortical surface appears semi-transparent (ghosted)
- Active regions pulse with red/yellow glow
- Intensity follows the hemodynamic response function
- Updates occur in real-time (30-60 FPS)

## Installation


## Dataset

Due to file size constraints, the primary dataset (**CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-01.nii.gz**) can be downloaded here:
[Download Dataset from Google Drive](https://drive.google.com/drive/folders/1H_CMnKsQs_tzx8dDCshSVyUeMO8Hf8rG?usp=sharing)


### Requirements

- Python 3.8 or higher
- GPU with OpenGL 3.3+ support (recommended)
- 4GB+ RAM

### Quick Install

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install numpy scipy pyvista nilearn matplotlib
```

### Detailed Dependencies

```bash
# Core visualization
pip install pyvista>=0.43.0

# Neuroimaging data
pip install nilearn>=0.10.0

# Numerical computing
pip install numpy>=1.24.0 scipy>=1.10.0

# Optional: for advanced features
pip install vtk>=9.2.0  # VTK backend (usually installed with PyVista)
pip install matplotlib>=3.7.0  # For color maps
```

### Verify Installation

```bash
python -c "import pyvista as pv; print(f'PyVista {pv.__version__}')"
python -c "import nilearn; print('Nilearn OK')"
```

## Usage

### Basic Usage

```bash
python Nifti_projection_fixed.py
```

The script will:

1. Download the fsaverage cortical mesh (~50MB, first run only)
2. Initialize the BOLD signal simulator
3. Open an interactive 3D window with the pulsating brain

### Interactive Controls

| Action       | Control                                   |
| ------------ | ----------------------------------------- |
| Rotate view  | Left mouse button + drag                  |
| Pan view     | Middle mouse button + drag                |
| Zoom         | Right mouse button + drag OR scroll wheel |
| Reset camera | Press `r`                               |
| Quit         | Press `q` OR close window               |

### Customization

Modify these parameters in the `main()` function:

```python
# Change ROI location (motor cortex, visual cortex, etc.)
roi_center_vertex = 5000  # Try: 8000 (occipital), 3000 (frontal)

# Adjust activation radius (in millimeters)
roi_radius = 20.0  # Larger = broader activation

# Change pulsation speed
activation_frequency = 0.3  # Hz (0.1-1.0 typical)

# Modify color scheme
cmap = 'hot'  # Options: 'jet', 'plasma', 'viridis', 'inferno'
```

## Technical Details

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Application                          │
├─────────────────────────────────────────────────────────────┤
│  BoldSignalSimulator                                         │
│  ├─ Gaussian ROI Weighting                                   │
│  ├─ HRF Temporal Modulation                                  │
│  └─ Vectorized NumPy Updates                                 │
├─────────────────────────────────────────────────────────────┤
│  PyVista Rendering Pipeline                                  │
│  ├─ Depth Peeling (Transparency)                             │
│  ├─ Scalar Opacity Mapping                                   │
│  └─ In-place GPU Memory Updates                              │
├─────────────────────────────────────────────────────────────┤
│  VTK Backend                                                 │
│  └─ OpenGL Hardware Acceleration                             │
└─────────────────────────────────────────────────────────────┘
```

### BOLD Signal Model

The simulation implements a simplified hemodynamic response:

```
BOLD(t, v) = baseline + HRF(t) × Gaussian(distance(v, ROI_center))

where:
  - HRF(t) = 0.5 + 0.5 × sin(2πft)  [simplified sinusoidal model]
  - Gaussian(d) = exp(-0.5 × (d/σ)²)
  - baseline = 0.1 (resting-state activity)
```

**Note**: For research applications, replace the sinusoidal HRF with the canonical gamma function:

```python
def canonical_hrf(t, peak_time=5, undershoot_time=15):
    """Gamma function-based HRF"""
    h1 = (t ** (peak_time - 1)) * np.exp(-t)
    h2 = (t ** (undershoot_time - 1)) * np.exp(-t) / 6
    return h1 - h2
```

### Performance Optimization

The code achieves real-time performance through:

1. **Pre-computed Weights**: Gaussian activation pattern calculated once
2. **Vectorized Operations**: NumPy array operations (no Python loops)
3. **In-place Updates**: Direct GPU memory modification via `mesh['activation'] = ...`
4. **Depth Peeling**: Hardware-accelerated transparency (8 peeling layers)
5. **Fixed Color Range**: Prevents re-computation of color mapping bounds

**Expected Performance**:

- CPU: Intel i5/AMD Ryzen 5 or better
- GPU: Integrated graphics → 30 FPS, Dedicated GPU → 60+ FPS
- RAM: ~2GB used (mesh + textures)

### File Structure

```
fmri_bold_simulation.py
├─ BoldSignalSimulator      # Core simulation engine
│  ├─ __init__()            # Initialize with mesh and ROI parameters
│  ├─ _compute_activation_weights()  # Pre-compute Gaussian kernel
│  ├─ simulate_hrf()        # Hemodynamic response function
│  └─ update()              # Frame update logic
│
├─ load_cortical_mesh()     # Nilearn data loader
├─ create_glass_brain_plotter()  # PyVista renderer setup
└─ main()                   # Application entry point
```

## Advanced Usage

### Multiple ROIs

Simulate simultaneous activation in different brain regions:

```python
# Create multiple simulators
motor_roi = BoldSignalSimulator(mesh, roi_center_vertex=5000, roi_radius=15)
visual_roi = BoldSignalSimulator(mesh, roi_center_vertex=8000, roi_radius=20)

# In update callback:
motor_signal = motor_roi.update(dt)
visual_signal = visual_roi.update(dt)
combined_signal = np.maximum(motor_signal, visual_signal)  # Combine activations
mesh['activation'] = combined_signal
```

### Task-Based Paradigm

Implement a block-design fMRI experiment:

```python
def task_paradigm(t, block_duration=10, rest_duration=5):
    """Alternating task/rest blocks"""
    cycle_time = block_duration + rest_duration
    phase = t % cycle_time
    return 1.0 if phase < block_duration else 0.0

# In simulate_hrf:
task_state = task_paradigm(t)
amplitude = task_state * (0.5 + 0.5 * np.sin(2 * np.pi * self.frequency * t))
```

### Export Animation

Record the visualization as a video:

```python
# In main(), before plotter.show():
plotter.open_movie('bold_simulation.mp4', framerate=30)

# In callback function:
plotter.write_frame()

# After plotter.show():
plotter.close()
```

### Custom Color Maps

Use scientific color maps optimized for perceptual uniformity:

```python
from matplotlib import cm
from matplotlib.colors import ListedColormap

# Create custom colormap
colors = cm.get_cmap('viridis', 256)(np.linspace(0, 1, 256))
colors[:10, 3] = 0  # Make lowest values fully transparent
custom_cmap = ListedColormap(colors)

plotter.add_mesh(mesh, cmap=custom_cmap, ...)
```

## Troubleshooting

### Common Issues

**Issue**: "Cannot download fsaverage dataset"

```bash
# Solution: Manually download to cache
python -c "from nilearn import datasets; datasets.fetch_surf_fsaverage()"
```

**Issue**: Low FPS (<20)

```bash
# Solutions:
# 1. Reduce depth peeling layers
plotter.enable_depth_peeling(number_of_peels=4)

# 2. Lower window resolution
plotter = pv.Plotter(window_size=[1280, 720])

# 3. Disable smooth shading
plotter.add_mesh(mesh, smooth_shading=False, ...)
```

**Issue**: "OpenGL version too low"

```bash
# Check OpenGL version
python -c "import pyvista as pv; pv.Report()"

# Update graphics drivers or use software rendering
export MESA_GL_VERSION_OVERRIDE=3.3  # Linux
```

**Issue**: "Segmentation fault on macOS"

```bash
# Use XQuartz or conda-installed VTK
conda install -c conda-forge vtk pyvista
```

## Scientific Background

### Neurovascular Coupling

The BOLD signal reflects the coupling between neural activity and blood flow:

1. **Neural Activation**: Neurons fire in response to stimuli
2. **Metabolic Demand**: Active neurons consume oxygen
3. **Vascular Response**: Blood flow increases (overcompensation)
4. **BOLD Contrast**: Ratio of oxygenated/deoxygenated hemoglobin changes
5. **MRI Signal**: T2* contrast detects this magnetic susceptibility change

### Hemodynamic Response Function (HRF)

The HRF describes the temporal relationship between neural activity and BOLD signal:

- **Onset**: ~1-2 seconds after stimulus
- **Peak**: ~5-6 seconds
- **Undershoot**: ~10-15 seconds (post-peak dip)
- **Return to Baseline**: ~20-30 seconds

This temporal delay is why fMRI has poor temporal resolution (~2s) despite excellent spatial resolution (~2-3mm).

## Applications

- **Education**: Teaching fMRI principles and neurovascular coupling
- **Methods Development**: Testing new analysis pipelines
- **Protocol Design**: Planning fMRI experiments and ROI selection
- **Visualization**: Creating figures and animations for publications
- **Demos**: Science outreach and public engagement

## Contributing

Contributions welcome! Areas for enhancement:

- [ ] Implement canonical HRF (double-gamma function)
- [ ] Add connectivity patterns between regions
- [ ] Support bilateral hemisphere rendering
- [ ] Integrate with real fMRI data (NIfTI files)
- [ ] Add GUI controls for parameter adjustment
- [ ] Implement event-related designs
- [ ] Support volumetric rendering (not just surface)

## Citation

If you use this code in academic work, please cite:

```bibtex
@software{fmri_bold_simulation,
  author = {Claude (Anthropic)},
  title = {Real-Time 3D fMRI BOLD Signal Simulation},
  year = {2026},
  url = {https://github.com/yourusername/fmri-bold-simulation}
}
```

## References

1. Huettel, S. A., Song, A. W., & McCarthy, G. (2014). *Functional Magnetic Resonance Imaging* (3rd ed.). Sinauer Associates.
2. Poldrack, R. A., Mumford, J. A., & Nichols, T. E. (2011). *Handbook of Functional MRI Data Analysis*. Cambridge University Press.
3. Lindquist, M. A., et al. (2009). Modeling the hemodynamic response function in fMRI: Efficiency, bias and mis-modeling. *NeuroImage*, 45(1), S187-S198.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **Nilearn**: For providing standardized neuroimaging datasets
- **PyVista**: For the excellent VTK Python interface
- **VTK**: For robust 3D rendering capabilities
- **fsaverage**: FreeSurfer average brain template (Fischl et al., 1999)

## Contact

For questions or feedback, please open an issue on GitHub or reach out to the maintainers.

---

**Disclaimer**: This is a simplified simulation for educational and visualization purposes. It does not replace proper fMRI analysis software (FSL, SPM, AFNI, etc.) for research applications.
