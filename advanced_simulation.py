#!/usr/bin/env python3
"""
Advanced fMRI BOLD Simulation
===============================
Multi-ROI activation with canonical hemodynamic response function.

This example demonstrates:
- Canonical double-gamma HRF
- Multiple simultaneous ROIs
- Task-based block design
- Network connectivity patterns

Author: Claude
Date: February 2026
"""

import numpy as np
import pyvista as pv
from nilearn import datasets
from nilearn.surface import load_surf_mesh
import time


class CanonicalHRF:
    """
    Canonical Hemodynamic Response Function using double-gamma model.

    This is the standard HRF model used in SPM and other fMRI analysis packages.
    """

    @staticmethod
    def evaluate(t, peak_time=6, peak_disp=1, undershoot_time=16,
                 undershoot_disp=1, ratio=6):
        """
        Double-gamma HRF function.

        Parameters
        ----------
        t : float or array
            Time points to evaluate (in seconds)
        peak_time : float
            Time to peak of positive response
        peak_disp : float
            Dispersion of positive response
        undershoot_time : float
            Time to peak of undershoot
        undershoot_disp : float
            Dispersion of undershoot
        ratio : float
            Ratio of positive to negative response

        Returns
        -------
        float or array
            HRF values at time t
        """
        from scipy.stats import gamma as gamma_dist

        # Positive response (main peak)
        shape1 = peak_time / peak_disp
        scale1 = peak_disp
        positive = gamma_dist.pdf(t, shape1, scale=scale1)

        # Negative response (undershoot)
        shape2 = undershoot_time / undershoot_disp
        scale2 = undershoot_disp
        negative = gamma_dist.pdf(t, shape2, scale=scale2) / ratio

        # Combine and normalize
        hrf = positive - negative
        return hrf / hrf.max() if hrf.max() > 0 else hrf


class MultiROISimulator:
    """
    Simulates BOLD signals across multiple brain regions.

    Supports:
    - Independent activation patterns per ROI
    - Functional connectivity between regions
    - Task-based paradigms (block/event-related designs)
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self.vertices = mesh.points
        self.n_vertices = len(self.vertices)

        # ROI configurations
        self.rois = []

        # HRF model
        self.hrf_model = CanonicalHRF()

        # Task paradigm state
        self.time = 0.0
        self.task_active = False

        # Stimulus history for convolution with HRF
        self.max_hrf_duration = 32  # seconds
        self.stimulus_history = []

    def add_roi(self, center_vertex, radius, label='ROI'):
        """
        Add a region of interest.

        Parameters
        ----------
        center_vertex : int
            Central vertex index
        radius : float
            Spatial extent (mm)
        label : str
            ROI name/label
        """
        center_pos = self.vertices[center_vertex]
        distances = np.linalg.norm(self.vertices - center_pos, axis=1)

        # Gaussian spatial kernel
        sigma = radius / 2.5
        weights = np.exp(-0.5 * (distances / sigma) ** 2)
        weights = weights / weights.max()

        roi_info = {
            'center': center_vertex,
            'position': center_pos,
            'radius': radius,
            'weights': weights,
            'label': label,
            'baseline': 0.05,
            'amplitude': 1.0
        }

        self.rois.append(roi_info)
        print(f"Added ROI: {label} at vertex {center_vertex}")

    def block_design_paradigm(self, t, block_on=10, block_off=10, delay=0):
        """
        Classic block-design paradigm: alternating ON/OFF periods.

        Parameters
        ----------
        t : float
            Current time (seconds)
        block_on : float
            Duration of ON blocks
        block_off : float
            Duration of OFF blocks
        delay : float
            Initial delay before first block

        Returns
        -------
        float
            Task state (1.0 = ON, 0.0 = OFF)
        """
        adjusted_time = t - delay
        if adjusted_time < 0:
            return 0.0

        cycle_duration = block_on + block_off
        phase = adjusted_time % cycle_duration
        return 1.0 if phase < block_on else 0.0

    def event_related_paradigm(self, t, events, event_duration=0.5):
        """
        Event-related design: discrete stimulus presentations.

        Parameters
        ----------
        t : float
            Current time
        events : list of float
            List of event onset times
        event_duration : float
            Duration of each event

        Returns
        -------
        float
            Event presence (1.0 during event, 0.0 otherwise)
        """
        for event_time in events:
            if event_time <= t < event_time + event_duration:
                return 1.0
        return 0.0

    def convolve_with_hrf(self, stimulus_train):
        """
        Convolve stimulus train with canonical HRF.

        Parameters
        ----------
        stimulus_train : array
            Binary time series (1 = stimulus present, 0 = absent)

        Returns
        -------
        array
            BOLD signal after HRF convolution
        """
        # Time points for HRF (0 to 32 seconds)
        hrf_time = np.arange(0, self.max_hrf_duration, 0.1)
        hrf_values = self.hrf_model.evaluate(hrf_time)

        # Convolve (numpy is fast for this)
        bold_signal = np.convolve(stimulus_train, hrf_values, mode='full')
        return bold_signal[:len(stimulus_train)]

    def update(self, dt, paradigm='block'):
        """
        Update BOLD signal for all ROIs.

        Parameters
        ----------
        dt : float
            Time step
        paradigm : str
            'block' or 'event' or 'resting'
        """
        self.time += dt

        # Determine task state
        if paradigm == 'block':
            task_state = self.block_design_paradigm(
                self.time,
                block_on=10,
                block_off=8
            )
        elif paradigm == 'event':
            # Example: events at 5, 15, 25, 35 seconds
            events = [5, 15, 25, 35, 45, 55]
            task_state = self.event_related_paradigm(self.time, events)
        else:  # resting state
            task_state = 1.0  # Always active

        # Update stimulus history
        self.stimulus_history.append(task_state)

        # Keep only recent history (for HRF convolution)
        if dt > 0:
            max_samples = int(self.max_hrf_duration / dt)
            if len(self.stimulus_history) > max_samples:
                self.stimulus_history.pop(0)

        # Compute HRF response
        if len(self.stimulus_history) > 10:  # Need some history
            stimulus_array = np.array(self.stimulus_history)
            bold_response = self.convolve_with_hrf(stimulus_array)
            current_amplitude = bold_response[-1]
        else:
            current_amplitude = 0.0

        # Composite signal across all ROIs
        total_signal = np.zeros(self.n_vertices)
        for roi in self.rois:
            roi_signal = roi['baseline'] + current_amplitude * roi['amplitude'] * roi['weights']
            total_signal = np.maximum(total_signal, roi_signal)

        # Add subtle noise for realism
        noise = np.random.normal(0, 0.01, self.n_vertices)
        total_signal += noise

        return np.clip(total_signal, 0, 1)


def main():
    """Advanced multi-ROI simulation with canonical HRF."""

    print("=" * 70)
    print("Advanced fMRI BOLD Simulation: Multi-ROI + Canonical HRF")
    print("=" * 70)

    # Load mesh
    print("\nLoading cortical surface...")
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
    vertices, faces = load_surf_mesh(fsaverage['pial_left'])

    faces_vtk = np.hstack([
        np.full((faces.shape[0], 1), 3),
        faces
    ]).astype(np.int64)

    mesh = pv.PolyData(vertices, faces_vtk)
    print(f"Loaded: {mesh.n_points} vertices")

    # Initialize multi-ROI simulator
    print("\nConfiguring ROIs...")
    simulator = MultiROISimulator(mesh)

    # Add multiple ROIs representing a functional network
    simulator.add_roi(
        center_vertex=5000,
        radius=18,
        label='Motor Cortex'
    )
    simulator.add_roi(
        center_vertex=3500,
        radius=15,
        label='Premotor Cortex'
    )
    simulator.add_roi(
        center_vertex=7200,
        radius=20,
        label='Supplementary Motor Area'
    )

    # Create plotter
    print("\nInitializing renderer...")
    plotter = pv.Plotter(window_size=[1920, 1080])
    plotter.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0.0)
    plotter.background_color = 'black'

    # Enhanced lighting
    plotter.add_light(pv.Light(position=(100, 100, 100), intensity=0.7))
    plotter.add_light(pv.Light(position=(-100, -50, 100), intensity=0.4))
    plotter.add_light(pv.Light(position=(0, -100, -100), intensity=0.3))

    # Add mesh
    initial_scalars = simulator.update(0, paradigm='block')
    mesh['bold_signal'] = initial_scalars

    plotter.add_mesh(
        mesh,
        scalars='bold_signal',
        cmap= 'coolwarm',
        opacity='linear',        # <--- FIXED: Use simple linear mapping
        clim=[0, 0.9],           # Range: 0 (Invisible) to 0.9 (Max Brightness)
        smooth_shading=True,
        specular=1.0,            # High shininess for "Glass" look
        specular_power=15,       # Tight highlights
        ambient=0.3,             # <--- FIXED: Material property, not Light object
        show_scalar_bar=True,
        scalar_bar_args={
            'title': 'BOLD Activity',
            'color': 'white',
            'position_x': 0.85,
            'position_y': 0.1,
            'width': 0.1,
            'height': 0.5
        }
    )

    # Camera setup
    plotter.camera_position = [
        (110, 40, 140),
        (0, 5, 20),
        (0, 1, 0)
    ]

    # Add text annotations
    plotter.add_text(
        "Multi-ROI BOLD Simulation\nCanonical HRF | Block Design",
        position='upper_left',
        font_size=12,
        color='white'
    )

    # Performance tracking
    last_time = time.time()
    frame_count = 0
    fps_history = []

    def callback(step):
        nonlocal last_time, frame_count, fps_history

        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        # Update simulation
        new_scalars = simulator.update(dt, paradigm='block')
        mesh['bold_signal'] = new_scalars

        # FPS tracking
        if dt > 0:
            fps = 1.0 / dt
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)

        frame_count += 1

        # Display info
        if frame_count % 15 == 0:
            avg_fps = np.mean(fps_history) if fps_history else 0
            max_activation = new_scalars.max()
            print(f"Frame: {frame_count:5d} | "
                  f"FPS: {avg_fps:5.1f} | "
                  f"Time: {simulator.time:6.2f}s | "
                  f"Peak: {max_activation:.4f} | "
                  f"Task: {'ON ' if simulator.block_design_paradigm(simulator.time, 10, 8) > 0.5 else 'OFF'}",
                  end='    \r')

    print("\n" + "=" * 70)
    print("Starting simulation with canonical HRF and block design...")
    print("Press 'q' to quit")
    print("=" * 70 + "\n")

    plotter.add_timer_event(max_steps=1_000_000, duration=16,
                            callback=callback)  # ~60 FPS (16 ms)
    plotter.show()

    print("\n\nSimulation complete!")


if __name__ == "__main__":
    main()
