#!/usr/bin/env python3
"""
Brain Region Explorer
=====================
Interactive tool to identify vertex indices for different cortical regions.

This utility helps you find the correct vertex indices for placing ROIs
in specific anatomical locations (motor cortex, visual cortex, etc.)

Author: Claude
Date: February 2026
"""

import numpy as np
import pyvista as pv
from nilearn import datasets
from nilearn.surface import load_surf_mesh


class BrainRegionExplorer:
    """Interactive tool for exploring cortical surface regions."""
    
    def __init__(self):
        self.mesh = None
        self.plotter = None
        self.selected_vertex = None
        self.selected_point = None
        
    def load_mesh(self, hemisphere='left', resolution='fsaverage5'):
        """Load cortical surface mesh."""
        print(f"Loading {hemisphere} hemisphere ({resolution})...")
        
        fsaverage = datasets.fetch_surf_fsaverage(mesh=resolution)
        
        mesh_file = fsaverage[f'pial_{hemisphere}']
        vertices, faces = load_surf_mesh(mesh_file)
        
        # Convert to PyVista format
        faces_vtk = np.hstack([
            np.full((faces.shape[0], 1), 3),
            faces
        ]).astype(np.int64)
        
        self.mesh = pv.PolyData(vertices, faces_vtk)
        self.mesh['vertex_id'] = np.arange(self.mesh.n_points)
        
        print(f"Mesh loaded: {self.mesh.n_points} vertices")
        
    def get_anatomical_landmarks(self):
        """
        Return approximate vertex indices for major anatomical regions.
        
        Note: These are approximate and may vary slightly based on
        fsaverage version and individual brain anatomy.
        """
        landmarks = {
            'Motor Cortex (Precentral Gyrus)': 5000,
            'Premotor Cortex': 3500,
            'Primary Somatosensory Cortex': 5800,
            'Supplementary Motor Area': 7200,
            'Visual Cortex (V1, Occipital)': 8000,
            'Auditory Cortex (Superior Temporal)': 6500,
            'Prefrontal Cortex (DLPFC)': 2500,
            'Broca\'s Area (Inferior Frontal)': 4200,
            'Angular Gyrus (Parietal)': 7800,
            'Fusiform Gyrus (Temporal)': 6800,
        }
        return landmarks
    
    def visualize_landmarks(self):
        """Create visualization showing anatomical landmark locations."""
        if self.mesh is None:
            self.load_mesh()
        
        # Create plotter
        self.plotter = pv.Plotter(window_size=[1600, 900])
        self.plotter.background_color = 'white'
        
        # Add base mesh (semi-transparent)
        self.plotter.add_mesh(
            self.mesh,
            color='lightgray',
            opacity=0.6,
            show_edges=False,
            smooth_shading=True
        )
        
        # Get landmarks
        landmarks = self.get_anatomical_landmarks()
        
        # Color each landmark region
        print("\nAnatomical Landmarks:")
        print("-" * 60)
        
        colors = ['red', 'blue', 'green', 'yellow', 'orange', 
                  'purple', 'cyan', 'magenta', 'brown', 'pink']
        
        for idx, (name, vertex_id) in enumerate(landmarks.items()):
            if vertex_id >= self.mesh.n_points:
                continue
            
            # Create sphere at landmark location
            position = self.mesh.points[vertex_id]
            sphere = pv.Sphere(radius=3, center=position)
            
            color = colors[idx % len(colors)]
            self.plotter.add_mesh(sphere, color=color, label=name)
            
            # Add text label
            self.plotter.add_point_labels(
                [position],
                [f"{name}\nVertex: {vertex_id}"],
                font_size=8,
                point_color=color,
                point_size=5,
                render_points_as_spheres=True,
                always_visible=False
            )
            
            print(f"{name:40s}: Vertex {vertex_id:6d}")
        
        print("-" * 60)
        
        # Add instructions
        self.plotter.add_text(
            "Anatomical Landmark Explorer\n"
            "Colored spheres show major cortical regions\n"
            "Hover over spheres to see labels",
            position='upper_left',
            font_size=12
        )
        
        # Set camera
        self.plotter.camera_position = [
            (120, 60, 150),
            (0, 0, 20),
            (0, 1, 0)
        ]
        
        self.plotter.add_legend(size=(0.3, 0.3))
        self.plotter.show()
    
    def interactive_selector(self):
        """
        Launch interactive tool to click on mesh and get vertex indices.
        """
        if self.mesh is None:
            self.load_mesh()
        
        self.plotter = pv.Plotter(window_size=[1600, 900])
        self.plotter.background_color = 'black'
        
        # Add mesh with vertex IDs as scalars
        self.plotter.add_mesh(
            self.mesh,
            scalars='vertex_id',
            cmap='viridis',
            opacity=0.8,
            show_scalar_bar=True,
            scalar_bar_args={
                'title': 'Vertex Index',
                'position_x': 0.85,
                'position_y': 0.25
            },
            smooth_shading=True
        )
        
        # Add instructions
        instruction_text = (
            "Interactive Vertex Selector\n"
            "===========================\n"
            "Click on the mesh to select a vertex\n"
            "Vertex index and coordinates will be printed\n\n"
            "Controls:\n"
            "  - Left Click: Select vertex\n"
            "  - Mouse Drag: Rotate view\n"
            "  - Scroll: Zoom\n"
            "  - 'r': Reset camera\n"
            "  - 'q': Quit\n"
        )
        
        self.plotter.add_text(
            instruction_text,
            position='upper_left',
            font_size=10,
            color='white'
        )
        
        # Enable point picking
        def callback(mesh, point_id):
            """Handle point selection."""
            if point_id < 0:
                return
            
            self.selected_vertex = point_id
            self.selected_point = self.mesh.points[point_id]
            
            # Remove previous selection marker
            for actor_name in list(self.plotter.actors.keys()):
                if 'selected_marker' in str(actor_name):
                    self.plotter.remove_actor(actor_name)
            
            # Add selection marker
            marker = pv.Sphere(radius=3, center=self.selected_point)
            self.plotter.add_mesh(
                marker, 
                color='red',
                name='selected_marker',
                reset_camera=False
            )
            
            # Print info
            print("\n" + "=" * 60)
            print(f"Selected Vertex: {point_id}")
            print(f"Coordinates: ({self.selected_point[0]:.2f}, "
                  f"{self.selected_point[1]:.2f}, "
                  f"{self.selected_point[2]:.2f})")
            print("-" * 60)
            print("Add this to your simulation:")
            print(f"  simulator.add_roi(center_vertex={point_id}, "
                  f"radius=20.0, label='My ROI')")
            print("=" * 60 + "\n")
        
        self.plotter.enable_point_picking(
            callback=callback,
            show_message=False,
            picker='cell',
            use_mesh=True
        )
        
        # Set camera
        self.plotter.camera_position = [
            (100, 50, 150),
            (0, 0, 20),
            (0, 1, 0)
        ]
        
        self.plotter.show()
    
    def export_coordinates(self, filename='roi_coordinates.txt'):
        """Export selected coordinates to file."""
        if self.selected_vertex is None:
            print("No vertex selected. Use interactive_selector() first.")
            return
        
        with open(filename, 'w') as f:
            f.write(f"Vertex Index: {self.selected_vertex}\n")
            f.write(f"Coordinates: {self.selected_point}\n")
            f.write(f"\n# Code snippet:\n")
            f.write(f"simulator.add_roi(\n")
            f.write(f"    center_vertex={self.selected_vertex},\n")
            f.write(f"    radius=20.0,\n")
            f.write(f"    label='Custom ROI'\n")
            f.write(f")\n")
        
        print(f"Coordinates exported to {filename}")


def print_menu():
    """Print main menu."""
    print("\n" + "=" * 60)
    print("Brain Region Explorer")
    print("=" * 60)
    print("\nOptions:")
    print("  1. Show anatomical landmarks")
    print("  2. Interactive vertex selector")
    print("  3. Print landmark coordinates")
    print("  4. Quit")
    print("\nEnter your choice (1-4): ", end="")


def main():
    """Main application loop."""
    explorer = BrainRegionExplorer()
    
    while True:
        print_menu()
        choice = input().strip()
        
        if choice == '1':
            print("\nLaunching anatomical landmark viewer...")
            explorer.visualize_landmarks()
        
        elif choice == '2':
            print("\nLaunching interactive vertex selector...")
            print("Click on the brain to select vertices!")
            explorer.interactive_selector()
        
        elif choice == '3':
            print("\n" + "=" * 60)
            print("Anatomical Landmark Coordinates")
            print("=" * 60)
            landmarks = explorer.get_anatomical_landmarks()
            
            if explorer.mesh is None:
                explorer.load_mesh()
            
            print(f"\n{'Region':<40s} {'Vertex':<10s} {'Coordinates'}")
            print("-" * 60)
            
            for name, vertex_id in landmarks.items():
                if vertex_id < explorer.mesh.n_points:
                    pos = explorer.mesh.points[vertex_id]
                    print(f"{name:<40s} {vertex_id:<10d} "
                          f"({pos[0]:6.1f}, {pos[1]:6.1f}, {pos[2]:6.1f})")
            
            print("-" * 60)
            input("\nPress Enter to continue...")
        
        elif choice == '4':
            print("\nExiting Brain Region Explorer. Goodbye!")
            break
        
        else:
            print("\nInvalid choice. Please enter 1-4.")


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║           Brain Region Explorer v1.0                       ║
    ║     Interactive tool for fMRI ROI vertex selection         ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    main()
