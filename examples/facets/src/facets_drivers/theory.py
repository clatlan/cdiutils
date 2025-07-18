#!/usr/bin/env python

import os
from pathlib import Path
from ewokscore import Task
import facets.get_facets as get_facets
import facets.get_orientation as get_orientation

import vtk
from IPython.display import Image, display

class GetFacets(Task, input_names=["scratch_directory", "vti_filepath"], output_names=["vtp_facets_filepath", "obj_filepath", "scratch_directory"]):
    def run(self):
        vti_filepath = self.inputs.vti_filepath
        scratch_directory = self.inputs.scratch_directory
        os.makedirs(scratch_directory, exist_ok=True)
        os.chdir(scratch_directory)
        arguments = [f'{vti_filepath}', 'exp', '--create-obj-0', 'exp.obj']
        get_facets.main(arguments)
        self.outputs.obj_filepath = str(Path(scratch_directory) / f"exp.obj")
        print(self.outputs.obj_filepath)
        self.outputs.vtp_facets_filepath = str(Path(scratch_directory) / f"exp_1.vtp")
        self.outputs.scratch_directory = scratch_directory

class GetOrientation(Task, input_names=["scratch_directory", "vtp_facets_filepath"], output_names=["orientation_filepath", "nsparam_filepath", "scratch_directory", "vtp_facets_filepath"]):
    def run(self):
        vtp_facets_filepath = self.inputs.vtp_facets_filepath
        scratch_directory = self.inputs.scratch_directory
        os.makedirs(scratch_directory, exist_ok=True)
        os.chdir(scratch_directory)
        arguments = [vtp_facets_filepath, "--axis", "Y",
                     "--hkl", "111", "--exclude_hkl", "1,1,1;-1,-1,-1",
                     "--save-orientation", "orientation.txt",
                     "--create-nsparam", "exp.nsparam"]
        get_orientation.main(arguments)
        self.outputs.nsparam_filepath = str(Path(scratch_directory) / "exp.nsparam")
        self.outputs.orientation_filepath = str(Path(scratch_directory) / "orientation.txt")
        self.outputs.scratch_directory = scratch_directory
        self.outputs.vtp_facets_filepath = vtp_facets_filepath
        print(self.outputs.nsparam_filepath)

class GetOrientedFacets(Task, input_names=["scratch_directory", "vtp_facets_filepath", "orientation_filepath"], output_names=["vtp_oriented_files", "scratch_directory"]):
    def run(self):
        vtp_facets_filepath = self.inputs.vtp_facets_filepath
        orientation_filepath = self.inputs.orientation_filepath
        scratch_directory = self.inputs.scratch_directory
        os.makedirs(scratch_directory, exist_ok=True)
        os.chdir(scratch_directory)
        arguments = [vtp_facets_filepath, "oriented_exp", "--relabel-from-hull", "--orientation-matrix", orientation_filepath, "--visualize"]
        get_facets.main(arguments)
        self.outputs.vtp_oriented_files = [str(Path(scratch_directory) / f'oriented_exp_{i}.vtp') for i in range(0, 4)]
        self.outputs.scratch_directory = scratch_directory

######## Monkey ################
# Use
from xvfbwrapper import Xvfb
vdisplay = Xvfb()
vdisplay.start()
# ...
# vdisplay.stop()
# around call in notebook
# Monkey patching.
def non_blocking_visualization(hull_polydata, labels_polydata):
    """
    A non-blocking replacement that renders the scene to a PNG file
    and displays it in the notebook.
    """
    # --- Basic VTK pipeline setup ---
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.1, 0.2, 0.4)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1200, 900)

    # IMPORTANT: Ensure off-screen rendering is enabled
    render_window.SetOffScreenRendering(1)

    # --- Your existing actor and mapper setup code ---
    # (Copied from your script for completeness)
    hull_mapper = vtk.vtkPolyDataMapper()
    hull_mapper.SetInputData(hull_polydata)

    facet_ids_array = hull_polydata.GetCellData().GetArray("FacetIds")
    if facet_ids_array:
        lookup_table = vtk.vtkLookupTable()
        id_range = facet_ids_array.GetRange()
        lookup_table.SetNumberOfTableValues(int(id_range[1]) + 1)
        lookup_table.Build()
        hull_mapper.SetScalarModeToUseCellFieldData()
        hull_mapper.SelectColorArray("FacetIds")
        hull_mapper.SetScalarVisibility(True)
        hull_mapper.SetLookupTable(lookup_table)
        hull_mapper.SetScalarRange(id_range)

    hull_actor = vtk.vtkActor()
    hull_actor.SetMapper(hull_mapper)
    hull_actor.GetProperty().EdgeVisibilityOn()

    select_visible_points = vtk.vtkSelectVisiblePoints()
    select_visible_points.SetInputData(labels_polydata)
    select_visible_points.SetRenderer(renderer)

    label_mapper = vtk.vtkLabeledDataMapper()
    label_mapper.SetInputConnection(select_visible_points.GetOutputPort())
    label_mapper.SetLabelModeToLabelFieldData()
    label_mapper.SetFieldDataName("MillerLabels")

    label_actor = vtk.vtkActor2D()
    label_actor.SetMapper(label_mapper)

    # --- Add actors, render, and save the image ---
    renderer.AddActor(hull_actor)
    renderer.AddActor(label_actor)
    renderer.ResetCamera()
    render_window.Render()

    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.Update()

    writer = vtk.vtkPNGWriter()
    output_filename = "visualization_patched.png"
    writer.SetFileName(output_filename)
    writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    writer.Write()

    print(f"Visualization saved via monkey-patch to {output_filename}. Displaying below:")
    display(Image(filename=output_filename))
get_facets.create_and_show_visualization = non_blocking_visualization

#  def non_blocking_visualization(hull_polydata, labels_polydata):
#      """
#      A non-blocking replacement that renders the scene to a PNG file
#      and displays it in the notebook.
#      """
#      # --- Basic VTK pipeline setup ---
#      renderer = vtk.vtkRenderer()
#      renderer.SetBackground(0.1, 0.2, 0.4)
#
#      render_window = vtk.vtkRenderWindow()
#      render_window.AddRenderer(renderer)
#      render_window.SetSize(1200, 900)
#
#      # IMPORTANT: Ensure off-screen rendering is enabled
#      render_window.SetOffScreenRendering(1)
#
#      # --- Your existing actor and mapper setup code ---
#      hull_mapper = vtk.vtkPolyDataMapper()
#      hull_mapper.SetInputData(hull_polydata)
#
#      facet_ids_array = hull_polydata.GetCellData().GetArray("FacetIds")
#      if facet_ids_array:
#          lookup_table = vtk.vtkLookupTable()
#          id_range = facet_ids_array.GetRange()
#          lookup_table.SetNumberOfTableValues(int(id_range[1]) + 1)
#          lookup_table.Build()
#          hull_mapper.SetScalarModeToUseCellFieldData()
#          hull_mapper.SelectColorArray("FacetIds")
#          hull_mapper.SetScalarVisibility(True)
#          hull_mapper.SetLookupTable(lookup_table)
#          hull_mapper.SetScalarRange(id_range)
#
#      hull_actor = vtk.vtkActor()
#      hull_actor.SetMapper(hull_mapper)
#      hull_actor.GetProperty().EdgeVisibilityOn()
#
#      select_visible_points = vtk.vtkSelectVisiblePoints()
#      select_visible_points.SetInputData(labels_polydata)
#      select_visible_points.SetRenderer(renderer)
#
#      label_mapper = vtk.vtkLabeledDataMapper()
#      label_mapper.SetInputConnection(select_visible_points.GetOutputPort())
#      label_mapper.SetLabelModeToLabelFieldData()
#      label_mapper.SetFieldDataName("MillerLabels")
#
#      label_actor = vtk.vtkActor2D()
#      label_actor.SetMapper(label_mapper)
#
#      # --- Add actors and setup camera for bottom view ---
#      renderer.AddActor(hull_actor)
#      renderer.AddActor(label_actor)
#
#      # Get the bounds of the object to position camera appropriately
#      bounds = hull_polydata.GetBounds()
#      center = [(bounds[0] + bounds[1]) / 2,
#                (bounds[2] + bounds[3]) / 2,
#                (bounds[4] + bounds[5]) / 2]
#
#      # Set up camera for bottom view (looking up from below)
#      camera = renderer.GetActiveCamera()
#
#      # Position camera below the object (negative Y direction)
#      camera_distance = (bounds[3] - bounds[2]) * 2  # Distance based on Y extent
#      camera.SetPosition(center[0], bounds[2] - camera_distance, center[2])
#
#      # Point camera toward the center of the object
#      camera.SetFocalPoint(center[0], center[1], center[2])
#
#      # Set the view up vector (Z-axis pointing up in this bottom view)
#      camera.SetViewUp(0, 0, 1)
#
#      # Optional: adjust the view angle for better visualization
#      camera.SetViewAngle(45)
#
#      renderer.ResetCameraClippingRange()
#      render_window.Render()
#
#      window_to_image_filter = vtk.vtkWindowToImageFilter()
#      window_to_image_filter.SetInput(render_window)
#      window_to_image_filter.Update()
#
#      writer = vtk.vtkPNGWriter()
#      output_filename = "visualization_patched.png"
#      writer.SetFileName(output_filename)
#      writer.SetInputConnection(window_to_image_filter.GetOutputPort())
#      writer.Write()
#
#      print(f"Visualization saved via monkey-patch to {output_filename}. Displaying below:")
#      display(Image(filename=output_filename))
#  get_facets.create_and_show_visualization = non_blocking_visualization
