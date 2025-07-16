#!/usr/bin/env python

import os
from pathlib import Path
import pickle
from ewokscore import Task
import importlib
import math
# because I do not want to rename the files for now…
get_facets = importlib.import_module("facets.01_get_facets")
get_orientation = importlib.import_module("facets.02_get_orientation")
analyse = importlib.import_module("facets.03_analyse")

import vtk
from IPython.display import Image, display

def pickle_to_rick(path):
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

def rick_to_pickle(pipeline, dump_dir, filename):
    output_path = os.path.join(dump_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump(pipeline, f)
    return output_path

# ../src/01_get_facets.py sample_0001_scan487.vti 487 --visualize
class GetFacets(Task, input_names=["scratch_dirpath", "vti_filepath"], output_names=["vtp_facets_filepaths", "obj_filepath"]):
    def run(self):
        vti_filepath = self.inputs.vti_filepath
        scratch_dirpath = self.inputs.scratch_dirpath
        os.makedirs(scratch_dirpath, exist_ok=True)
        os.chdir(scratch_dirpath)
        arguments = [f'{vti_filepath}', 'exp', '--create-obj-0', 'exp.obj']
        get_facets.main(arguments)
        self.outputs.obj_filepath = str(Path(scratch_dirpath) / f"exp.obj")
        print(self.outputs.obj_filepath)
        self.outputs.vtp_facets_filepaths = str(Path(scratch_dirpath) / f"exp_1.vtp")

class GetOrientation(Task, input_names=["scratch_dirpath", "vtp_facets_filepath"], output_names=["orientation_filepath", "nsparam_filepath"]):
    def run(self):
        vtp_facets_filepath = self.inputs.vtp_facets_filepath
        scratch_dirpath = self.inputs.scratch_dirpath
        os.makedirs(scratch_dirpath, exist_ok=True)
        os.chdir(scratch_dirpath)
        arguments = [vtp_facets_filepath, "--axis", "Y",
                     "--hkl", "111", "--exclude_hkl", "1,1,1;-1,-1,-1",
                     "--save-orientation", "orientation.txt",
                     "--create-nsparam", "exp.nsparam"]
        get_orientation.main(arguments)
        self.outputs.nsparam_filepath = str(Path(scratch_dirpath) / "exp.nsparam")
        self.outputs.orientation_filepath = str(Path(scratch_dirpath) / "orientation.txt")
        print(self.outputs.nsparam_filepath)

class GetOrientedFacets(Task, input_names=["scratch_dirpath", "vti_filepath", "orientation_filepath"], output_names=["vtp_oriented_files"]):
    def run(self):
        vti_filepath = self.inputs.vti_filepath
        orientation_filepath = self.inputs.orientation_filepath
        scratch_dirpath = self.inputs.scratch_dirpath
        os.makedirs(scratch_dirpath, exist_ok=True)
        os.chdir(scratch_dirpath)
        arguments = [vti_filepath, "oriented_exp", "--orientation-matrix", orientation_filepath, "--visualize"]
        get_facets.main(arguments)
        self.outputs.vtp_oriented_files = [str(Path(scratch_dirpath) / f'oriented_exp_{i}.vtp') for i in range(0, 4)]

class FullCircleAnalysis(Task, input_names=["scratch_dirpath"], optional_input_names=["vti_filepath", "xyz_filepath"]):
    def run(self):
        scratch_dirpath = self.inputs.scratch_dirpath
        vti_filepath = self.get_input_value("vti_filepath", None)
        xyz_filepath = self.get_input_value("xyz_filepath", None)
        os.makedirs(scratch_dirpath, exist_ok=True)
        os.chdir(scratch_dirpath)
        if vti_filepath is not None:
            arguments = ["--hkl", "0", "2", "0", "--exp-qnorm", "3.2",
                         #"--exp-voxel-size", "5", "5", "5",
                         #"--exp-amp-key", "amp",
                         #"--final-shape", "64", "64", "64",
                         "--final-shape", "64", "64", "64",
                         "--phase-range", f"{math.pi/12}",
                         "--strain-range", "5e-4",
                         "--exp-data", vti_filepath]

        if xyz_filepath is not None:
            arguments = ["--hkl", "0", "2", "0",
                         "--nstep", "800",
                         "--strain-range", "5e-4",          # remove to match Corentin’s
                         "--phase-range", f"{math.pi/12}",  # remove to match Corentin’s
                         "--input-file", xyz_filepath]

        analyse.main(arguments)

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
