# Created by Bastien Rigaud at 21/01/2022
# Bastien Rigaud, PhD
# Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099
# Campus de Beaulieu, Université de Rennes 1
# 35042 Rennes, FRANCE
# bastien.rigaud@univ-rennes1.fr
# Description:

import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import pyvista as pv
import pyacvd
import math


class DataReaderWriter(object):
    def __init__(self, filepath):
        self.filepath = filepath

    def import_data(self):
        return

    def export_data(self):
        return


class ImageReaderWriter(DataReaderWriter):
    def __init__(self, filepath, image=None, cast_float32=False):
        super().__init__(filepath)
        self.image = image
        self.cast_float32 = cast_float32

    def import_data(self):
        img_pointer = sitk.ReadImage(self.filepath)
        if self.cast_float32:
            cast_filter = sitk.CastImageFilter()
            cast_filter.SetNumberOfThreads(0)
            cast_filter.SetOutputPixelType(sitk.sitkFloat32)
            img_pointer = cast_filter.Execute(img_pointer)
        return img_pointer

    def export_data(self):
        sitk.WriteImage(self.image, self.filepath)


class PolydataReaderWriter(DataReaderWriter):
    def __init__(self, filepath, polydata=None):
        super().__init__(filepath)
        self.polydata = polydata
        self.reader = None
        self.writer = None
        self.init_reader()
        if polydata:
            self.init_writer()
        self.set_filepath(filepath)

    def init_reader(self):
        if '.stl' in self.filepath:
            self.reader = vtk.vtkSTLReader()
        elif '.vtk' in self.filepath:
            self.reader = vtk.vtkPolyDataReader()
            self.reader.ReadAllScalarsOn()
            self.reader.ReadAllVectorsOn()
        else:
            raise ValueError('File extension not supported')

    def init_writer(self):
        if '.stl' in self.filepath:
            self.writer = vtk.vtkSTLWriter()
        elif '.vtk' in self.filepath:
            self.writer = vtk.vtkPolyDataWriter()
        else:
            raise ValueError('File extension not supported')
        self.writer.SetInputData(self.polydata)

    def set_filepath(self, filepath):
        self.reader.SetFileName(filepath)
        self.writer.SetFileName(filepath)

    def import_data(self):
        self.reader.Update()
        return self.reader.GetOutput()

    def export_data(self):
        if self.polydata:
            self.writer.Write()


class DataConverter(object):
    def __init__(self, polydata=None, image=None, nb_points=None, spacing=None, inval=1, outval=0, cast_float32=True):
        self.polydata = polydata
        self.spacing = spacing if spacing else (1.0, 1.0, 1.0)
        if image is not None:
            self.image = image
            self.numpy_array = sitk.GetArrayFromImage(image)
            self.size = list(image.GetSize())
            self.origin = list(image.GetOrigin())
            self.spacing = list(image.GetSpacing())
        self.nb_points = nb_points
        self.inval = inval
        self.outval = outval
        self.cast_float32 = cast_float32

    def mask_to_polydata(self):
        label = numpy_support.numpy_to_vtk(num_array=self.numpy_array.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        # Convert the VTK array to vtkImageData
        img_vtk = vtk.vtkImageData()
        img_vtk.SetDimensions(self.size)
        img_vtk.SetSpacing(self.spacing)
        img_vtk.SetOrigin(self.origin)
        img_vtk.GetPointData().SetScalars(label)

        MarchingCubeFilter = vtk.vtkDiscreteMarchingCubes()
        MarchingCubeFilter.SetInputData(img_vtk)
        MarchingCubeFilter.GenerateValues(1, 1, 1)
        MarchingCubeFilter.Update()

        if self.nb_points:
            # wrapper vtk polydata to pyvista polydata
            pv_temp = pv.PolyData(MarchingCubeFilter.GetOutput())
            cluster = pyacvd.Clustering(pv_temp)
            cluster.cluster(int(self.nb_points))
            remesh = cluster.create_mesh()
            remesh_vtk = vtk.vtkPolyData()
            remesh_vtk.SetPoints(remesh.GetPoints())
            remesh_vtk.SetVerts(remesh.GetVerts())
            remesh_vtk.SetPolys(remesh.GetPolys())
            return remesh_vtk
        else:
            return MarchingCubeFilter.GetOutput()

    def polydata_to_mask(self):
        if not self.polydata:
            raise ValueError("Specify polydata")

        # compute dimensions
        bounds = self.polydata.GetBounds()
        dim = [0] * 3
        for i in range(3):
            dim[i] = int(math.ceil((bounds[i * 2 + 1] - bounds[i * 2]) / self.spacing[i])) + 1
            if dim[i] < 1:
                dim[i] = 1

        origin = [0] * 3
        # NOTE: I am not sure if we have to add some offset!
        origin[0] = bounds[0]  # + spacing[0] / 2
        origin[1] = bounds[2]  # + spacing[1] / 2
        origin[2] = bounds[4]  # + spacing[2] / 2

        # Convert the VTK array to vtkImageData
        whiteImage = vtk.vtkImageData()
        whiteImage.SetDimensions(dim)
        whiteImage.SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1)
        whiteImage.SetSpacing(self.spacing)
        whiteImage.SetOrigin(origin)
        whiteImage.GetPointData()
        whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        # fill the image with foreground voxels:
        count = whiteImage.GetNumberOfPoints()
        for i in range(count):
            whiteImage.GetPointData().GetScalars().SetTuple1(i, self.inval)

        # polygonal data -. image stencil:
        pol2stenc = vtk.vtkPolyDataToImageStencil()
        pol2stenc.SetTolerance(0)  # important if extruder.SetVector(0, 0, 1) !!!
        pol2stenc.SetInputData(self.polydata)
        pol2stenc.SetOutputOrigin(origin)
        pol2stenc.SetOutputSpacing(self.spacing)
        pol2stenc.SetOutputWholeExtent(whiteImage.GetExtent())
        pol2stenc.Update()

        # cut the corresponding white image and set the background:
        imgstenc = vtk.vtkImageStencil()
        imgstenc.SetInputData(whiteImage)
        imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
        imgstenc.ReverseStencilOff()
        imgstenc.SetBackgroundValue(self.outval)
        imgstenc.Update()

        # imgstenc.GetOutput().GetPointData().GetArray(0)
        np_array = numpy_support.vtk_to_numpy(imgstenc.GetOutput().GetPointData().GetScalars())
        sitk_img = sitk.GetImageFromArray(np_array.reshape(dim[2], dim[1], dim[0]))  # reversed dimension here
        sitk_img.SetSpacing(self.spacing)
        sitk_img.SetOrigin(origin)

        if self.cast_float32:
            cast_filter = sitk.CastImageFilter()
            cast_filter.SetNumberOfThreads(0)
            cast_filter.SetOutputPixelType(sitk.sitkFloat32)
            sitk_img = cast_filter.Execute(sitk_img)

        return sitk_img
