# Created by Bastien Rigaud at 21/01/2022
# Bastien Rigaud, PhD
# Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099
# Campus de Beaulieu, Université de Rennes 1
# 35042 Rennes, FRANCE
# bastien.rigaud@univ-rennes1.fr
# Description:

import vtk
from vtk.util import numpy_support
import pyvista as pv
import pyacvd
import math

class ImageReaderWriter(object):
    def __init__(self, filepath, binary=False):
        xxx = 1



class PolydataReaderWriter(object):
    def __init__(self, filepath, polydata=None):
        self.filepath = filepath
        self.polydata = polydata
        self.reader = None
        self.writer = None

    def InitReader(self):
        if '.stl' in self.filepath:
            self.reader = vtk.vtkSTLReader()
        elif '.vtk' in self.filepath:
            self.reader = vtk.vtkPolyDataReader()
            self.reader.ReadAllScalarsOn()
            self.reader.ReadAllVectorsOn()
        else:
            raise ValueError('File extension not supported')

    def InitWriter(self):
        if '.stl' in self.filepath:
            self.writer = vtk.vtkSTLWriter()
        elif '.vtk' in self.filepath:
            self.writer = vtk.vtkPolyDataWriter()
        else:
            raise ValueError('File extension not supported')
        self.writer.SetInputData(self.polydata)

    def SetFilepath(self, filepath):
        self.reader.SetFileName(filepath)

    def ImportPolydata(self):
        self.reader.Update()
        return self.reader.GetOutput()

    def ExportPolydata(self):
        self.writer.Write()


class DataConverter(object):
    def __init__(self):

    def PolydataToMask(self):
        # compute dimensions
        bounds = polydata.GetBounds()
        dim = [0] * 3
        for i in range(3):
            dim[i] = int(math.ceil((bounds[i * 2 + 1] - bounds[i * 2]) / spacing[i])) + 1
            if dim[i] < 1:
                dim[i] = 1

        origin = [0] * 3
        # NOTE: I am not sure whether or not we had to add some offset!
        origin[0] = bounds[0]  # + spacing[0] / 2
        origin[1] = bounds[2]  # + spacing[1] / 2
        origin[2] = bounds[4]  # + spacing[2] / 2

        # Convert the VTK array to vtkImageData
        whiteImage = vtk.vtkImageData()
        whiteImage.SetDimensions(dim)
        whiteImage.SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1)
        whiteImage.SetSpacing(spacing)
        whiteImage.SetOrigin(origin)
        whiteImage.GetPointData()
        whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        # fill the image with foreground voxels:
        count = whiteImage.GetNumberOfPoints()
        for i in range(count):
            whiteImage.GetPointData().GetScalars().SetTuple1(i, inval)

        # polygonal data -. image stencil:
        pol2stenc = vtk.vtkPolyDataToImageStencil()
        pol2stenc.SetTolerance(0)  # important if extruder.SetVector(0, 0, 1) !!!
        pol2stenc.SetInputData(polydata)
        pol2stenc.SetOutputOrigin(origin)
        pol2stenc.SetOutputSpacing(spacing)
        pol2stenc.SetOutputWholeExtent(whiteImage.GetExtent())
        pol2stenc.Update()

        # cut the corresponding white image and set the background:
        imgstenc = vtk.vtkImageStencil()
        imgstenc.SetInputData(whiteImage)
        imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
        imgstenc.ReverseStencilOff()
        imgstenc.SetBackgroundValue(outval)
        imgstenc.Update()

        # imgstenc.GetOutput().GetPointData().GetArray(0)
        np_array = numpy_support.vtk_to_numpy(imgstenc.GetOutput().GetPointData().GetScalars())
        sitk_img = sitk.GetImageFromArray(np_array.reshape(dim[2], dim[1], dim[0]))  # reversed dimension here
        sitk_img.SetSpacing(spacing)
        sitk_img.SetOrigin(origin)

        if cast_float32:
            cast_filter = sitk.CastImageFilter()
            cast_filter.SetNumberOfThreads(0)
            cast_filter.SetOutputPixelType(sitk.sitkFloat32)
            sitk_img = cast_filter.Execute(sitk_img)

        return sitk_img

    def MaskToPolydata(self):
