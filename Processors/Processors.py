# Created by Bastien Rigaud at 21/01/2022
# Bastien Rigaud, PhD
# Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099
# Campus de Beaulieu, Université de Rennes 1
# 35042 Rennes, FRANCE
# bastien.rigaud@univ-rennes1.fr
# Description:

import math
import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import pyvista as pv
import pyacvd

from IOTools.IOTools import DataConverter
from PlotVTK.PlotVTK import plot_vtk


def _check_keys_(input_features, keys):
    if type(keys) is list or type(keys) is tuple:
        for key in keys:
            assert key in input_features.keys(), 'Make sure the key you are referring to is present in the features, ' \
                                                 '{} was not found'.format(key)
    else:
        assert keys in input_features.keys(), 'Make sure the key you are referring to is present in the features, ' \
                                              '{} was not found'.format(keys)


def get_polydata_centroid(polydata):
    center_filter = vtk.vtkCenterOfMass()
    center_filter.SetInputData(polydata)
    center_filter.SetUseScalarsAsWeights(False)
    center_filter.Update()
    centroid = center_filter.GetCenter()
    return centroid


def translate_polydata(polydata, translation=(0., 0., 0.)):
    transform = vtk.vtkTransform()
    transform.Translate(translation[0], translation[1], translation[2])
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(polydata)
    transform_filter.Update()
    return transform_filter.GetOutput()


class Processor(object):
    def pre_process(self, input_features):
        return input_features

    def post_process(self, input_features):
        return input_features


class ACVDResampling(Processor):
    def __init__(self, input_keys=('input_name',), output_keys=('output_name',), np_points=(5000,)):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.np_points = np_points

    def compute_acvd(self, polydata, nb_points):
        pv_temp = pv.PolyData(polydata)
        cluster = pyacvd.Clustering(pv_temp)
        cluster.cluster(int(nb_points))
        remesh = cluster.create_mesh()
        remesh_vtk = vtk.vtkPolyData()
        remesh_vtk.SetPoints(remesh.GetPoints())
        remesh_vtk.SetVerts(remesh.GetVerts())
        remesh_vtk.SetPolys(remesh.GetPolys())
        return remesh_vtk

    def pre_process(self, input_features):
        _check_keys_(input_features, self.input_keys)
        for input_key, output_key, np_point in zip(self.input_keys, self.output_keys, self.np_points):
            input_features[output_key] = self.compute_acvd(input_features[input_key], np_point)
        return input_features


class ConvertMaskToPoly(Processor):
    def __init__(self, input_keys=('xmask', 'ymask'), output_keys=('xpoly', 'ypoly')):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.converter = DataConverter()

    def pre_process(self, input_features):
        _check_keys_(input_features, self.input_keys)
        for input_key, output_key in zip(self.input_keys, self.output_keys):
            image = input_features[input_key]
            converter = DataConverter(image=image, inval=1, outval=0, cast_float32=True)
            input_features[output_key] = converter.mask_to_polydata()
        return input_features


class GetSITKInfo(Processor):
    def __init__(self, input_keys=('xmask', 'ymask')):
        self.input_keys = input_keys

    def pre_process(self, input_features):
        _check_keys_(input_features, self.input_keys)
        for input_key in self.input_keys:
            img_pointer = input_features[input_key]
            input_features[input_key + '_spacing'] = np.array(list(img_pointer.GetSpacing()))
            input_features[input_key + '_origin'] = np.array(list(img_pointer.GetOrigin()))
        return input_features


class SITKToNumpy(Processor):
    def __init__(self, input_keys=('xmask', 'ymask'), output_keys=('xmask', 'ymask')):
        self.input_keys = input_keys
        self.output_keys = output_keys

    def pre_process(self, input_features):
        _check_keys_(input_features, self.input_keys)
        for input_key, output_key in zip(self.input_keys, self.output_keys):
            img_pointer = input_features[input_key]
            input_features[output_key] = sitk.GetArrayFromImage(img_pointer)
        return input_features


class ZNormPoly(Processor):
    def __init__(self, input_keys=('xpoly', 'ypoly'), output_keys=('xpoly', 'ypoly'),
                 centroid_keys=('centroid', 'centroid',), scale_keys=('scale', 'scale')):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.centroid_keys = centroid_keys
        self.scale_keys = scale_keys

    def get_scale(self, polydata, centroid):
        '''
        :param polydata: vtk polydata
        :param centroid: polydata centroid as tuple
        :return: return squared root of the summed euclidean distance between points and centroid divided by number of points
        '''
        points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
        scale = math.sqrt(np.sum(np.sum(np.square(points - centroid), axis=-1)) / points.shape[0])
        return scale

    def apply_z_norm(self, polydata, centroid, scale):
        point_data = polydata.GetPoints().GetData()
        np_points = numpy_support.vtk_to_numpy(point_data)
        scaled_np_points = (np_points - centroid) / scale
        scaled_point_data = numpy_support.numpy_to_vtk(scaled_np_points)
        output = vtk.vtkPolyData()
        output.DeepCopy(polydata)
        output.GetPoints().SetData(scaled_point_data)
        return output

    def unapply_z_norm(self, polydata, centroid, scale):
        point_data = polydata.GetPoints().GetData()
        np_points = numpy_support.vtk_to_numpy(point_data)
        unscaled_np_points = (np_points * scale) + centroid
        unscaled_point_data = numpy_support.numpy_to_vtk(unscaled_np_points)
        output = vtk.vtkPolyData()
        output.DeepCopy(polydata)
        output.GetPoints().SetData(unscaled_point_data)
        return output

    def pre_process(self, input_features):
        _check_keys_(input_features, self.input_keys)
        for input_key, output_key, centroid_key, scale_key in zip(self.input_keys, self.output_keys, self.centroid_keys,
                                                                  self.scale_keys):
            polydata = input_features[input_key]
            centroid = get_polydata_centroid(polydata)
            scale = self.get_scale(polydata, centroid)
            input_features[output_key] = self.apply_z_norm(polydata, centroid, scale)
            input_features[output_key + '_' + centroid_key] = centroid
            input_features[output_key + '_' + scale_key] = scale
        return input_features

    def post_process(self, input_features):
        _check_keys_(input_features, self.input_keys)
        for input_key, output_key, centroid_key, scale_key in zip(self.input_keys, self.output_keys, self.centroid_keys,
                                                                  self.scale_keys):
            polydata = input_features[input_key]
            centroid = input_features[input_key + '_' + centroid_key]
            scale = input_features[input_key + '_' + scale_key]
            input_features[output_key] = self.unapply_z_norm(polydata, centroid, scale)
        return input_features


class AlignCentroid(Processor):
    def __init__(self, fixed_keys=('xpoly',), moving_keys=('ypoly',), output_keys=('aligned_ypoly',),
                 run_post_process=False):
        self.fixed_keys = fixed_keys
        self.moving_keys = moving_keys
        self.output_keys = output_keys
        self.run_post_process = run_post_process

    def pre_process(self, input_features):
        _check_keys_(input_features, self.fixed_keys + self.moving_keys)
        for fixed_key, moving_key, output_key in zip(self.fixed_keys, self.moving_keys, self.output_keys):
            fcentroid = get_polydata_centroid(input_features[fixed_key])
            mcentroid = get_polydata_centroid(input_features[moving_key])
            translation = tuple(np.subtract(fcentroid, mcentroid))
            input_features[output_key] = translate_polydata(input_features[moving_key], translation=translation)
            input_features[output_key + '_translation'] = translation
        return input_features

    def post_process(self, input_features):
        if self.run_post_process:
            for output_key in self.output_keys:
                translation = tuple([-trans for trans in input_features[output_key + '_translation']])
                input_features[output_key] = translate_polydata(input_features[output_key], translation=translation)
        return input_features
