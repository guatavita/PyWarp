# Created by Bastien Rigaud at 21/01/2022
# Bastien Rigaud, PhD
# Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099
# Campus de Beaulieu, Université de Rennes 1
# 35042 Rennes, FRANCE
# bastien.rigaud@univ-rennes1.fr
# Description:

import sys, os
import itertools
import math
import numpy as np
import SimpleITK as sitk
import skimage.morphology
import vtk
from vtk.util import numpy_support
import pyvista as pv
import pyacvd

from skimage import morphology
from scipy.ndimage import binary_opening, binary_closing, binary_erosion, binary_dilation

from threading import Thread
from multiprocessing import cpu_count
from queue import *

import collections
import networkx as nx

from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
from .Laplacian.Laplacian import Laplacian
from Image_Processors_Utils.Image_Processor_Utils import compute_binary_morphology, compute_bounding_box, \
    Remove_Smallest_Structures

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from IOTools.IOTools import DataConverter, PolydataReaderWriter
from PlotVTK.PlotVTK import plot_vtk


def _check_keys_(input_features, keys):
    if type(keys) is list or type(keys) is tuple:
        for key in keys:
            assert key in input_features.keys(), 'Make sure the key you are referring to is present in the features, ' \
                                                 '{} was not found'.format(key)
    else:
        assert keys in input_features.keys(), 'Make sure the key you are referring to is present in the features, ' \
                                              '{} was not found'.format(keys)


def get_polydata_scale(polydata, centroid):
    '''
    :param polydata: vtk polydata
    :param centroid: polydata centroid as tuple
    :return: return squared root of the summed euclidean distance between points and centroid divided by number of points
    '''
    points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
    scale = math.sqrt(np.sum(np.sum(np.square(points - centroid), axis=-1)) / points.shape[0])
    return scale


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


def threshold_polydata(polydata, lower_th, upper_th=None, return_selection=False):
    output = vtk.vtkPolyData()
    threshold_filter = vtk.vtkThresholdPoints()
    threshold_filter.SetInputData(polydata)
    if upper_th:
        threshold_filter.ThresholdBetween(lower_th, upper_th)
    else:
        threshold_filter.ThresholdBetween(lower_th, lower_th + 0.5)
    threshold_filter.Update()
    output.DeepCopy(threshold_filter.GetOutput())
    return output


def threshold_selection_polydata(polydata, value_condition=0):
    output = vtk.vtkPolyData()

    np_scalars = numpy_support.vtk_to_numpy(polydata.GetPointData().GetScalars())
    np_indices = np.flatnonzero(np_scalars == value_condition)
    extract_selection_filter = vtk.vtkExtractSelection()

    vtk_ids = vtk.vtkIdTypeArray()
    vtk_ids.SetNumberOfComponents(1)
    for i in range(len(np_indices)):
        vtk_ids.InsertNextValue(np_indices[i])

    selection_node = vtk.vtkSelectionNode()
    selection_node.SetFieldType(vtk.vtkSelectionNode.POINT)
    selection_node.SetContentType(vtk.vtkSelectionNode.INDICES)
    selection_node.SetSelectionList(vtk_ids)
    selection_node.GetProperties().Set(vtk.vtkSelectionNode.CONTAINING_CELLS(), 1)

    selection = vtk.vtkSelection()
    selection.AddNode(selection_node)

    extract_selection_filter.SetInputData(0, polydata)
    extract_selection_filter.SetInputData(1, selection)
    extract_selection_filter.Update()

    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputConnection(extract_selection_filter.GetOutputPort())
    geometry_filter.Update()
    output.DeepCopy(geometry_filter.GetOutput())

    return output


class Processor(object):
    def pre_process(self, input_features):
        return input_features

    def post_process(self, input_features):
        return input_features


class ACVDResampling(Processor):
    def __init__(self, input_keys=('input_name',), output_keys=('output_name',), nb_points=(5000,)):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.nb_points = nb_points

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
        for input_key, output_key, nb_point in zip(self.input_keys, self.output_keys, self.nb_points):
            input_features[output_key] = self.compute_acvd(input_features[input_key], nb_point)
        return input_features


class ConvertMaskToPoly(Processor):
    def __init__(self, input_keys=('xmask', 'ymask'), output_keys=('xpoly', 'ypoly'), fill_holes=True):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.converter = DataConverter()
        self.fill_holes = fill_holes

    def pre_process(self, input_features):
        _check_keys_(input_features, self.input_keys)
        for input_key, output_key in zip(self.input_keys, self.output_keys):
            image = input_features[input_key]
            converter = DataConverter(image=image, inval=1, outval=0, cast_float32=True)
            if self.fill_holes:
                fill_holes_filter = vtk.vtkFillHolesFilter()
                fill_holes_filter.SetInputData(converter.mask_to_polydata())
                fill_holes_filter.SetHoleSize(999999.9)
                fill_holes_filter.Update()
                input_features[output_key] = fill_holes_filter.GetOutput()
            else:
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


class GetZNormParameters(Processor):
    def __init__(self, input_keys=('xpoly', 'ypoly'), centroid_keys=('KEY_centroid', 'KEY_centroid',),
                 scale_keys=('KEY_scale', 'KEY_scale')):
        self.input_keys = input_keys
        self.centroid_keys = centroid_keys
        self.scale_keys = scale_keys

    def pre_process(self, input_features):
        _check_keys_(input_features, self.input_keys)
        for input_key, centroid_key, scale_key in zip(self.input_keys, self.centroid_keys, self.scale_keys):
            polydata = input_features[input_key]
            centroid = get_polydata_centroid(polydata)
            scale = get_polydata_scale(polydata, centroid)
            input_features[centroid_key] = centroid
            input_features[scale_key] = scale
        return input_features


class ZNormPoly(Processor):
    def __init__(self, input_keys=('xpoly', 'ypoly'), output_keys=('xpoly', 'ypoly'),
                 post_process_keys=(), centroid_keys=('KEY_centroid', 'KEY_centroid',),
                 scale_keys=('KEY_scale', 'KEY_scale')):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.post_process_keys = post_process_keys
        self.centroid_keys = centroid_keys
        self.scale_keys = scale_keys

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
            centroid = input_features[centroid_key]
            scale = input_features[scale_key]
            input_features[output_key] = self.apply_z_norm(polydata, centroid, scale)
        return input_features

    def post_process(self, input_features):
        _check_keys_(input_features, self.input_keys)
        for post_process_key, output_key, centroid_key, scale_key in zip(self.post_process_keys, self.output_keys,
                                                                         self.centroid_keys,
                                                                         self.scale_keys):
            polydata = input_features[post_process_key]
            centroid = input_features[centroid_key]
            scale = input_features[scale_key]
            input_features[output_key] = self.unapply_z_norm(polydata, centroid, scale)
        return input_features


class CopyKey(Processor):
    def __init__(self, input_keys=('xpoly_centroid', 'ypoly_centroid', 'xpoly_scale', 'ypoly_scale'),
                 output_keys=('xpoly_centroid', 'ypoly_centroid', 'xpoly_scale', 'ypoly_scale')):
        self.input_keys = input_keys
        self.output_keys = output_keys

    def pre_process(self, input_features):
        for input_key, output_key in zip(self.input_keys, self.output_keys):
            input_features[output_key] = input_features[input_key]
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


class CreateDVF(Processor):
    def __init__(self, reference_keys=('xpoly', 'ypoly',), deformed_keys=('ft_poly', 'bt_poly',),
                 output_keys=('ft_dvf', 'bt_dvf',), run_pre_process=False, run_post_process=False, set_scalars=True):
        self.reference_keys = reference_keys
        self.deformed_keys = deformed_keys
        self.output_keys = output_keys
        self.run_pre_process = run_pre_process
        self.run_post_process = run_post_process
        self.set_scalars = set_scalars

    def create_dvf(self, reference, deformed, set_scalars=True):
        if reference.GetNumberOfPoints() != deformed.GetNumberOfPoints():
            raise ValueError("Fixed and moving polydata must have same number of points")
        output = vtk.vtkPolyData()
        output.DeepCopy(reference)
        fixed_points = numpy_support.vtk_to_numpy(reference.GetPoints().GetData())
        moving_points = numpy_support.vtk_to_numpy(deformed.GetPoints().GetData())
        vector_field = numpy_support.numpy_to_vtk(moving_points - fixed_points)
        vector_field.SetName('VectorField')
        output.GetPointData().SetVectors(vector_field)
        if set_scalars:
            vector_magn = numpy_support.numpy_to_vtk(np.sqrt(np.sum(np.square(moving_points - fixed_points), axis=-1)))
            vector_magn.SetName('Magnitude')
            output.GetPointData().SetScalars(vector_magn)
        return output

    def pre_process(self, input_features):
        if self.run_pre_process:
            for reference_key, deformed_key, output_key in zip(self.reference_keys, self.deformed_keys,
                                                               self.output_keys):
                input_features[output_key] = self.create_dvf(input_features[reference_key],
                                                             input_features[deformed_key],
                                                             set_scalars=self.set_scalars)
        return input_features

    def post_process(self, input_features):
        if self.run_post_process:
            for reference_key, deformed_key, output_key in zip(self.reference_keys, self.deformed_keys,
                                                               self.output_keys):
                input_features[output_key] = self.create_dvf(input_features[reference_key],
                                                             input_features[deformed_key],
                                                             set_scalars=self.set_scalars)
        return input_features


class JoinPoly(Processor):
    def __init__(self, input_key_list=None, output_key='xpoly', use_scalar=True, scalar_name='label_scalar'):
        if input_key_list is None:
            input_key_list = []
        self.input_key_list = input_key_list
        self.output_key = output_key
        self.use_scalar = use_scalar
        self.scalar_name = scalar_name

    def pre_process(self, input_features):
        append_filter = vtk.vtkAppendPolyData()
        i = 0
        for poly_key in self.input_key_list:
            temp = vtk.vtkPolyData()
            temp.DeepCopy(input_features[poly_key])
            if self.use_scalar:
                array_names = [temp.GetPointData().GetArrayName(arrayid) for arrayid in
                               range(temp.GetPointData().GetNumberOfArrays())]
                if self.scalar_name not in array_names:
                    label_color = numpy_support.numpy_to_vtk(i * np.ones(temp.GetNumberOfPoints()))
                    label_color.SetName(self.scalar_name)
                    temp.GetPointData().AddArray(label_color)
                    temp.GetPointData().SetActiveScalars(self.scalar_name)
            append_filter.AddInputData(temp)
            i += 1
        append_filter.Update()
        input_features[self.output_key] = append_filter.GetOutput()
        return input_features


class DistanceBasedMetrics(Processor):
    def __init__(self, reference_keys=('xpoly', 'ypoly',), pre_process_keys=('ypoly', 'xpoly',),
                 post_process_keys=('bt_poly', 'ft_poly',), paired=False, use_scalars=False,
                 scalar_name='label_scalar'):
        '''
        :param reference_keys:
        :param pre_process_keys: compute distance before deformation to evaluate rigid alignement (for ex)
        :param post_process_keys: compute distance after deformation to evaluate rigid alignement (for ex)
        :param paired: bool if points from the reference and moving mesh have the same index (order)
        '''
        self.reference_keys = reference_keys
        self.pre_process_keys = pre_process_keys
        self.post_process_keys = post_process_keys
        self.paired = paired
        self.use_scalars = use_scalars
        self.scalar_name = scalar_name

    def compute_distance_metrics(self, reference, moving, paired=False):
        reference = numpy_support.vtk_to_numpy(reference.GetPoints().GetData())
        moving = numpy_support.vtk_to_numpy(moving.GetPoints().GetData())
        if paired:
            distances = np.sort(np.sqrt(np.sum(np.square(reference - moving), -1)))
            dta_metric = np.mean(distances)
            hd_metric = distances[-1]
            hd_95th_metric = distances[int(0.95 * len(distances))]
        else:
            M = reference.shape[0]
            N = moving.shape[0]
            D = reference.shape[-1]
            kernel = np.zeros((N, M))
            for i in range(D):
                temp1 = np.tile(reference[:, i], (N, 1))
                temp2 = np.transpose(np.tile(moving[:, i], (M, 1)))
                kernel = kernel + np.square(temp1 - temp2)
            kernel = np.sqrt(kernel)
            # compute distances in both directions and return average
            distances1 = np.sort(np.min(kernel, axis=-1))
            distances2 = np.sort(np.min(kernel, axis=0))
            dta_metric = (np.mean(distances1) + np.mean(distances2)) / 2
            hd_metric = (distances1[-1] + distances2[-1]) / 2
            hd_95th_metric = (distances1[int(0.95 * len(distances1))] + distances2[int(0.95 * len(distances2))]) / 2
        return dta_metric, hd_metric, hd_95th_metric

    def compute_metric(self, ref_key, input_key, input_features):
        refence_polydata = input_features[ref_key]
        input_polydata = input_features[input_key]

        if self.use_scalars:
            refence_polydata.GetPointData().SetActiveScalars(self.scalar_name)
            scalar_range = refence_polydata.GetScalarRange()
            for value in range(int(max(scalar_range)) + 1):
                reference_temp = threshold_polydata(refence_polydata, value)
                input_temp = threshold_polydata(input_polydata, value)
                dta_metric, hd_metric, hd_95th_metric = self.compute_distance_metrics(reference_temp, input_temp,
                                                                                      paired=self.paired)
                input_features["{}_{}_dta_{}".format(ref_key, input_key, value)] = dta_metric
                input_features["{}_{}_hd_{}".format(ref_key, input_key, value)] = hd_metric
                input_features["{}_{}_hd95th_{}".format(ref_key, input_key, value)] = hd_95th_metric
        else:
            dta_metric, hd_metric, hd_95th_metric = self.compute_distance_metrics(refence_polydata, input_polydata,
                                                                                  paired=self.paired)
            input_features["{}_{}_dta".format(ref_key, input_key)] = dta_metric
            input_features["{}_{}_hd".format(ref_key, input_key)] = hd_metric
            input_features["{}_{}_hd95th".format(ref_key, input_key)] = hd_95th_metric
        return input_features

    def pre_process(self, input_features):
        _check_keys_(input_features, self.reference_keys + self.pre_process_keys)
        for reference_key, pre_process_key in zip(self.reference_keys, self.pre_process_keys):
            self.compute_metric(reference_key, pre_process_key, input_features)
        return input_features

    def post_process(self, input_features):
        _check_keys_(input_features, self.reference_keys + self.post_process_keys)
        for reference_key, post_process_key in zip(self.reference_keys, self.post_process_keys):
            self.compute_metric(reference_key, post_process_key, input_features)
        return input_features


class VolumeBasedMetrics(Processor):
    def __init__(self, reference_keys=('xpoly', 'ypoly',), pre_process_keys=('ypoly', 'xpoly',),
                 post_process_keys=('bt_poly', 'ft_poly',), use_scalars=False, scalar_name='label_scalar'):
        '''
        :param reference_keys:
        :param pre_process_keys: compute dice before deformation to evaluate rigid alignement (for ex)
        :param post_process_keys: compute dice after deformation to evaluate rigid alignement (for ex)
        '''
        self.reference_keys = reference_keys
        self.pre_process_keys = pre_process_keys
        self.post_process_keys = post_process_keys
        self.use_scalars = use_scalars
        self.scalar_name = scalar_name

    def compute_metric(self, ref_key, input_key, input_features):
        refence_polydata = input_features[ref_key]
        input_polydata = input_features[input_key]
        if self.use_scalars:
            refence_polydata.GetPointData().SetActiveScalars(self.scalar_name)
            scalar_range = refence_polydata.GetScalarRange()
            for value in range(int(max(scalar_range)) + 1):
                reference_temp = threshold_selection_polydata(refence_polydata, value)
                ref_converter = DataConverter(polydata=reference_temp, cast_float32=False)
                ref_mask = ref_converter.polydata_to_mask()
                input_temp = threshold_selection_polydata(input_polydata, value)
                input_converter = DataConverter(polydata=input_temp, cast_float32=False)
                input_mask = input_converter.polydata_to_mask()
                input_mask = sitk.Resample(input_mask, ref_mask, interpolator=sitk.sitkNearestNeighbor,
                                           defaultPixelValue=0)
                overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
                overlap_filter.Execute(ref_mask, input_mask)
                input_features["{}_{}_dsc_{}".format(ref_key, input_key, value)] = overlap_filter.GetDiceCoefficient()
                input_features[
                    "{}_{}_jacc_{}".format(ref_key, input_key, value)] = overlap_filter.GetJaccardCoefficient()
                input_features[
                    "{}_{}_overlap_{}".format(ref_key, input_key, value)] = overlap_filter.GetVolumeSimilarity()
        else:
            ref_converter = DataConverter(polydata=refence_polydata, cast_float32=False)
            ref_mask = ref_converter.polydata_to_mask()
            input_converter = DataConverter(polydata=input_polydata, cast_float32=False)
            input_mask = input_converter.polydata_to_mask()
            input_mask = sitk.Resample(input_mask, ref_mask, interpolator=sitk.sitkNearestNeighbor, defaultPixelValue=0)
            overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
            overlap_filter.Execute(ref_mask, input_mask)
            input_features["{}_{}_dsc".format(ref_key, input_key)] = overlap_filter.GetDiceCoefficient()
            input_features["{}_{}_jacc".format(ref_key, input_key)] = overlap_filter.GetJaccardCoefficient()
            input_features["{}_{}_overlap".format(ref_key, input_key)] = overlap_filter.GetVolumeSimilarity()
        return input_features

    def pre_process(self, input_features):
        _check_keys_(input_features, self.reference_keys + self.pre_process_keys)
        for reference_key, pre_process_key in zip(self.reference_keys, self.pre_process_keys):
            self.compute_metric(reference_key, pre_process_key, input_features)
        return input_features

    def post_process(self, input_features):
        _check_keys_(input_features, self.reference_keys + self.post_process_keys)
        for reference_key, post_process_key in zip(self.reference_keys, self.post_process_keys):
            self.compute_metric(reference_key, post_process_key, input_features)
        return input_features


class SimplifyMask(Processor):
    def __init__(self, input_keys=('mask1', 'mask2'), output_keys=('mask1', 'mask2'), type_keys=('closing', 'closing'),
                 radius_keys=(1, 1), thread_count=8):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.type_keys = type_keys
        self.radius_keys = radius_keys
        self.thread_count = thread_count

    def worker_def(self, A):
        q = A
        while True:
            item = q.get()
            if item is None:
                break
            else:
                iteration, input_features, input_key, output_key, type_key, radius_key = item
                try:
                    image = input_features[input_key]
                    if not isinstance(image, np.ndarray):
                        spacing = image.GetSpacing()
                        origin = image.GetOrigin()
                        direction = image.GetDirection()
                        mask = compute_binary_morphology(sitk.GetArrayFromImage(image), radius_key, type_key)
                        image = sitk.GetImageFromArray(mask.astype(np.int8))
                        image.SetSpacing(spacing)
                        image.SetOrigin(origin)
                        image.SetDirection(direction)
                        input_features[output_key] = image
                    else:
                        input_features[output_key] = compute_binary_morphology(image, radius_key, type_key).astype(
                            np.int8)
                except:
                    print('failed on class {}, '.format(iteration))
                q.task_done()

    def pre_process(self, input_features):
        _check_keys_(input_features, self.input_keys)

        # init threads
        q = Queue(maxsize=self.thread_count)
        threads = []
        for worker in range(self.thread_count):
            t = Thread(target=self.worker_def, args=(q,))
            t.start()
            threads.append(t)

        iteration = 1
        for input_key, output_key, type_key, radius_key in zip(self.input_keys, self.output_keys, self.type_keys,
                                                               self.radius_keys):
            item = [iteration, input_features, input_key, output_key, type_key, radius_key]
            iteration += 1
            q.put(item)

        for i in range(self.thread_count):
            q.put(None)
        for t in threads:
            t.join()
        return input_features


class ExtractCenterline(Processor):
    def __init__(self, input_keys=('fixed_rectum', 'moving_rectum',),
                 output_keys=('fixed_centerline', 'moving_centerline',)):
        '''
        centerline extraction and filtering based on https://github.com/gabyx/WormAnalysis
        :param input_keys:
        :param output_keys:
        '''
        self.input_keys = input_keys
        self.output_keys = output_keys

    class Vertex:
        def __init__(self, point, degree=0, edges=None):
            self.point = np.asarray(point)
            self.degree = degree
            self.edges = []
            self.visited = False
            if edges is not None:
                self.edges = edges

        def __str__(self):
            return str(self.point)

    class Edge:
        def __init__(self, start, end=None, pixels=None):
            self.start = start
            self.end = end
            self.pixels = []
            if pixels is not None:
                self.pixels = pixels
            self.visited = False

    def buildTree(self, img, start=None):
        # copy image since we set visited pixels to black
        img = img.copy()
        shape = img.shape
        nWhitePixels = np.sum(img)
        # neighbor offsets (8 nbors)
        nbPxOff = np.array([[-1, -1, 0], [-1, 0, 0], [-1, 1, 0], [0, -1, 0],
                            [0, 1, 0], [1, -1, 0], [1, 0, 0], [1, 1, 0],
                            [-1, -1, 1], [-1, 0, 1], [-1, 1, 1], [0, -1, 1],
                            [0, 1, 1], [1, -1, 1], [1, 0, 1], [1, 1, 1],
                            [-1, -1, -1], [-1, 0, -1], [-1, 1, -1], [0, -1, -1],
                            [0, 1, -1], [1, -1, -1], [1, 0, -1], [1, 1, -1],
                            [0, 0, 1], [0, 0, -1]])
        queue = collections.deque()
        # a list of all graphs extracted from the skeleton
        graphs = []
        blackedPixels = 0
        # we build our graph as long as we have not blacked all white pixels!
        while nWhitePixels != blackedPixels:
            # if start not given: determine the first white pixel
            if start is None:
                it = np.nditer(img, flags=['multi_index'])
                while not it[0]:
                    it.iternext()
                start = it.multi_index
            startV = self.Vertex(start)
            queue.append(startV)
            # print("Start vertex: ", startV)
            # set start pixel to False (visited)
            img[startV.point[0], startV.point[1], startV.point[2]] = False
            blackedPixels += 1
            # create a new graph
            G = nx.Graph()
            G.add_node(startV)
            # build graph in a breath-first manner by adding
            # new nodes to the right and popping handled nodes to the left in queue
            while len(queue):
                currV = queue[0]  # get current vertex
                # print("Current vertex: ", currV)
                # check all neigboor pixels
                for nbOff in nbPxOff:
                    # pixel index
                    pxIdx = currV.point + nbOff
                    if (pxIdx[0] < 0 or pxIdx[0] >= shape[0]) or (pxIdx[1] < 0 or pxIdx[1] >= shape[1]) or (
                            pxIdx[2] < 0 or pxIdx[2] >= shape[2]):
                        continue  # current neigbor pixel out of image
                    if img[pxIdx[0], pxIdx[1], pxIdx[2]]:
                        # print( "nb: ", pxIdx, " white ")
                        # pixel is white
                        newV = self.Vertex([pxIdx[0], pxIdx[1], pxIdx[2]])
                        # add edge from currV <-> newV
                        G.add_edge(currV, newV, object=self.Edge(currV, newV))
                        # G.add_edge(newV,currV)
                        # add node newV
                        G.add_node(newV)
                        # push vertex to queue
                        queue.append(newV)
                        # set neighbor pixel to black
                        img[pxIdx[0], pxIdx[1], pxIdx[2]] = False
                        blackedPixels += 1
                # pop currV
                queue.popleft()
            # end while
            # empty queue
            # current graph is finished ->store it
            graphs.append(G)
            # reset start
            start = None
        # end while
        return graphs, img

    def getEndNodes(self, g):
        return [n for n in nx.nodes(g) if nx.degree(g, n) == 1]

    def mergeEdges(self, graph):
        # copy the graph
        g = graph.copy()

        # v0 -----edge 0--- v1 ----edge 1---- v2
        #        pxL0=[]       pxL1=[]           the pixel lists
        #
        # becomes:
        #
        # v0 -----edge 0--- v1 ----edge 1---- v2
        # |_________________________________|
        #               new edge
        #    pxL = pxL0 + [v.point]  + pxL1      the resulting pixel list on the edge
        #
        # an delete the middle one
        # result:
        #
        # v0 --------- new edge ------------ v2
        #
        # where new edge contains all pixels in between!

        # start not at degree 2 nodes
        startNodes = [startN for startN in g.nodes() if nx.degree(g, startN) != 2]

        for v0 in startNodes:
            # start a line traversal from each neighbor
            # startNNbs = nx.neighbors(g, v0) ONLY FOR nx<2.0
            startNNbs = list(g.neighbors(v0))
            if not len(startNNbs):
                continue
            counter = 0
            v1 = startNNbs[counter]  # next nb of v0
            while True:
                if nx.degree(g, v1) == 2:
                    # we have a node which has 2 edges = this is a line segment
                    # make new edge from the two neighbors
                    # nbs = nx.neighbors(g, v1) ONLY FOR nx<2.0
                    nbs = list(g.neighbors(v1))
                    # if the first neihbor is not n, make it so!
                    if nbs[0] != v0:
                        nbs.reverse()
                    pxL0 = g[v0][v1]["object"].pixels  # the pixel list of the edge 0
                    pxL1 = g[v1][nbs[1]]["object"].pixels  # the pixel list of the edge 1
                    # fuse the pixel list from right and left and add our pixel n.point
                    g.add_edge(v0, nbs[1], object=self.Edge(v0, nbs[1], pixels=pxL0 + [v1.point] + pxL1))
                    # delete the node n
                    g.remove_node(v1)
                    # set v1 to new left node
                    v1 = nbs[1]
                else:
                    counter += 1
                    if counter == len(startNNbs):
                        break;
                    v1 = startNNbs[counter]  # next nb of v0
        # weight the edges according to their number of pixels
        for u, v, o in g.edges(data="object"):
            g[u][v]["weight"] = len(o.pixels)
        return g

    def getLongestPath(self, graph, endNodes):
        """
            graph is a fully reachable graph = every node can be reached from every node
        """

        if len(endNodes) < 2:
            raise ValueError("endNodes need to contain at least 2 nodes!")

        # get all shortest paths from each endpoint to another endpoint
        allEndPointsComb = itertools.combinations(endNodes, 2)
        maxLength = 0
        maxPath = None
        for ePoints in allEndPointsComb:
            # get shortest path for these end points pairs
            sL = nx.dijkstra_path_length(graph, source=ePoints[0], target=ePoints[1])
            # dijkstra can throw if now path, but we are sure we have a path
            # store maximum
            if (sL > maxLength):
                maxPath = ePoints
                maxLength = sL
        if maxPath is None:
            raise ValueError("No path found!")
        return nx.dijkstra_path(graph, source=maxPath[0], target=maxPath[1]), maxLength

    def filter_graphs(self, graphs):
        # filter graphs (remove branches)
        for i, g in enumerate(graphs):
            # endNodes = self.getEndNodes(g)
            # merge the nodes between endNodes
            merged_graph = self.mergeEdges(g)
            merged_endNodes = self.getEndNodes(merged_graph)
            # compute the largest path for each sub graph
            longestPathNodes = self.getLongestPath(merged_graph, merged_endNodes)
            longestPathEdges = [(longestPathNodes[0][j], longestPathNodes[0][j + 1]) for j in
                                range(0, len(longestPathNodes[0]) - 1)]
            # extract only the graph with the longest path
            filtered_graph = nx.Graph()
            for e in longestPathEdges:
                filtered_graph.add_node(e[0])
                list_nodes = [self.Vertex([point[0], point[1], point[2]]) for point in
                              merged_graph[e[0]][e[1]]["object"].pixels]
                filtered_graph.add_nodes_from(list_nodes)
                filtered_graph.add_node(e[1])
        return filtered_graph

    def pre_process(self, input_features):
        _check_keys_(input_features, self.input_keys)
        for input_key, output_key in zip(self.input_keys, self.output_keys):
            image = input_features[input_key]
            bb_parameters = compute_bounding_box(image)
            cropped_image = image[bb_parameters[0]:bb_parameters[1],
                            bb_parameters[2]:bb_parameters[3],
                            bb_parameters[4]:bb_parameters[5]]
            skelet = skimage.morphology.skeletonize_3d(cropped_image)
            skelet[skelet > 0] = 1
            skelet[skelet != 1] = 0
            graphs, imgB = self.buildTree(img=skelet, start=None)
            filtered_graph = self.filter_graphs(graphs)
            filtered_skelet = np.zeros_like(skelet)
            indices = np.array([n.point for n in list(nx.nodes(filtered_graph))])
            filtered_skelet[tuple(np.transpose(indices))] = 1
            skelet_image = np.zeros_like(image)
            skelet_image[bb_parameters[0]:bb_parameters[1],
            bb_parameters[2]:bb_parameters[3],
            bb_parameters[4]:bb_parameters[5]] = filtered_skelet
            input_features[output_key] = skelet_image
            # add offset of cropping
            indices = indices + [bb_parameters[0], bb_parameters[2], bb_parameters[4]]
            # reorder into X, Y, Z direction here
            input_features[output_key + '_sorted_pts'] = indices[:, [2, 1, 0]]
        return input_features


class CenterlineToPolydata(Processor):
    def __init__(self, input_keys=('fixed_centerline', 'moving_centerline',),
                 output_keys=('fpoly_centerline', 'mpoly_centerline',),
                 spacing_keys=(), origin_keys=(), nbsplinepts=100):
        self.input_keys = input_keys
        self.output_keys = output_keys
        empty_tuple = tuple([None for i in self.input_keys])
        self.spacing_keys = empty_tuple if len(spacing_keys) == 0 else spacing_keys
        self.origin_keys = empty_tuple if len(origin_keys) == 0 else origin_keys
        self.nbsplinepts = nbsplinepts

    def pre_process(self, input_features):
        _check_keys_(input_features, self.input_keys)
        for input_key, output_key, spacing_key, origin_key in zip(self.input_keys, self.output_keys, self.spacing_keys,
                                                                  self.origin_keys):
            sorted_points = input_features[input_key + '_sorted_pts']
            if spacing_key:
                spacing = input_features[spacing_key]
                sorted_points = sorted_points * spacing
            if origin_key:
                origin = input_features[origin_key]
                sorted_points = sorted_points + origin
            new_points = numpy_support.numpy_to_vtk(sorted_points)
            vtkPts = vtk.vtkPoints()
            vtkPts.SetData(new_points)
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(vtkPts)
            lines = vtk.vtkCellArray()
            for pts in range(vtkPts.GetNumberOfPoints() - 1):
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, pts)
                line.GetPointIds().SetId(1, pts + 1)
                lines.InsertNextCell(line)
            polydata.SetLines(lines)
            clean_polydata = vtk.vtkCleanPolyData()
            clean_polydata.SetInputData(polydata)
            clean_polydata.Update()
            xspline = vtk.vtkKochanekSpline()
            yspline = vtk.vtkKochanekSpline()
            zspline = vtk.vtkKochanekSpline()
            spline = vtk.vtkParametricSpline()
            spline.SetXSpline(xspline)
            spline.SetYSpline(yspline)
            spline.SetZSpline(zspline)
            spline.SetPoints(clean_polydata.GetOutput().GetPoints())
            function_source = vtk.vtkParametricFunctionSource()
            function_source.SetParametricFunction(spline)
            function_source.SetUResolution(self.nbsplinepts)
            function_source.SetVResolution(self.nbsplinepts)
            function_source.SetWResolution(self.nbsplinepts)
            function_source.Update()
            input_features[output_key] = function_source.GetOutput()
        return input_features


class ComputeLaplacianCorrespondence(Processor):
    def __init__(self, input_keys=('fixed_rectum', 'moving_rectum',),
                 centerline_keys=('fixed_centerline', 'moving_centerline',),
                 output_keys=('fixed_correspondence', 'moving_correspondence',),
                 spacing_keys=((1.0, 1.0, 1.0), (1.0, 1.0, 1.0),), dilate_centerline=False):
        """
        :param input_keys: input
        :param centerline_keys: can be tuple of None
        :param output_keys:
        :param spacing_keys: for example ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0),))
        :param dilate_centerline: bool to increase size of internal condition
        """
        self.input_keys = input_keys
        empty_tuple = tuple([None for i in self.input_keys])
        self.centerline_keys = empty_tuple if len(centerline_keys) == 0 else centerline_keys
        self.output_keys = output_keys
        identity_tuple = tuple([(1.0, 1.0, 1.0) for i in self.input_keys])
        self.spacing_keys = identity_tuple if len(spacing_keys) == 0 else spacing_keys
        self.dilate_centerline = dilate_centerline

    def pre_process(self, input_features):
        _check_keys_(input_features, self.input_keys + self.centerline_keys)
        for input_key, centerline_key, output_key, spacing_key in zip(self.input_keys, self.centerline_keys,
                                                                      self.output_keys, self.spacing_keys):
            input = input_features[input_key]
            internal = input_features[centerline_key]
            spacing = input_features[spacing_key]

            if self.dilate_centerline:
                erode_input = compute_binary_morphology(input, radius=1, morph_type='erosion')
                internal = compute_binary_morphology(internal, radius=2, morph_type='dilation')
                # avoid internal leaking to input border
                internal = internal * erode_input

            laplacian_filter = Laplacian(input=input, internal=internal, spacing=spacing[[2, 1, 0]],
                                         cl_max=1000, cl_min=200, compute_thickness=False,
                                         compute_internal_corresp=True, compute_external_corresp=False, verbose=False)
            input_features[output_key] = laplacian_filter.phi0
        return input_features


class CenterlineProjection(Processor):
    def __init__(self, input_keys=('xpoly', 'ypoly',), centerline_keys=('fpoly_centerline', 'mpoly_centerline',),
                 correspondence_keys=('fixed_correspondence', 'moving_correspondence',),
                 output_keys=('xpoly', 'ypoly',), origin_keys=(), spacing_keys=()):
        self.input_keys = input_keys
        self.centerline_keys = centerline_keys
        self.correspondence_keys = correspondence_keys
        self.output_keys = output_keys
        zeros_tuple = tuple([(0.0, 0.0, 0.0) for i in self.input_keys])
        identity_tuple = tuple([(1.0, 1.0, 1.0) for i in self.input_keys])
        self.origin_keys = zeros_tuple if len(origin_keys) == 0 else origin_keys
        self.spacing_keys = identity_tuple if len(spacing_keys) == 0 else spacing_keys

    def find_closest_vector_index(self, point, vector_indices, spacing, origin):
        distances = np.sqrt(
            np.sum(np.square((vector_indices * spacing[[2, 1, 0]] + origin[[2, 1, 0]]) - point[[2, 1, 0]]), axis=-1))
        closest_index = np.argmin(distances)
        return closest_index

    def pre_process(self, input_features):
        _check_keys_(input_features, self.input_keys + self.centerline_keys + self.correspondence_keys)
        for input_key, centerline_key, correspondence_key, output_key, origin_key, spacing_key in zip(self.input_keys,
                                                                                                      self.centerline_keys,
                                                                                                      self.correspondence_keys,
                                                                                                      self.output_keys,
                                                                                                      self.origin_keys,
                                                                                                      self.spacing_keys):
            polydata = input_features[input_key]
            centerline = input_features[centerline_key]
            correspondence = input_features[correspondence_key]
            origin = input_features[origin_key]
            spacing = input_features[spacing_key]

            polydata_pts = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
            centerline_pts = numpy_support.vtk_to_numpy(centerline.GetPoints().GetData())
            indices = np.argwhere(correspondence > 0)[..., 0:3]
            scalar = np.zeros(len(polydata_pts))
            vector_field = np.zeros_like(polydata_pts)
            vector_field_filtered = np.zeros_like(polydata_pts)

            for i in range(len(polydata_pts)):
                point = polydata_pts[i]
                vector_index = self.find_closest_vector_index(point, indices, spacing, origin)
                vector = correspondence[tuple(indices[vector_index])]
                vector_field[i] = vector[[2, 1, 0]]
                projected_point = point + vector[[2, 1, 0]]
                pts_distances = np.sqrt(np.sum(np.square(centerline_pts - projected_point), axis=-1))
                closest_index = np.argmin(pts_distances)
                scalar[i] = closest_index / (len(centerline_pts) - 1)
                vector_field_filtered[i] = centerline_pts[closest_index] - point

            output = vtk.vtkPolyData()
            output.DeepCopy(polydata)
            # vtk_vector_field = numpy_support.numpy_to_vtk(vector_field)
            # vtk_vector_field.SetName('Correspondence')
            # output.GetPointData().SetVectors(vtk_vector_field)
            vtk_vector_field = numpy_support.numpy_to_vtk(vector_field_filtered)
            vtk_vector_field.SetName('Filtered_correspondence')
            output.GetPointData().SetVectors(vtk_vector_field)
            vtk_scalar = numpy_support.numpy_to_vtk(scalar)
            vtk_scalar.SetName('Length')
            output.GetPointData().SetScalars(vtk_scalar)
            input_features[output_key] = output
        return input_features


class ComputePolydataICP(Processor):
    def __init__(self, fixed_key='fpoly_prostate', moving_key='fpoly_prostate', type='centroid',
                 matrix_key='transform_matrix'):
        """
        :param fixed_key:
        :param moving_key:
        :param type: centroid does not include rotation, icp does
        :param matrix_key: vtkMatrix4x4 to be used by a vtk.vtkTransform() filter
        """
        self.fixed_key = fixed_key
        self.moving_key = moving_key
        self.matrix_key = matrix_key
        if type == 'centroid':
            self.iteration = 0
        elif type == 'icp':
            self.iteration = 50
        else:
            raise ValueError("Type not in [centroid, icp]")

    def pre_process(self, input_features):
        _check_keys_(input_features, (self.fixed_key, self.moving_key))
        icp_filter = vtk.vtkIterativeClosestPointTransform()
        icp_filter.SetSource(input_features[self.moving_key])
        icp_filter.SetTarget(input_features[self.fixed_key])
        icp_filter.GetLandmarkTransform().SetModeToRigidBody()
        icp_filter.SetMaximumNumberOfIterations(self.iteration)
        icp_filter.StartByMatchingCentroidsOn()
        icp_filter.Modified()
        icp_filter.Update()
        input_features[self.matrix_key] = icp_filter.GetMatrix()
        # icp_transform_filter = vtk.vtkTransformPolyDataFilter()
        # icp_transform_filter.SetInputData(input_features[self.moving_key])
        # icp_transform_filter.SetTransform(icp_filter);
        # icp_transform_filter.Update()
        # test = icp_transform_filter.GetOutput()
        return input_features


class ApplyPolydataTransform(Processor):
    def __init__(self, moving_keys=('',), matrix_keys=('',), output_keys=('',)):
        """
        :param moving_keys:
        :param matrix_keys: vtkMatrix4x4 to be used by a vtk.vtkTransform() filter
        :param output_keys:
        """
        self.moving_keys = moving_keys
        self.matrix_keys = matrix_keys
        self.output_keys = output_keys

    def pre_process(self, input_features):
        _check_keys_(input_features, self.moving_keys + self.matrix_keys)
        for moving_key, matrix_key, output_key in zip(self.moving_keys, self.matrix_keys, self.output_keys):
            polydata = input_features[moving_key]
            transform = vtk.vtkTransform()
            transform.SetMatrix(input_features[matrix_key])
            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetInputData(polydata)
            transform_filter.SetTransform(transform)
            transform_filter.SetOutputPointsPrecision(2)
            transform_filter.Update()
            input_features[output_key] = transform_filter.GetOutput()
        return input_features


class RemoveUnconnectedComponents(Processor):
    def __init__(self, input_keys=('',), output_keys=('',)):
        self.input_keys = input_keys
        self.output_keys = output_keys

    def pre_process(self, input_features):
        _check_keys_(input_features, self.input_keys)
        for input_key, output_key in zip(self.input_keys, self.output_keys):
            component_filter = Remove_Smallest_Structures()
            input_features[output_key] = component_filter.remove_smallest_component(input_features[input_key])
        return input_features


class ComputeDistanceGeodesic(Processor):
    def __init__(self, input_keys=('',), output_keys=('',)):
        self.input_keys = input_keys
        self.output_keys = output_keys

    def find_basis_point(self, polydata):
        polydata_pts = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
        lastz = 99999
        nb = 0
        coord = [0] * 3
        centroid = [0] * 3
        # get last point
        for i in range(len(polydata_pts)):
            point = polydata_pts[i]
            if point[2] < lastz:
                lastz = point[2]
        # get "floor" of last point
        for i in range(len(polydata_pts)):
            point = polydata_pts[i]
            if math.floor(point[2]) == math.floor(lastz):
                for j in range(3):
                    coord[j] += point[j]
                nb += 1
        for j in range(3):
            centroid[j] = coord[j] / nb
        # tree locator
        treelocator = vtk.vtkKdTreePointLocator()
        treelocator.SetDataSet(polydata)
        treelocator.BuildLocator()
        plist = vtk.vtkIdList()
        treelocator.FindClosestNPoints(1, centroid, plist)
        return plist.GetId(0)

    def compute_geodesic_distance(self, polydata, source_point_id, norm=True, norm_factor=1.0):
        output_scalars = vtk.vtkDoubleArray()
        dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
        dijkstra.SetInputData(polydata)
        dijkstra.SetStartVertex(source_point_id)
        dijkstra.Update()
        dijkstra.GetCumulativeWeights(output_scalars)
        if norm:
            scalars_np = numpy_support.vtk_to_numpy(output_scalars)
            max = np.max(scalars_np)
            scalars_np = norm_factor * scalars_np / max
            output_scalars = numpy_support.numpy_to_vtk(scalars_np)
        output_scalars.SetName('Geodesic')
        output_poly = vtk.vtkPolyData()
        output_poly.DeepCopy(polydata)
        output_poly.GetPointData().SetScalars(output_scalars)
        return output_poly

    def pre_process(self, input_features):
        _check_keys_(input_features, self.input_keys)
        for input_key, output_key in zip(self.input_keys, self.output_keys):
            input_poly = input_features[input_key]
            basis_point_id = self.find_basis_point(input_poly)
            input_features[output_key] = self.compute_geodesic_distance(input_poly, basis_point_id)
        return input_features
