# Created by Bastien Rigaud at 21/01/2022
# Bastien Rigaud, PhD
# Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099
# Campus de Beaulieu, Université de Rennes 1
# 35042 Rennes, FRANCE
# bastien.rigaud@univ-rennes1.fr
# Description:

import numpy as np
import vtk
import pyvista as pv
import pyacvd

from IOTools.IOTools import DataConverter

def _check_keys_(input_features, keys):
    if type(keys) is list or type(keys) is tuple:
        for key in keys:
            assert key in input_features.keys(), 'Make sure the key you are referring to is present in the features, ' \
                                                 '{} was not found'.format(key)
    else:
        assert keys in input_features.keys(), 'Make sure the key you are referring to is present in the features, ' \
                                              '{} was not found'.format(keys)

class Processor(object):
    def pre_process(self, input_features):
        return input_features

    def post_pocess(self, input_features):
        return input_features

class ACVD_resampling(Processor):
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
        _check_keys_(input_features)
        for input_key, output_key, np_point in zip(self.input_keys,self.output_keys, self.np_points):
            input_features[output_key] = self.compute_acvd(input_features[input_key], np_point)

class Convert_Mask_To_Poly(Processor):
    def __init__(self, input_keys=('xmask', 'ymask'), output_keys=('xpoly','ypoly')):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.converter = DataConverter()

    def pre_process(self, input_features):
        _check_keys_(input_features)
        for input_key, output_key in zip(self.input_keys, self.output_keys):
            image = input_features[input_key]
            converter = DataConverter(image=image, inval=1, outval=0, cast_float32=True)
            input_features[output_key] = converter.MaskToPolydata()

