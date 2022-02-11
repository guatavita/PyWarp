# Created by Bastien Rigaud at 21/01/2022
# Bastien Rigaud, PhD
# Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099
# Campus de Beaulieu, Université de Rennes 1
# 35042 Rennes, FRANCE
# bastien.rigaud@univ-rennes1.fr
# Description:

import sys, os
import numpy as np
import vtk
from vtk.util import numpy_support
import arrayfire as af
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PlotVTK.PlotVTK import plot_vtk

backends = af.get_available_backends()
if 'cuda' in backends:
    af.set_backend('cuda')
elif 'opencl' in backends:
    af.set_backend('opencl')
else:
    af.set_backend('cpu')

f32 = af.Dtype.f32


def _check_keys_(input_features, keys):
    if type(keys) is list or type(keys) is tuple:
        for key in keys:
            assert key in input_features.keys(), 'Make sure the key you are referring to is present in the features, ' \
                                                 '{} was not found'.format(key)
    else:
        assert keys in input_features.keys(), 'Make sure the key you are referring to is present in the features, ' \
                                              '{} was not found'.format(keys)


# TODO add support for multiparametric scalar (for example, label + topology)
# TODO add a costfunction merger to compute 2 cost functions (stpsrpm and DMR) and merge them with a lambda
# TODO add target mean dist stop criteria such as % of change from the past 5 iterations

def smooth_polydata(polydata, passband):
    smooth_filter = vtk.vtkWindowedSincPolyDataFilter()
    smooth_filter.SetInputData(polydata)
    # degree of the polynomial that is used to approximate the windowed sinc function
    smooth_filter.SetNumberOfIterations(20)
    smooth_filter.BoundarySmoothingOff()
    # interior vertices are classified as either "simple", "interior edge", or "fixed", and smoothed differently.
    smooth_filter.FeatureEdgeSmoothingOff()
    # a feature edge occurs when the angle between the two surface normals of a polygon sharing an edge is greater than the FeatureAngle
    smooth_filter.SetFeatureAngle(120.0)
    smooth_filter.SetPassBand(passband)  # lower PassBand values produce more smoothing
    smooth_filter.NonManifoldSmoothingOn()
    smooth_filter.NormalizeCoordinatesOn()
    smooth_filter.Update()
    return smooth_filter.GetOutput()


def convert_vtk_to_af(polydata, force_float32=True, transpose=True):
    points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
    if force_float32:
        points = points.astype(dtype=np.float32)
    af_array = af.Array(points.ctypes.data, points.shape[::-1], points.dtype.char)
    if transpose:
        af_array = af.transpose(af_array)
    af.eval(af_array)
    return af_array


def convert_af_to_vtk(template, array):
    new_points = numpy_support.numpy_to_vtk(array.to_ndarray())
    polydata = vtk.vtkPolyData()
    polydata.DeepCopy(template)
    polydata.GetPoints().SetData(new_points)
    return polydata


def convert_scalars_to_af(scalar, force_float32=True, transpose=True):
    scalars = numpy_support.vtk_to_numpy(scalar)
    if force_float32:
        scalars = scalars.astype(dtype=np.float32)
    if len(scalars.shape) == 1:
        scalars = scalars[..., None]
    af_array = af.Array(scalars.ctypes.data, scalars.shape[::-1], scalars.dtype.char)
    if transpose:
        af_array = af.transpose(af_array)
    af.eval(af_array)
    return af_array


def compute_centroid_af(array):
    centroid = af.sum(array, 0) / array.dims()[0]
    af.eval(centroid)
    return centroid


def compute_mean_dist(poly, virtual_poly):
    return af.mean(af.min(af.sqrt(compute_kernel_2(virtual_poly, poly)), 0))


def compute_kernel(array):
    if not len(array.dims()) > 1:
        raise ValueError("Array must be 2D to compute kernel")
    N = array.dims()[0]
    D = array.dims()[1]
    kernel = af.constant(0, N, N, dtype=f32)
    for i in range(D):
        temp1 = af.tile(array[:, i], 1, N)
        temp2 = af.tile(af.transpose(array[:, i]), N, 1)
        af.eval(temp1)
        af.eval(temp2)
        kernel = kernel + af.pow(temp1 - temp2, 2)
        af.eval(kernel)
    return kernel


def compute_kernel_2(array1, array2):
    M = array1.dims()[0]
    N = array2.dims()[0]
    if len(array1.dims()) > 1:
        D = array1.dims()[1]
    else:
        D = 1

    kernel = af.constant(0, M, N, dtype=f32)
    for i in range(D):
        temp1 = af.tile(array1[:, i], 1, N)
        temp2 = af.tile(af.transpose(array2[:, i]), M, 1)
        af.eval(temp1)
        af.eval(temp2)
        kernel = kernel + af.pow(temp1 - temp2, 2)
        af.eval(kernel)
    return kernel


class CostFunction(object):
    def parse(self, input_features):
        return input_features


class STPSRPM(CostFunction):
    def __init__(self, xpoly_key='xpoly', ypoly_key='xpoly', ft_out_key='ft_poly', bt_out_key='bt_poly',
                 lambda1_init=0.01, lambda2_init=1, T_init=0.5, T_final=0.001, anneal_rate=0.93, threshold=0.000001,
                 use_scalar_vtk=False, xlm_key=None, ylm_key=None, passband=[1], iterative_norm=True, nbiter=None,
                 scalars_name=None):
        """
        :param xpoly_key: source input as a VTK polydata
        :param ypoly_key: target input as a VTK polydata
        :param ft_out_key: forward transformation output key (x deform towards y)
        :param bt_out_key: backward transformation output key (y deform towards x)
        :param lambda1: lambda1 init value, weight for the 'non-linear' part of the TPS
        :param lambda2: lambda2 init value, Weight for the affine part of the TPS
        :param t_init: initial temperature value
        :param t_final: final temperature value
        :param rate: annealing rate (default: 0.93)
        :param threshold: threshold the matrix-m to remove outliers (default: 0.000001)
        :param use_scalar_vtk: use_scalar_vtk used to cluster group of points using scalar information from the vtk
        :param xlm: vtk pts (landmarks) related to the x shape with same index ordering as lmy (use as constraint in the matrix m)
        :param ylm: vtk pts (landmarks) related to the y shape with same index ordering as lmy (use as constraint in the matrix m)
        :param passband: passband value for smooth filter (advice: [0.01,0.1,1]) (default=1 /eq to no smoothing)
        :param iterative_norm: iterative normalization of the matrix m (default: True)
        :param nbiter: override number of iteration (mostly for quick debug)
        """
        self.xpoly_key = xpoly_key
        self.ypoly_key = ypoly_key
        self.ft_out_key = ft_out_key
        self.bt_out_key = bt_out_key
        self.xlm_key = xlm_key
        self.ylm_key = ylm_key
        self.D = 3
        self.threshold = threshold
        self.use_scalar_vtk = use_scalar_vtk
        self.scalars_name = scalars_name
        self.passband = passband
        self.iterative_norm = iterative_norm
        self.perT_maxit = 2
        self.ita = 0
        self.lambda1_init = lambda1_init
        self.lambda2_init = lambda2_init
        self.T_init = T_init
        self.T = T_init
        self.anneal_rate = anneal_rate
        self.nbiter = nbiter
        # self.metric_watcher = []
        # self.metric_threshold = 0.05
        # self.watcher_size = 5
        if not self.nbiter:
            self.nbiter = round(math.log(T_final / T_init) / math.log(anneal_rate))

    def parse(self, input_features):
        _check_keys_(input_features, (self.xpoly_key, self.ypoly_key))
        self.xpoly_vtk = input_features[self.xpoly_key]
        self.ypoly_vtk = input_features[self.ypoly_key]
        if self.use_scalar_vtk and self.scalars_name:
            self.xpoly_vtk.GetPointData().SetActiveScalars(self.scalars_name)
            self.ypoly_vtk.GetPointData().SetActiveScalars(self.scalars_name)
            self.xscalar_vtk = self.xpoly_vtk.GetPointData().GetScalars()
            self.yscalar_vtk = self.ypoly_vtk.GetPointData().GetScalars()
            if self.xscalar_vtk == None or self.yscalar_vtk == None:
                raise ValueError("One of the two inputs has no scalar.")
        self.xpoints = self.xpoly_vtk.GetNumberOfPoints()
        self.ypoints = self.ypoly_vtk.GetNumberOfPoints()
        self.xlm_vtk = input_features.get(self.xlm_key)
        self.ylm_vtk = input_features.get(self.ylm_key)
        self.lm_size = 0
        if self.xlm_vtk and self.ylm_vtk:
            if self.xlm_vtk.GetNumberOfPoints() != self.ylm_vtk.GetNumberOfPoints():
                raise ValueError("Provided landmarks does not have the same size")
            self.lm_size = self.xlm_vtk.GetNumberOfPoints()
        self.allocate_data()
        virtual_xpoly, virtual_ypoly = self.update()
        input_features[self.ft_out_key] = convert_af_to_vtk(self.xpoly_vtk, virtual_xpoly[self.lm_size:self.xpoints, :])
        input_features[self.bt_out_key] = convert_af_to_vtk(self.ypoly_vtk, virtual_ypoly[self.lm_size:self.ypoints, :])
        return input_features

    def allocate_data(self):
        self.xpoly = convert_vtk_to_af(self.xpoly_vtk, force_float32=True)
        self.ypoly = convert_vtk_to_af(self.ypoly_vtk, force_float32=True)

        if self.use_scalar_vtk:
            self.xscalar = convert_scalars_to_af(self.xscalar_vtk, force_float32=True)
            self.yscalar = convert_scalars_to_af(self.yscalar_vtk, force_float32=True)

        if self.xlm_vtk and self.ylm_vtk:
            self.xlm = convert_vtk_to_af(self.xlm_vtk, force_float32=True)
            self.ylm = convert_vtk_to_af(self.ylm_vtk, force_float32=True)
            # join landmark with xpoly between[0:lm_size]
            self.xpoly = af.join(0, self.xlm, self.xpoly)
            self.ypoly = af.join(0, self.ylm, self.ypoly)
            # increase size
            self.xpoints += self.lm_size
            self.ypoints += self.lm_size
            # update scalar with dummy scalars for the landmark (not influenced)
            if self.use_scalar_vtk:
                self.xscalar = af.join(0, af.constant(0, self.lm_size, self.xscalar_vtk.GetNumberOfComponents(),
                                                      dtype=f32), self.xscalar)
                self.yscalar = af.join(0, af.constant(0, self.lm_size, self.yscalar_vtk.GetNumberOfComponents(),
                                                      dtype=f32), self.yscalar)

        self.m_matrix = af.constant(0, self.xpoints, self.ypoints, dtype=f32)
        self.m_outliers_row = af.constant(0, self.ypoints, dtype=f32)
        self.m_outliers_col = af.constant(0, self.xpoints, dtype=f32)
        self.K_ft = af.constant(0, self.xpoints, self.xpoints, dtype=f32)
        self.c_ft = af.constant(0, self.xpoints, self.D + 1, dtype=f32)
        self.d_ft = af.constant(0, self.D + 1, self.D + 1, dtype=f32)
        self.K_bt = af.constant(0, self.ypoints, self.ypoints, dtype=f32)
        self.c_bt = af.constant(0, self.ypoints, self.D + 1, dtype=f32)
        self.d_bt = af.constant(0, self.D + 1, self.D + 1, dtype=f32)
        self.xcentroid = compute_centroid_af(self.xpoly)
        self.ycentroid = compute_centroid_af(self.ypoly)

    def compute_m(self, virtual_xpoly, virtual_ypoly, use_scalar_vtk=False):
        '''
        Compute the soft assignment matrix:
        . virtual_xpoly - updated source points
        . ypoly         - target points
        . m_matrix      - soft assignment matrix
        . xsize         - number of source points + 1 (outliers) (should be the line)
        . ysize         - number of target points + 1 (outliers) (should be the column)
        :param virtual_xpoly:
        :param virtual_ypoly:
        :param use_scalar_vtk:
        :return:
        '''
        xpoly_dim = self.xpoly.dims()[0]
        ypoly_dim = self.ypoly.dims()[0]

        ft_dist = compute_kernel_2(virtual_xpoly, self.ypoly)
        bt_dist = compute_kernel_2(self.xpoly, virtual_ypoly)

        if use_scalar_vtk:
            scalar_dist = compute_kernel_2(self.xscalar, self.yscalar)
            self.m_matrix = (1 / self.T) * af.exp(-(ft_dist + bt_dist) / (4 * self.T) - scalar_dist / self.T)
        else:
            self.m_matrix = (1 / self.T) * af.exp(-(ft_dist + bt_dist) / (4 * self.T))
        self.m_outliers_row = (1 / self.T_init) * af.exp(-(
                af.sum(af.pow(virtual_ypoly - af.tile(self.ycentroid, ypoly_dim, 1), 2), 1) + af.sum(
            af.pow(self.ypoly - af.tile(self.ycentroid, ypoly_dim, 1), 2), 1)) / (4 * self.T_init))
        self.m_outliers_col = (1 / self.T_init) * af.exp(-(
                af.sum(af.pow(virtual_xpoly - af.tile(self.xcentroid, xpoly_dim, 1), 2), 1) + af.sum(
            af.pow(self.xpoly - af.tile(self.xcentroid, xpoly_dim, 1), 2), 1)) / (4 * self.T_init))

        af.eval(self.m_matrix)
        af.eval(self.m_outliers_row)
        af.eval(self.m_outliers_col)

    def normalize_it_m(self):
        '''
        Iteratively normalize the rows and columns of the soft assignment matrix:
        . m_matrix      - soft assignment matrix
        . xsize         - number of source points + 1 (outliers) (should be the line)
        . ysize         - number of target points + 1 (outliers) (should be the column)
        :param m_matrix:
        :param m_outliers_row:
        :param m_outliers_col:
        :return:
        '''
        norm_threshold = 0.05
        norm_maxit = 10
        norm_it = 0
        xpoly_dim = self.m_matrix.dims()[0]
        ypoly_dim = self.m_matrix.dims()[1]

        while True:
            # --- Row normalization - -------------------------------------------
            sumx = af.sum(self.m_matrix, 1) + self.m_outliers_col
            self.m_matrix = self.m_matrix / af.tile(sumx, 1, ypoly_dim)
            self.m_outliers_col = self.m_outliers_col / sumx

            # --- Column normalization - ----------------------------------------
            sumy = af.sum(self.m_matrix, 0) + af.transpose(self.m_outliers_row)
            self.m_matrix = self.m_matrix / af.tile(sumy, xpoly_dim, 1)
            self.m_outliers_row = self.m_outliers_row / af.transpose(sumy)
            err = (af.matmul(sumx - 1, sumx - 1, af.MATPROP.TRANS) + af.matmul(sumy - 1, sumy - 1, af.MATPROP.NONE,
                                                                               af.MATPROP.TRANS)) / (
                          xpoly_dim + ypoly_dim)

            af.eval(sumy)
            af.eval(sumx)
            af.eval(self.m_matrix)
            af.eval(self.m_outliers_col)
            af.eval(self.m_outliers_row)
            af.eval(err)

            if err[0, 0].scalar() < norm_threshold or norm_it >= norm_maxit:
                return
            norm_it += 1

    def normalize_m(self):
        '''
        Normalize the rows the soft assignment matrix:
        :param m_matrix: soft assignment matrix
        :return:
        '''
        ypoly_dim = self.m_matrix.dims()[1]
        sumx = af.sum(self.m_matrix, 1)
        self.m_matrix = self.m_matrix / af.tile(sumx, 1, ypoly_dim)
        af.eval(self.m_matrix)

    def threshold_m(self, threshold):
        '''
        threshold the matrix_m according to the value of the sum of each col/row:
        xsize         - number of source points + 1 (outliers) (should be the line)
        ysize         - number of target points + 1 (outliers) (should be the column)
        :param m_matrix: soft assignment matrix
        :param threshold:
        :return:
        '''
        sumx = af.sum(self.m_matrix, 1)
        mask = sumx > threshold
        self.m_matrix = self.m_matrix * af.tile(mask, 1, self.m_matrix.dims()[1])
        sumy = af.sum(self.m_matrix, 0)
        mask = sumy > threshold
        self.m_matrix = self.m_matrix * af.tile(mask, self.m_matrix.dims()[0], 1)
        af.eval(self.m_matrix)

    def update_virtual(self, poly, m_matrix):
        '''
        Compute virtual_poly
        :param poly: target points
        :param m_matrix: soft assignment matrix
        :return: virtualypoly, transformed target points
        '''
        pts_dim = poly.dims()[1]
        sumx = af.sum(m_matrix, 1)
        af.eval(sumx)
        virtual_poly = af.matmul(m_matrix, poly) / af.tile(sumx, 1, pts_dim)
        af.eval(virtual_poly)
        return virtual_poly

    def computeTPS_QR(self, poly, virtual_poly, lambda1, lambda2, sigma=1):
        '''
        Compute the TPS transformation with QR decomposition
        # TODO sigma currently not used
        :param poly: source points
        :param virtual_poly: transformed poly
        :param lambda1: non linear weight
        :param lambda2: affine weight
        :param K: Kernet of the TPS
        :param d: Affine part of the TPS
        :param c: Non linear part of the TPS
        :param sigma: stiffness of the plate (1 by default)
        :return:
        '''
        N = poly.dims()[0]
        D = poly.dims()[1]
        ones = af.constant(1, N, dtype=f32)

        dist = compute_kernel(poly)

        # create kernel
        K = -af.sqrt(dist)
        af.eval(K)

        # QR decomposition
        # create source point array for AF with extra "1" column
        S = af.join(1, ones, poly)
        af.eval(S)

        # create target point array for AF
        T = af.join(1, ones, virtual_poly)
        af.eval(T)

        # create QR matrices in AF (tau is not used)
        q, r, tau = af.qr(S)

        # Still need to extract Q1, Q2 and R
        q1 = q[:, 0:D + 1]  # size is [N][D+1]
        q2 = q[:, D + 1:N]  # size is [N][N - D - 1]
        R = r[0:D + 1, :]  # size is [D + 1, D + 1]

        # create some matrices to compute c and d
        gamma = af.matmul(af.inverse(
            af.matmul(af.matmul(q2, K, af.MATPROP.TRANS), q2) + lambda1 * af.identity(N - D - 1, N - D - 1, dtype=f32)),
            af.matmul(q2, T, af.MATPROP.TRANS))
        c = af.matmul(q2, gamma)

        # d = inv(R) * q1' * (y-K*q2*gamma);
        # d = matmul(inverse(R), matmul(q1, (T - matmul(K, c)), AF_MAT_TRANS));
        # with regularization using lambda2
        d = af.matmul(af.inverse(af.matmul(R, R, af.MATPROP.TRANS) + lambda2 * af.identity(D + 1, D + 1, dtype=f32)),
                      af.matmul(af.transpose(R), af.matmul(af.transpose(q1), (T - af.matmul(K, c)))) - af.matmul(R, R,
                                                                                                                 af.MATPROP.TRANS)) + af.identity(
            D + 1, D + 1, dtype=f32)

        af.eval(K)
        af.eval(c)
        af.eval(d)

        return K, d, c

    def warp_QR(self, poly, K, d, c):
        '''
        Apply the TPS transformation from the QR decomposition
        :param poly:
        :param K:
        :param d:
        :param c:
        :return:
        '''
        N = poly.dims()[0]
        ones = af.constant(1, N, dtype=f32)
        #  create source point array for AF with extra "1" column
        S = af.join(1, ones, poly)
        output = af.matmul(S, d) + af.matmul(K, c)
        virtual_poly = output[:, 1:]
        af.eval(virtual_poly)
        return virtual_poly

    def update(self):
        for res in range(len(self.passband)):
            xpoly_res = self.xpoly.copy()
            ypoly_res = self.ypoly.copy()

            if self.passband[res] != 1:
                xpoly_smooth_vtk = convert_af_to_vtk(self.xpoly_vtk, self.xpoly[self.lm_size:self.xpoints, :])
                ypoly_smooth_vtk = convert_af_to_vtk(self.ypoly_vtk, self.ypoly[self.lm_size:self.ypoints, :])
                xpoly_smooth_vtk = smooth_polydata(xpoly_smooth_vtk, self.passband[res])
                ypoly_smooth_vtk = smooth_polydata(ypoly_smooth_vtk, self.passband[res])
                xpoly_res = convert_vtk_to_af(xpoly_smooth_vtk)
                ypoly_res = convert_vtk_to_af(ypoly_smooth_vtk)
                if self.xlm_vtk and self.ylm_vtk:
                    xpoly_res = af.join(0, self.xlm, xpoly_res)
                    ypoly_res = af.join(0, self.ylm, ypoly_res)

            if res == 0:
                virtual_xpoly = xpoly_res.copy()
                virtual_ypoly = ypoly_res.copy()

            for it in range(math.floor(self.nbiter / len(self.passband))):
                lambda1 = self.lambda1_init * self.xpoints * self.T
                lambda2 = self.lambda2_init * self.xpoints * self.T
                print(
                    "res: {}, iter: {}, T: {:5.4f}, lambda1: {:5.2f}, lambda2: {:5.2f}".format(res, it, self.T, lambda1,
                                                                                               lambda2))

                for i in range(self.perT_maxit):
                    self.compute_m(virtual_xpoly, virtual_ypoly, use_scalar_vtk=self.use_scalar_vtk)
                    if self.xlm_vtk and self.ylm_vtk:
                        self.m_matrix[0:self.lm_size, :] = af.identity(self.lm_size, self.ypoints)
                        self.m_matrix[:, 0:self.lm_size] = af.identity(self.xpoints, self.lm_size)
                        self.m_outliers_row[0:self.lm_size] = 0
                        self.m_outliers_col[0:self.lm_size] = 0

                    # check if nan or + / -inf in the m_matrix (bad mapping)
                    if af.sum(af.isnan(self.m_matrix) + af.isinf(self.m_matrix)) > 1:
                        print("---------------------------------------------")
                        print("    EXIT_FAILURE")
                        print("    NaN or -/+inf were found in the m_matrix")
                        print("    Program has been interrupted")
                        print("---------------------------------------------")
                        return virtual_xpoly, virtual_ypoly

                    if self.iterative_norm:
                        self.normalize_it_m()
                    else:
                        self.normalize_m()

                    if self.threshold:
                        self.threshold_m(self.threshold)

                    virtual_ypoly = self.update_virtual(ypoly_res, self.m_matrix)
                    virtual_xpoly = self.update_virtual(xpoly_res, af.transpose(self.m_matrix))

                    self.K_ft, self.d_ft, self.c_ft = self.computeTPS_QR(xpoly_res, virtual_ypoly, lambda1, lambda2)
                    self.K_bt, self.d_bt, self.c_bt = self.computeTPS_QR(ypoly_res, virtual_xpoly, lambda1, lambda2)

                    virtual_xpoly = self.warp_QR(xpoly_res, self.K_ft, self.d_ft, self.c_ft)
                    virtual_ypoly = self.warp_QR(ypoly_res, self.K_bt, self.d_bt, self.c_bt)

                # monitor distance to agreement between the meshes
                # compute average change over the last 5 iterations
                # xdta = compute_mean_dist(self.ypoly, virtual_xpoly)
                # ydta = compute_mean_dist(self.xpoly, virtual_ypoly)
                # print("     DTA, forward: {:5.4f}, backward: {:5.4f}".format(xdta, ydta))
                # self.metric_watcher.append((xdta+ydta)/2)
                # if len(self.metric_watcher) == self.watcher_size:
                #     avg_metric = sum(self.metric_watcher) / len(self.metric_watcher)
                #     avg_diff = np.sqrt(np.sum(np.square(np.array(self.metric_watcher)-avg_metric))/self.watcher_size)
                #     print("     CONV: {:7.6f}".format(avg_diff))
                #     del self.metric_watcher[0]
                self.T *= self.anneal_rate
        return virtual_xpoly, virtual_ypoly
