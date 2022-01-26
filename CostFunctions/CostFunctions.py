# Created by Bastien Rigaud at 21/01/2022
# Bastien Rigaud, PhD
# Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099
# Campus de Beaulieu, Université de Rennes 1
# 35042 Rennes, FRANCE
# bastien.rigaud@univ-rennes1.fr
# Description:

import numpy as np
import vtk
from vtk.util import numpy_support
import arrayfire as af
import math

backends = af.get_available_backends()
if 'cuda' in backends:
    af.set_backend('cuda')
# elif 'opencl' in backends:
#     af.set_backend('opencl')
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


# TODO ADD documentation for each function
# TODO eval points and do we return af_array?
# TODO change input to key and load keys
# TODO add a costfunction merger to compute 2 cost functions (stpsrpm and DMR) and merge them with a lambda
# TODO put the TPS outside?
# TODO mettre en place in pipeline template pour les cost functions comme les processors

def ConvertVTKtoAF(polydata, force_float32=True):
    points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
    if force_float32:
        points = points.astype(dtype=np.float32)
    af_array = af.Array(points.ctypes.data, points.shape, points.dtype.char)
    return af_array


def compute_centroid_af(array):
    return af.sum(array, 0) / array.dims()[0]


class CostFunction(object):
    def parse(self, input_features):
        return input_features


def compute_kernel(array):
    N = array.dims()[0]
    D = array.dims()[1]
    kernel = af.constant(0, N, N, dtype=f32)
    for i in range(D):
        temp1 = af.tile(array[:, i], 1, N)
        temp2 = af.tile(af.transpose(array[:, i]), N, 1)
        kernel = kernel + af.pow(temp1 - temp2, 2)
    return kernel


def compute_kernel_2(array1, array2):
    M = array1.dims()[0]
    N = array2.dims()[0]
    D = array1.dims()[1]
    kernel = af.constant(0, M, N, dtype=f32)
    for i in range(D):
        temp1 = af.tile(array1[:, i], 1, N)
        temp2 = af.tile(af.transpose(array2[:, i]), M, 1)
        kernel = kernel + af.pow(temp1 - temp2, 2)
    return kernel


class STPSRPM(CostFunction):
    def __init__(self, xpoly_key='xpoly', ypoly_key='xpoly', ft_out_key='ft_poly', bt_out_key='bt_poly',
                 ft_dvf_key='ft_dvf', bt_dvf_key='bt_dvf', lambda1_init=0.01, lambda2_init=1, T_init=0.5,
                 T_final=0.001, anneal_rate=0.93, threshold=0.000001, scalarvtk=False, xlm_key=None, ylm_key=None,
                 passband=[1], centroid=False, iterative_norm=True):
        """
        :param xpoly_key: source input as a VTK polydata
        :param ypoly_key: target input as a VTK polydata
        :param lambda1: lambda1 init value, weight for the 'non-linear' part of the TPS
        :param lambda2: lambda2 init value, Weight for the affine part of the TPS
        :param t_init: initial temperature value
        :param t_final: final temperature value
        :param rate: annealing rate (default: 0.93)
        :param threshold: threshold the matrix-m to remove outliers (default: 0.000001)
        :param scalarvtk: scalarvtk used to cluster group of points (0/1 none/values; default: 0)
        :param xlm: vtk pts (landmarks) related to the x shape with same index ordering as lmy (use as constraint in the matrix m)
        :param ylm: vtk pts (landmarks) related to the y shape with same index ordering as lmy (use as constraint in the matrix m)
        :param passband: passband value for smooth filter (advice: 0.01x0.1x1) (default=1 /eq to no smoothing)
        :param centroid: normalize by taking in account the centroid of the shape (0/1 no/yes; default: 1)
        :param iterative_norm: normalization of the matrix m (False/True = single/iterative; default: True)
        """
        self.xpoly_key = xpoly_key
        self.ypoly_key = ypoly_key
        self.ft_out_key = ft_out_key
        self.bt_out_key = bt_out_key
        self.ft_dvf_key = ft_dvf_key
        self.bt_dvf_key = bt_dvf_key
        self.bt_dvf_key = bt_dvf_key
        self.bt_dvf_key = bt_dvf_key
        self.xlm_key = xlm_key
        self.ylm_key = ylm_key
        self.D = 3
        self.threshold = threshold
        self.scalarvtk = scalarvtk
        self.passband = passband
        self.centroid = centroid
        self.iterative_norm = iterative_norm
        self.perT_maxit = 2
        self.ita = 0
        self.lambda1_init = lambda1_init
        self.lambda2_init = lambda2_init
        self.T_init = T_init
        self.T = T_init
        self.anneal_rate = anneal_rate
        self.nbiter = round(math.log(T_final / T_init) / math.log(anneal_rate))

    def parse(self, input_features):
        _check_keys_(input_features, (self.xpoly_key, self.ypoly_key))
        self.xpoly = input_features[self.xpoly_key]
        self.ypoly = input_features[self.ypoly_key]
        self.xpoints = self.xpoly.GetNumberOfPoints()
        self.ypoints = self.ypoly.GetNumberOfPoints()

        self.xlm = input_features.get(self.xlm_key)
        self.ylm = input_features.get(self.ylm_key)

        if self.xlm and self.ylm:
            self.lm_size = self.xlm.GetNumberOfPoints()
            if self.xlm.GetNumberOfPoints() != self.ylm.GetNumberOfPoints():
                raise ValueError("Provided landmarks does not have the same size")

        self.ft_output = None
        self.bt_output = None
        self.ft_vectorfield = None
        self.bt_vectorfield = None
        self.allocate_data()
        self.update()

    def allocate_data(self):
        self.xpoly = ConvertVTKtoAF(self.xpoly, force_float32=True)
        self.ypoly = ConvertVTKtoAF(self.ypoly, force_float32=True)

        if self.scalarvtk:
            xxx = 1
            # xscalar_vtk = vtkDoubleArray::SafeDownCast(xpoly_vtk->GetPointData()->GetScalars());
            # yscalar_vtk = vtkDoubleArray::SafeDownCast(ypoly_vtk->GetPointData()->GetScalars());
            # xscalar = constant(0, xpoints, xscalar_vtk->GetNumberOfComponents(), dtype=f32);
            # yscalar = constant(0, ypoints, yscalar_vtk->GetNumberOfComponents(), dtype=f32);
            # ConvertScalarstoAF(xscalar_vtk, xscalar);
            # ConvertScalarstoAF(yscalar_vtk, yscalar);

        if self.xlm and self.ylm:
            self.xlm = ConvertVTKtoAF(self.xlm, force_float32=True)
            self.ylm = ConvertVTKtoAF(self.ylm, force_float32=True)

            # join landmark with xpoly between[0:lm_size]
            self.xpoly = af.join(0, self.xlm, self.xpoly)
            self.ypoly = af.join(0, self.ylm, self.ypoly)

            # increase size
            self.xpoints += self.lm_size
            self.ypoints += self.lm_size

            # update scalar with dummy scalars for the landmark ( not influenced)
            # if self.scalarflag:
            #     xscalar = af.join(0, af.constant(0, self.lm_size, self.xscalar_vtk.GetNumberOfComponents(), dtype=f32), self.xscalar)
            #     yscalar = af.join(0, af.constant(0, self.lm_size, self.yscalar_vtk.GetNumberOfComponents(), dtype=f32), self.yscalar)

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

    def compute_m(self, virtual_xpoly, virtual_ypoly):
        xpoly_dim = self.xpoly.dims()[0]
        ypoly_dim = self.ypoly.dims()[0]

        ft_dist = compute_kernel_2(virtual_xpoly, self.ypoly)
        bt_dist = compute_kernel_2(self.xpoly, virtual_ypoly)

        m_matrix = (1 / self.T) * af.exp(-(ft_dist + bt_dist) / (4 * self.T))
        m_outliers_row = (1 / self.T_init) * af.exp(-(
                af.sum(af.pow(virtual_ypoly - af.tile(self.ycentroid, ypoly_dim, 1), 2), 1) + af.sum(
            af.pow(self.ypoly - af.tile(self.ycentroid, ypoly_dim, 1), 2), 1)) / (4 * self.T_init))
        m_outliers_col = (1 / self.T_init) * af.exp(-(
                af.sum(af.pow(virtual_xpoly - af.tile(self.xcentroid, xpoly_dim, 1), 2), 1) + af.sum(
            af.pow(self.xpoly - af.tile(self.xcentroid, xpoly_dim, 1), 2), 1)) / (4 * self.T_init))
        return m_matrix, m_outliers_row, m_outliers_col


    def normalize_it_m(self, m_matrix, m_outliers_row, m_outliers_col):
        norm_threshold = 0.05
        norm_maxit = 10
        norm_it = 0
        xpoly_dim = m_matrix.dims()[0]
        ypoly_dim = m_matrix.dims()[1]

        while True:
            # --- Row normalization - -------------------------------------------
            sumx = af.sum(m_matrix, 1) + m_outliers_col
            m_matrix = m_matrix / af.tile(sumx, 1, ypoly_dim)
            m_outliers_col = m_outliers_col / sumx

            # --- Column normalization - ----------------------------------------
            sumy = af.sum(m_matrix, 0) + af.transpose(m_outliers_row)
            m_matrix = m_matrix / af.tile(sumy, xpoly_dim, 1)
            m_outliers_row = m_outliers_row / af.transpose(sumy)
            err = (af.matmul(sumx - 1, sumx - 1, af.MATPROP.TRANS) + af.matmul(sumy - 1, sumy - 1, af.MATPROP.NONE, af.MATPROP.TRANS)) / (xpoly_dim + ypoly_dim)

            if err[0, 0].scalar() < norm_threshold or norm_it >= norm_maxit:
                return m_matrix, m_outliers_row, m_outliers_col
            norm_it += 1

    def normalize_m(self, m_matrix):
        ypoly_dim = m_matrix.dims()[1]
        sumx = af.sum(m_matrix, 1)
        m_matrix = m_matrix / af.tile(sumx, 1, ypoly_dim)
        return m_matrix

    def threshold_m(self, m_matrix, threshold):
        '''
        threshold the matrix_m according to the value of the sum of each col/row:
        :param m_matrix: soft assignment matrix
        :param threshold:
        :return:
        xsize         - number of source points + 1 (outliers) (should be the line)
        ysize         - number of target points + 1 (outliers) (should be the column)
        '''
        sumx = af.sum(m_matrix, 1)
        mask = sumx > threshold
        m_matrix = m_matrix * af.tile(mask, 1, m_matrix.dims()[1])
        sumy = af.sum(m_matrix, 0)
        mask = sumy > threshold
        m_matrix = m_matrix * af.tile(mask, m_matrix.dims()[0], 1)
        return m_matrix

    def update_virtual(self, poly, m_matrix):
        pts_dim = poly.dims()[1]
        sumx = af.sum(m_matrix, 1)
        return af.matmul(m_matrix, poly) / af.tile(sumx, 1, pts_dim)

    def computeTPS_QR(self, poly, virtual_poly, lambda1, lambda2, sigma=1):
        '''
        Compute the TPS transformation with QR decomposition
        # TODO check if we can change sigma, currently not used
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

        # create kernel
        K = -af.sqrt(compute_kernel(poly))

        # QR decomposition
        # create source point array for AF with extra "1" column
        S = af.join(1, ones, poly)

        # create target point array for AF
        T = af.join(1, ones, virtual_poly)

        # create QR matrices in AF (tau is not used)
        q, r, tau = af.qr(S)

        # Still need to extract Q1, Q2 and R
        q1 = q[:,0:D+1] # size is [N][D+1]
        q2 = q[:,D + 1:N] # size is [N][N - D - 1]
        R = r[0:D+1,:] # size is [D + 1, D + 1]

        # create some matrices to compute c and d
        gamma = af.matmul(af.inverse(af.matmul(af.matmul(q2, K, af.MATPROP.TRANS), q2) + lambda1 * af.identity(N - D - 1, N - D - 1, dtype=f32)), af.matmul(q2, T, af.MATPROP.TRANS))
        c = af.matmul(q2, gamma)

        # d = inv(R) * q1' * (y-K*q2*gamma);
        # d = matmul(inverse(R), matmul(q1, (T - matmul(K, c)), AF_MAT_TRANS));
        # with regularization using lambda2
        d = af.matmul(af.inverse(af.matmul(R, R, af.MATPROP.TRANS) + lambda2 * af.identity(D + 1, D + 1, dtype=f32)), af.matmul(af.transpose(R), af.matmul(af.transpose(q1), (T - af.matmul(K, c)))) - af.matmul(R, R, af.MATPROP.TRANS)) + af.identity(D + 1, D + 1, dtype=f32)
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
        #TODO virtual_poly = output.cols(1, af::end)
        virtual_poly = output[:,1:]
        return virtual_poly

    def update(self):
        for res in range(len(self.passband)):
            xpoly_res = self.xpoly.copy()
            ypoly_res = self.ypoly.copy()

            if self.passband[res] != 1:
                xxx = 1
                xxx = 1
                # ConvertAFtoVTK(xpoly_smooth_vtk, xpoly(seq(lm_size, xpoints - 1), span));
                # ConvertAFtoVTK(ypoly_smooth_vtk, ypoly(seq(lm_size, ypoints - 1), span));
                # xpoly_smooth_vtk = smoothPoly(xpoly_smooth_vtk, passband[res]);
                # ypoly_smooth_vtk = smoothPoly(ypoly_smooth_vtk, passband[res]);
                #
                # ConvertVTKtoAF(xpoly_smooth_vtk, xpoly_res);
                # ConvertVTKtoAF(ypoly_smooth_vtk, ypoly_res);
                # if (lm_flag) {
                #     xpoly_res = join(0, xlm, xpoly_res);
                #     ypoly_res = join(0, ylm, ypoly_res);
                # }

            if res == 0:
                virtual_xpoly = xpoly_res.copy()
                virtual_ypoly = ypoly_res.copy()

            for it in range(math.floor(self.nbiter / len(self.passband))):
                lambda1 = self.lambda1_init * self.xpoints * self.T
                lambda2 = self.lambda2_init * self.xpoints * self.T
                print("res: {}, iter: {}; T: {}, lambda1: {}, lambda2: {}".format(res, it, self.T, lambda1, lambda2))

                for i in range(self.perT_maxit):
                    if self.scalarvtk:
                        xxx = 1
                    else:
                        m_matrix, m_outliers_row, m_outliers_col = self.compute_m(virtual_xpoly, virtual_ypoly)

                    # if (lm_flag) {
                    #     m_matrix(seq(lm_size), span) = af::identity(lm_size, ypoints);
                    #     m_matrix(span, seq(lm_size)) = af::identity(xpoints, lm_size);
                    #     m_outliers_row(seq(lm_size)) = 0;
                    #     m_outliers_col(seq(lm_size)) = 0;
                    # }

                    # check if nan or + / -inf in the m_matrix (bad mapping)
                    if af.sum(af.isnan(m_matrix) + af.isinf(m_matrix)) > 1:
                        print("---------------------------------------------")
                        print("    EXIT_FAILURE")
                        print("    NaN or -/+inf were found in the m_matrix")
                        print("    Program has been interrupted")
                        print("---------------------------------------------")
                        return

                    if self.iterative_norm:
                        m_matrix, m_outliers_row, m_outliers_col = self.normalize_it_m(m_matrix, m_outliers_row, m_outliers_col)
                    else:
                        m_matrix = self.normalize_m(m_matrix)

                    if self.threshold:
                        m_matrix = self.threshold_m(m_matrix, self.threshold)

                    virtual_ypoly = self.update_virtual(ypoly_res, m_matrix)
                    virtual_xpoly = self.update_virtual(xpoly_res, af.transpose(m_matrix))

                    self.K_ft, self.d_ft, self.c_ft = self.computeTPS_QR(xpoly_res, virtual_ypoly, lambda1, lambda2)
                    self.K_bt, self.d_bt, self.c_bt = self.computeTPS_QR(ypoly_res, virtual_xpoly, lambda1, lambda2)

                    virtual_xpoly = self.warp_QR(xpoly_res, self.K_ft, self.d_ft, self.c_ft)
                    virtual_ypoly = self.warp_QR(ypoly_res, self.K_bt, self.d_bt, self.c_bt)
                self.T *= self.anneal_rate