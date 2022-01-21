# Created by Bastien Rigaud at 21/01/2022
# Bastien Rigaud, PhD
# Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099
# Campus de Beaulieu, Université de Rennes 1
# 35042 Rennes, FRANCE
# bastien.rigaud@univ-rennes1.fr
# Description:

import vtk
from vtk.util import numpy_support
import arrayfire as af
import math

backends = af.get_available_backends()
if 'cuda' in backends:
    af.set_backend('cuda')
elif 'opencl' in backends:
    af.set_backend('opencl')
else:
    af.set_backend('cpu')

f32 = af.Dtype.f32

def ConvertVTKtoAF(polydata, af_array, D = 3):
    N = polydata.GetNumberOfPoints()
    points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
    af_array = af.array(N, D, points);
    af.eval(af_array)
    # TODO eval points and do we return af_array?

class stps_rpm(object):

    def __init__(self, xpoly, ypoly, lambda1_init=0.01, lambda2_init=1, t_init=0.5, t_final=0.001, anneal_rate=0.93, threshold=0.000001,
                 scalarvtk=False, xlm=None, ylm=None, passband=1, centroid=False, double_norm=True):
        """
        :param xpoly: source input as a VTK polydata
        :param ypoly: target input as a VTK polydata
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
        :param double_norm: normalization of the matrix m (False/True = single/double; default: True)
        """
        self.D = 3
        self.xpoly = xpoly
        self.ypoly = ypoly
        self.xpoints = xpoly.GetNumberOfPoints()
        self.ypoints = ypoly.GetNumberOfPoints()
        perT_maxit = 2
        ita = 0
        lambda1 = lambda1_init
        lambda2 = lambda2_init;
        T = t_init;
        nbiter = round(math.log(t_final / t_init) / math.log(anneal_rate));
        lm_size = xlm.GetNumberOfPoints();
        if xlm.GetNumberOfPoints() != ylm.GetNumberOfPoints():
            raise ValueError("Provided landmarks does not have the same size")
        ft_output = None
        bt_output = None
        ft_vectorfield = None
        bt_vectorfield = None
        self.allocate_data()

    def allocate_data(self):

        xpoly = af.constant(0, self.xpoints, self.D, dtype=f32)
        ypoly = af.constant(0, self.ypoints, self.D, dtype=f32)

        ConvertVTKtoAF(self.xpoly, xpoly)
        ConvertVTKtoAF(self.ypoly, ypoly)
        #
        # vtkSmartPointer < vtkDoubleArray > xscalar_vtk = vtkSmartPointer < vtkDoubleArray >::New();
        # vtkSmartPointer < vtkDoubleArray > yscalar_vtk = vtkSmartPointer < vtkDoubleArray >::New();
        # af::array
        # xscalar, yscalar;
        #
        # if (scalarflag > 0) {
        # std::
        #     cout << "VTK scalars to ArrayFire array [...]" << std::endl;
        # xscalar_vtk = vtkDoubleArray::SafeDownCast(xpoly_vtk->GetPointData()->GetScalars());
        # yscalar_vtk = vtkDoubleArray::SafeDownCast(ypoly_vtk->GetPointData()->GetScalars());
        # xscalar = constant(0, xpoints, xscalar_vtk->GetNumberOfComponents(), f32);
        # yscalar = constant(0, ypoints, yscalar_vtk->GetNumberOfComponents(), f32);
        # ConvertScalarstoAF(xscalar_vtk, xscalar);
        # ConvertScalarstoAF(yscalar_vtk, yscalar);
        # }
        #
        # af::array
        # xlm, ylm;
        #
        # if (lm_flag) {
        # std::
        #     cout << "VTK landmarks to ArrayFire array [...]" << std::endl;
        # ConvertVTKtoAF(xlm_vtk, xlm);
        # ConvertVTKtoAF(ylm_vtk, ylm);
        # // join
        # landmark
        # with xpoly between[0:lm_size]
        # xpoly = join(0, xlm, xpoly);
        # ypoly = join(0, ylm, ypoly);
        # // increase
        # size
        # xpoints += lm_size;
        # ypoints += lm_size;
        #
        # // update
        # scalar
        # with dummy scalars for the landmark ( not influenced)
        # if (scalarflag > 0) {
        # xscalar = join(0, constant(0, lm_size, xscalar_vtk->GetNumberOfComponents(), f32), xscalar);
        # yscalar = join(0, constant(0, lm_size, yscalar_vtk->GetNumberOfComponents(), f32), yscalar);
        # }
        # }
        #
        # std::cout << "af::array allocation [...]" << std::endl;
        # af::array
        # m_matrix = constant(0, xpoints, ypoints, f32);
        # af::array
        # m_outliers_row = constant(0, ypoints, f32);
        # af::array
        # m_outliers_col = constant(0, xpoints, f32);
        #
        # af::array
        # K_ft = constant(0, xpoints, xpoints, f32);
        # af::array
        # c_ft = constant(0, xpoints, D + 1, f32);
        # af::array
        # d_ft = constant(0, D + 1, D + 1, f32);
        #
        # af::array
        # K_bt = constant(0, ypoints, ypoints, f32);
        # af::array
        # c_bt = constant(0, ypoints, D + 1, f32);
        # af::array
        # d_bt = constant(0, D + 1, D + 1, f32);
        #
        # std::cout << "Mesh normalization [...]" << std::endl;
        # af::array
        # xcentroid = constant(0, 1, D, f32);
        # af::array
        # ycentroid = constant(0, 1, D, f32);
        # af::array
        # xscale = constant(1, 1, f32);
        # af::array
        # yscale = constant(1, 1, f32);

    def update(self):
        xxx =1

