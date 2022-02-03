# Created by Bastien Rigaud at 21/01/2022
# Bastien Rigaud, PhD
# Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099
# Campus de Beaulieu, Université de Rennes 1
# 35042 Rennes, FRANCE
# bastien.rigaud@univ-rennes1.fr
# Description:
import time

from IOTools.IOTools import *
from Processors.Processors import *
from CostFunctions.CostFunctions import *


class BuildModel(object):
    def __init__(self, dataloader=DataReaderWriter):
        self.dataloader = dataloader
        self.processors = []
        self.cost_function = None

    def set_processors(self, processors):
        self.processors = processors

    def set_cost_functions(self, cost_function):
        self.cost_function = cost_function

    def load_data(self, input_features):
        list_path = list(input_features.keys())
        for key in list_path:
            filepath = input_features.get(key)
            temp_loader = self.dataloader(filepath=filepath)
            input_features[key.replace('_path', '')] = temp_loader.import_data()

    def pre_process(self, input_features):
        for processor in self.processors:
            print('Performing pre process {}'.format(processor))
            start_time = time.time()
            input_features = processor.pre_process(input_features=input_features)
            print("     Elapsed time: {:5.2f} seconds".format(time.time() - start_time))
        return input_features

    def post_process(self, input_features):
        for processor in self.processors[::-1]:  # In reverse order now
            print('Performing post process {}'.format(processor))
            start_time = time.time()
            input_features = processor.post_process(input_features=input_features)
            print("     Elapsed time: {:5.2f} seconds".format(time.time() - start_time))
        return input_features

    def run_cost_function(self, input_features):
        print('Performing cost function {}'.format(self.cost_function))
        start_time = time.time()
        input_features = self.cost_function.parse(input_features=input_features)
        print("     Elapsed time: {:5.2f} seconds".format(time.time() - start_time))
        return input_features


def compute_multi_organs():
    # define model
    deformable_model = BuildModel(dataloader=ImageReaderWriter)
    input_features = {
        'fixed_bladder_path': r'C:\Data\Data_test\Vessie_ext_0.nii.gz',
        'moving_bladder_path': r'C:\Data\Data_test\Vessie_ext_1.nii.gz',
        'fixed_prostate_path': r'C:\Data\Data_test\Prostate_0.nii.gz',
        'moving_prostate_path': r'C:\Data\Data_test\Prostate_1.nii.gz',
        'fixed_rectum_path': r'C:\Data\Data_test\Rectum_ext_0.nii.gz',
        'moving_rectum_path': r'C:\Data\Data_test\Rectum_ext_1.nii.gz',
    }
    deformable_model.set_processors([
        GetSITKInfo(input_keys=('fixed_bladder', 'fixed_prostate', 'fixed_rectum',
                                'moving_bladder', 'moving_prostate', 'moving_rectum')),
        SimplifyMask(input_keys=('fixed_bladder', 'fixed_prostate', 'fixed_rectum',
                                 'moving_bladder', 'moving_prostate', 'moving_rectum'),
                     output_keys=('fixed_bladder', 'fixed_prostate', 'fixed_rectum',
                                  'moving_bladder', 'moving_prostate', 'moving_rectum'),
                     type_keys=('opening', 'opening', 'opening', 'opening', 'opening', 'opening',),
                     radius_keys=(2, 2, 2, 2, 2, 2,)),
        ConvertMaskToPoly(input_keys=('fixed_bladder', 'fixed_prostate', 'fixed_rectum',
                                      'moving_bladder', 'moving_prostate', 'moving_rectum'),
                          output_keys=('fpoly_bladder', 'fpoly_prostate', 'fpoly_rectum',
                                       'mpoly_bladder', 'mpoly_prostate', 'mpoly_rectum')),
        SITKToNumpy(input_keys=('fixed_bladder', 'fixed_prostate', 'fixed_rectum',
                                'moving_bladder', 'moving_prostate', 'moving_rectum'),
                    output_keys=('fixed_bladder', 'fixed_prostate', 'fixed_rectum',
                                 'moving_bladder', 'moving_prostate', 'moving_rectum')),
        ACVDResampling(input_keys=('fpoly_bladder', 'fpoly_prostate', 'fpoly_rectum',
                                   'mpoly_bladder', 'mpoly_prostate', 'mpoly_rectum'),
                       output_keys=('fpoly_bladder', 'fpoly_prostate', 'fpoly_rectum',
                                    'mpoly_bladder', 'mpoly_prostate', 'mpoly_rectum'),
                       np_points=(750, 500, 750, 750, 500, 750,)),
        JoinPoly(input_key_list=['fpoly_bladder', 'fpoly_prostate', 'fpoly_rectum'], output_key='xpoly',
                 use_scalar=True),
        JoinPoly(input_key_list=['mpoly_bladder', 'mpoly_prostate', 'mpoly_rectum'], output_key='ypoly',
                 use_scalar=True),
        CreateDVF(reference_keys=('xpoly', 'ypoly',), deformed_keys=('ft_poly', 'bt_poly',),
                  output_keys=('ft_dvf', 'bt_dvf',)),
        DistanceBasedMetrics(reference_keys=('xpoly', 'ypoly',), pre_process_keys=('ypoly', 'xpoly',),
                             post_process_keys=('bt_poly', 'ft_poly',), paired=False),
        ZNormPoly(input_keys=('xpoly', 'ypoly'), output_keys=('xpoly', 'ypoly'),
                  post_process_keys=('xpoly', 'ypoly')),
        ZNormPoly(input_keys=(), output_keys=('ft_poly', 'bt_poly'), post_process_keys=('ft_poly', 'bt_poly')),
        CopyKey(input_keys=('xpoly_centroid', 'ypoly_centroid', 'xpoly_scale', 'ypoly_scale'),
                output_keys=('bt_poly_centroid', 'ft_poly_centroid', 'bt_poly_scale', 'ft_poly_scale'))
    ])
    deformable_model.set_cost_functions(
        STPSRPM(xpoly_key='xpoly', ypoly_key='ypoly', use_scalar_vtk=True, passband=[0.01, 0.1, 1])
    )

    # build model
    deformable_model.load_data(input_features)
    deformable_model.pre_process(input_features)
    deformable_model.run_cost_function(input_features)
    deformable_model.post_process(input_features)
    plot_vtk(input_features['ft_dvf'], input_features['ypoly'])


def compute_tubular():
    # define model
    deformable_model = BuildModel(dataloader=ImageReaderWriter)
    input_features = {
        'fixed_rectum_path': r'C:\Data\Data_test\Rectum_ext_0.nii.gz',
        'moving_rectum_path': r'C:\Data\Data_test\Rectum_ext_1.nii.gz',
    }
    deformable_model.set_processors([
        GetSITKInfo(input_keys=('fixed_rectum', 'moving_rectum')),
        SimplifyMask(input_keys=('fixed_rectum', 'moving_rectum'),
                     output_keys=('fixed_rectum', 'moving_rectum'),
                     type_keys=('opening', 'opening',),
                     radius_keys=(2, 2,)),
        ConvertMaskToPoly(input_keys=('fixed_rectum', 'moving_rectum'),
                          output_keys=('fpoly_rectum', 'mpoly_rectum')),
        SITKToNumpy(input_keys=('fixed_rectum', 'moving_rectum'),
                    output_keys=('fixed_rectum', 'moving_rectum')),
        ACVDResampling(input_keys=('fpoly_rectum', 'mpoly_rectum'),
                       output_keys=('xpoly', 'ypoly'),
                       np_points=(1000, 1000,)),
        ExtractCenterline(input_keys=('fixed_rectum', 'moving_rectum',),
                          output_keys=('fixed_centerline', 'moving_centerline',)),
        ComputeLaplacianCorrespondence(input_keys=('fixed_rectum', 'moving_rectum',),
                                       centerline_keys=('fixed_centerline', 'moving_centerline',),
                                       output_keys=('fixed_correspondence', 'moving_correspondence',),
                                       spacing_keys=('fixed_rectum_spacing', 'moving_rectum_spacing',),
                                       dilate_centerline=False),
        CenterlineToPolydata(input_keys=('fixed_centerline', 'moving_centerline',),
                             output_keys=('fpoly_centerline', 'mpoly_centerline',),
                             origin_keys=('fixed_rectum_origin', 'moving_rectum_origin'),
                             spacing_keys=('fixed_rectum_spacing', 'moving_rectum_spacing')),
        CenterlineProjection(input_keys=('xpoly', 'ypoly',),
                             centerline_keys=('fpoly_centerline', 'mpoly_centerline',),
                             correspondence_keys=('fixed_correspondence', 'moving_correspondence',),
                             output_keys=('xpoly', 'ypoly',),
                             origin_keys=('fixed_rectum_origin', 'moving_rectum_origin',),
                             spacing_keys=('fixed_rectum_spacing', 'moving_rectum_spacing',)),
        CreateDVF(reference_keys=('fpoly_centerline',), deformed_keys=('mpoly_centerline',),
                  output_keys=('centerline_dvf',), run_pre_process=True),
        CreateDVF(reference_keys=('xpoly', 'ypoly',), deformed_keys=('ft_poly', 'bt_poly',),
                  output_keys=('ft_dvf', 'bt_dvf',), run_post_process=True),
        DistanceBasedMetrics(reference_keys=('xpoly', 'ypoly'), pre_process_keys=('ypoly', 'xpoly'),
                             post_process_keys=('bt_poly', 'ft_poly'), paired=False),
        GetZNormParameters(input_keys=('xpoly', 'ypoly'), centroid_keys=('xpoly_centroid', 'ypoly_centroid'),
                           scale_keys=('xpoly_scale', 'ypoly_scale')),
        ZNormPoly(input_keys=('xpoly', 'ypoly', 'fpoly_centerline', 'mpoly_centerline'),
                  output_keys=('xpoly', 'ypoly', 'fpoly_centerline', 'mpoly_centerline'),
                  post_process_keys=('xpoly', 'ypoly', 'fpoly_centerline', 'mpoly_centerline'),
                  centroid_keys=('xpoly_centroid', 'ypoly_centroid', 'xpoly_centroid', 'ypoly_centroid'),
                  scale_keys=('xpoly_scale', 'ypoly_scale', 'xpoly_scale', 'ypoly_scale')),
        ZNormPoly(input_keys=(), output_keys=('ft_poly', 'bt_poly'), post_process_keys=('ft_poly', 'bt_poly'),
                  centroid_keys=('ypoly_centroid', 'xpoly_centroid'),
                  scale_keys=('ypoly_scale', 'xpoly_scale')),
        CopyKey(input_keys=('xpoly_centroid', 'ypoly_centroid', 'xpoly_scale', 'ypoly_scale'),
                output_keys=('bt_poly_centroid', 'ft_poly_centroid', 'bt_poly_scale', 'ft_poly_scale'))
    ])
    deformable_model.set_cost_functions(
        STPSRPM(xpoly_key='xpoly', ypoly_key='ypoly', xlm_key='fpoly_centerline', ylm_key='mpoly_centerline',
                use_scalar_vtk=True, scalars_name='Length', passband=[0.01, 0.1, 1])
    )

    # build model
    deformable_model.load_data(input_features)
    deformable_model.pre_process(input_features)
    deformable_model.run_cost_function(input_features)
    deformable_model.post_process(input_features)
    plot_vtk(input_features['ft_dvf'], input_features['ypoly'])


# TODO finite element model using unstructured structure
# TODO work on getting the scalars to the rectum polydata
# TODO select scalars in the STPSRPM ? (or define a function to set the active scalar of each polydata)

if __name__ == '__main__':
    # compute_multi_organs()
    compute_tubular()
