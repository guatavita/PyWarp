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

from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
from PlotVTK.PlotVTK import plot_vtk


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


def main():
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
        ConvertMaskToPoly(input_keys=('fixed_bladder', 'fixed_prostate', 'fixed_rectum',
                                      'moving_bladder', 'moving_prostate', 'moving_rectum'),
                          output_keys=('fpoly_bladder', 'fpoly_prostate', 'fpoly_rectum',
                                       'mpoly_bladder', 'mpoly_prostate', 'mpoly_rectum')),
        GetSITKInfo(input_keys=('fixed_bladder', 'fixed_prostate', 'fixed_rectum',
                                'moving_bladder', 'moving_prostate', 'moving_rectum')),
        SITKToNumpy(input_keys=('fixed_bladder', 'fixed_prostate', 'fixed_rectum',
                                'moving_bladder', 'moving_prostate', 'moving_rectum'),
                    output_keys=('fixed_bladder', 'fixed_prostate', 'fixed_rectum',
                                 'moving_bladder', 'moving_prostate', 'moving_rectum')),
        ACVDResampling(input_keys=('fpoly_bladder', 'fpoly_prostate', 'fpoly_rectum',
                                   'mpoly_bladder', 'mpoly_prostate', 'mpoly_rectum'),
                       output_keys=('fpoly_bladder', 'fpoly_prostate', 'fpoly_rectum',
                                    'mpoly_bladder', 'mpoly_prostate', 'mpoly_rectum'),
                       np_points=(1500, 1000, 1500, 1500, 1000, 1500)),
        JoinPoly(input_key_list=['fpoly_bladder', 'fpoly_prostate', 'fpoly_rectum'], output_key='xpoly', use_scalar=True),
        JoinPoly(input_key_list=['mpoly_bladder', 'mpoly_prostate', 'mpoly_rectum'], output_key='ypoly', use_scalar=True),
        CreateDVF(reference_keys=('xpoly', 'ypoly',), deformed_keys=('ft_poly', 'bt_poly',),
                  output_keys=('ft_dvf', 'bt_dvf',)),
        ZNormPoly(input_keys=('xpoly', 'ypoly'), output_keys=('xpoly', 'ypoly'),
                  post_process_keys=('xpoly', 'ypoly')),
        ZNormPoly(input_keys=(), output_keys=('ft_poly', 'bt_poly'), post_process_keys=('ft_poly', 'bt_poly')),
        CopyKey(input_keys=('xpoly_centroid', 'ypoly_centroid', 'xpoly_scale', 'ypoly_scale'),
                output_keys=('bt_poly_centroid', 'ft_poly_centroid', 'bt_poly_scale', 'ft_poly_scale'))
    ])
    deformable_model.set_cost_functions(
        STPSRPM(xpoly_key='xpoly', ypoly_key='ypoly')
    )

    # build model
    deformable_model.load_data(input_features)
    deformable_model.pre_process(input_features)
    deformable_model.run_cost_function(input_features)
    deformable_model.post_process(input_features)
    plot_vtk([input_features['ft_dvf'], input_features['bt_dvf']])


# TODO append structues and define label scalar on it for multi organ stpsrpm
# TODO finite element model using unstructured structure
# TODO label of class per structure
# TODO find extremities of a tubular structure by computing the centroid of the two most distant regions
if __name__ == '__main__':
    main()
