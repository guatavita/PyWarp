# Created by Bastien Rigaud at 21/01/2022
# Bastien Rigaud, PhD
# Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099
# Campus de Beaulieu, Université de Rennes 1
# 35042 Rennes, FRANCE
# bastien.rigaud@univ-rennes1.fr
# Description:

from IOTools.IOTools import *
from Processors.Processors import *
from CostFunctions.CostFunctions import *

class build_model(object):
    def __init__(self):
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
            temp_loader = PolydataReaderWriter(filepath=filepath)
            input_features[key.replace('_path', '')] = temp_loader.ImportPolydata()

    def pre_process(self, input_features):
        for processor in self.processors:
            print('Performing pre process {}'.format(processor))
            input_features = processor.pre_process(input_features=input_features)
        return input_features

    def post_process(self, input_features):
        for processor in self.processors[::-1]:  # In reverse order now
            print('Performing post process {}'.format(processor))
            input_features = processor.post_process(input_features=input_features)
        return input_features

    def run_cost_function(self, input_features):
        print('Performing cost function {}'.format(self.cost_function))
        input_features = self.cost_function.parse(input_features=input_features)
        return input_features


def main():

    # define model
    deformable_model = build_model()
    input_features = {
        'xpoly_path': r'C:\Bastien\sTPSRPM\examples\homer_3000pts.vtk',
        'ypoly_path': r'C:\Bastien\sTPSRPM\examples\homer_3000pts_transformed_remeshed.vtk',
    }
    deformable_model.set_processors([
        ACVD_resampling(input_keys=('xpoly', 'ypoly'), output_keys=('xpoly', 'ypoly'), np_points=(2000, 2000))
    ])
    deformable_model.set_cost_functions([
        stps_rpm(xpoly_key='xpoly', ypoly_key='ypoly')
    ])

    # build model
    deformable_model.load_data(input_features)
    deformable_model.pre_process()
    deformable_model.run_cost_function()
    deformable_model.post_process()



if __name__ == '__main__':
    main()
