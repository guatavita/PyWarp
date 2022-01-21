# Created by Bastien Rigaud at 21/01/2022
# Bastien Rigaud, PhD
# Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099
# Campus de Beaulieu, Université de Rennes 1
# 35042 Rennes, FRANCE
# bastien.rigaud@univ-rennes1.fr
# Description:

from IOTools.IOTools import *

class build_model(object):
    def __init__(self):
        self.data_path = {}
        self.processors = []

    def set_processors(self, processors):
        self.processors = processors

    def load_data(self, input_features):
        for key in self.data_path.keys():
            filepath = self.data_path.get(key)
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

def main():
    deformable_model = build_model()
    input_features = {
        'xpoly_path': r'C:\Bastien\sTPSRPM\examples\homer_3000pts.vtk',
        'ypoly_path': r'C:\Bastien\sTPSRPM\examples\homer_3000pts_transformed_remeshed.vtk',
    }
    deformable_model.set_processors([])
    deformable_model.load_data(input_features)


if __name__ == '__main__':
    main()
