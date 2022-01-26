# PyWarp, deformable mesh registration framework using Python

## Table of contents
* [General info](#general-info)
* [Example](#example)
* [Dependencies](#dependencies)
* [References](#references)

## General info
Bastien Rigaud, PhD
Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099
Campus de Beaulieu, Université de Rennes 1
35042 Rennes, FRANCE
bastien.rigaud@univ-rennes1.fr

<p align="center">
    <img src="Examples/output_colour.gif" height=200>
</p>

## Example 

```python
def main():
    deformable_model = BuildModel(dataloader=ImageReaderWriter)
    input_features = {
        'xmask_path': r'C:\Data\Data_test\Prostate.nii.gz',
        'ymask_path': r'C:\Data\Data_test\Vessie_ext.nii.gz',
    }
    deformable_model.set_processors([
        ConvertMaskToPoly(input_keys=('xmask', 'ymask'), output_keys=('xpoly', 'ypoly')),
        GetSITKInfo(input_keys=('xmask', 'ymask')),
        SITKToNumpy(input_keys=('xmask', 'ymask'), output_keys=('xmask', 'ymask')),
        ACVDResampling(input_keys=('xpoly', 'ypoly'), output_keys=('xpoly', 'ypoly'), np_points=(2000, 2000)),
        ZNormPoly(input_keys=('xpoly', 'ypoly'), output_keys=('xpoly', 'ypoly')),
    ])
    deformable_model.set_cost_functions(
        STPSRPM(xpoly_key='xpoly', ypoly_key='ypoly')
    )

    # build model
    deformable_model.load_data(input_features)
    deformable_model.pre_process(input_features)
    deformable_model.run_cost_function(input_features)
    deformable_model.post_process(input_features)
```

## Dependencies

Run:
```
pip install -r requirements.txt
```

List of required libraries:
```
This need framework relies on ArrayFire 3.6.2 (cpu or CUDA backend) https://arrayfire.com/
```


## References
- Rigaud, B., Cazoulat, G., Vedam, S., Venkatesan, A. M., Peterson, C. B., Taku, N., ... & Brock, K. K. (2020). Modeling complex deformations of the sigmoid colon between external beam radiation therapy and brachytherapy images of cervical cancer. International Journal of Radiation Oncology* Biology* Physics, 106(5), 1084-1094.
- Rigaud, B., Simon, A., Gobeli, M., Leseur, J., Duvergé, L., Williaume, D., ... & De Crevoisier, R. (2018). Statistical shape model to generate a planning library for cervical adaptive radiotherapy. IEEE transactions on medical imaging, 38(2), 406-416.

## Inspiration
[Brian Mark Anderson](https://github.com/brianmanderson)