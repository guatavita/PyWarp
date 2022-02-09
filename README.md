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
    <img src="Examples/animation_20220204_103713.gif" height=300>
    <img src="Examples/animation_20220204_104614.gif" height=300>
    <img src="Examples/output_colour.gif" height=200>
</p>

## Example 

### Example multi structures

```python
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
                       nb_points=(750, 500, 750, 750, 500, 750,)),
        JoinPoly(input_key_list=['fpoly_bladder', 'fpoly_prostate', 'fpoly_rectum'], output_key='xpoly',
                 use_scalar=True),
        JoinPoly(input_key_list=['mpoly_bladder', 'mpoly_prostate', 'mpoly_rectum'], output_key='ypoly',
                 use_scalar=True),
        CreateDVF(reference_keys=('xpoly', 'ypoly',), deformed_keys=('ft_poly', 'bt_poly',),
                  output_keys=('ft_dvf', 'bt_dvf',), run_post_process=True),
        DistanceBasedMetrics(reference_keys=('xpoly', 'ypoly',), pre_process_keys=('ypoly', 'xpoly',),
                             post_process_keys=('bt_poly', 'ft_poly',), paired=False),
        GetZNormParameters(input_keys=('xpoly', 'ypoly'), centroid_keys=('xpoly_centroid', 'ypoly_centroid'),
                           scale_keys=('xpoly_scale', 'ypoly_scale')),
        ZNormPoly(input_keys=('xpoly', 'ypoly',),
                  output_keys=('xpoly', 'ypoly',),
                  post_process_keys=('xpoly', 'ypoly',),
                  centroid_keys=('xpoly_centroid', 'ypoly_centroid',),
                  scale_keys=('xpoly_scale', 'ypoly_scale',)),
        ZNormPoly(input_keys=(), output_keys=('ft_poly', 'bt_poly'), post_process_keys=('ft_poly', 'bt_poly'),
                  centroid_keys=('ypoly_centroid', 'xpoly_centroid'), scale_keys=('ypoly_scale', 'xpoly_scale')),
        CopyKey(input_keys=('xpoly_centroid', 'ypoly_centroid', 'xpoly_scale', 'ypoly_scale'),
                output_keys=('bt_poly_centroid', 'ft_poly_centroid', 'bt_poly_scale', 'ft_poly_scale'))
    ])
    deformable_model.set_cost_functions(
        STPSRPM(xpoly_key='xpoly', ypoly_key='ypoly', use_scalar_vtk=False, passband=[0.01, 0.1, 1])
    )

    # build model
    deformable_model.load_data(input_features)
    deformable_model.pre_process(input_features)
    deformable_model.run_cost_function(input_features)
    deformable_model.post_process(input_features)
    plot_vtk(input_features['ft_dvf'], input_features['ypoly'])
```

### Example tubular structure
```python
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
                       nb_points=(1000, 1000,)),
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
```

## Dependencies

Run:
```
pip install -r requirements.txt
```

List of required libraries:
```
This framework needs ArrayFire DLL/sources and Arrayfire-python (cpu, OpenCL, or CUDA backend) https://arrayfire.com/
```


## References
- Rigaud, B., Cazoulat, G., Vedam, S., Venkatesan, A. M., Peterson, C. B., Taku, N., ... & Brock, K. K. (2020). Modeling complex deformations of the sigmoid colon between external beam radiation therapy and brachytherapy images of cervical cancer. International Journal of Radiation Oncology* Biology* Physics, 106(5), 1084-1094.
- Rigaud, B., Simon, A., Gobeli, M., Leseur, J., Duvergé, L., Williaume, D., ... & De Crevoisier, R. (2018). Statistical shape model to generate a planning library for cervical adaptive radiotherapy. IEEE transactions on medical imaging, 38(2), 406-416.

## Inspiration
[Brian Mark Anderson](https://github.com/brianmanderson)