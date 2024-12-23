import os
import SimpleITK as sitk
from nipype.interfaces import niftyreg
from multiprocessing import Pool
import multiprocessing
from itertools import repeat

def call_reg_sitk(fix_img_path, flo_img_path, res_img_path):
    fix_img = sitk.ReadImage(fix_img_path, sitk.sitkFloat32)
    flo_img = sitk.ReadImage(flo_img_path, sitk.sitkFloat32)
    initial_transform = sitk.CenteredTransformInitializer(fix_img, flo_img,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    # set similarity metric
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50) # BSpline2
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(0.01)
    # set interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)
    # set optimizer settings
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, 
                                                      numberOfIterations=200,
                                                      convergenceMinimumValue=1e-6, 
                                                      convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    # setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[1])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    # execute the registration
    final_transform = registration_method.Execute(sitk.Cast(fix_img, sitk.sitkFloat32),
                                                  sitk.Cast(flo_img, sitk.sitkFloat32))
    print("Final metric value: {0}".format(registration_method.GetMetricValue()))
    # resample with final transformation; 0.0: default value
    moving_registered = sitk.Resample(flo_img, fix_img, final_transform, 
                                      sitk.sitkLinear, 0.0, flo_img.GetPixelID())
    print("Optimizer stop condition: {0}".format(registration_method.GetOptimizerStopConditionDescription()))
    print("Iteration: {0}".format(registration_method.GetOptimizerIteration()))
    sitk.WriteImage(moving_registered, res_img_path)

def test_aladin(fix_img_path, flo_img_path, res_img_path):
    node = niftyreg.RegAladin()
    node.inputs.ref_file = fix_img_path
    node.inputs.flo_file = flo_img_path
    node.inputs.res_file = res_img_path
    print("cmd:", node.cmdline)
    node.run()
    # cmd = "reg_aladin -flo " + flo_img_path + " -ref " + fix_img_path + " -res " + out_img_path
    # node.cmdline
    # os.system("reg_aladin -ref {} -flo {} -rigOnly -interp {} -nac -pv 25 -pi 25 "
    #           "-aff /tmp/aff.txt -res {} -maxit 6 -pad {} >/dev/null 2>&1".format(
    #           fix_img_path, flo_img_path, 1, res_img_path, 0))

def call_reg_aladin(fix_img, flo_img, res_img, inaff=None, aff=None, pad=0, interp=1):
    if inaff is None and aff is None:
        os.system("reg_aladin -ref {} -flo {} -rigOnly -interp {} -nac -pv 25 -pi 25 "
                  "-aff /tmp/aff.txt -res {} -maxit 6 -pad {} >/dev/null 2>&1".format(
                  fix_img, flo_img, interp, res_img, pad))
    elif inaff is None:
        os.system(
            "reg_aladin -ref {} -flo {} -rigOnly -res {} -interp {} -nac -pv 25 -pi 25 "
            "-aff {} -maxit 6 -pad {} >/dev/null 2>&1".format(fix_img, flo_img, res_img, interp, aff, pad))
    elif aff is None:
        os.system(
            "reg_aladin -ref {} -flo {} -rigOnly -res {} -interp {} -nac -pv 25 -pi 25 "
            "-aff /tmp/aff.txt -inaff {} -maxit 0 -ln 1 -lp 1 -pad {} >/dev/null 2>&1".format(
            fix_img, flo_img, res_img, interp, inaff, pad))
    else:
        print("Error input args for reg_aladin!")
    return None

def call_mri_robust_register(flo, fix, out):
    os.system(
        "mri_robust_register --mov {} --dst {} --lta {} --noinit --satit --mapmov {} --maxit 10 --epsit 1e-6 --minsize 30 --verbose 0".format(
        flo, fix, out.replace(".nii.gz", ".lta"), out)
    )

def call_mri_robust_register_4d(flo_img_list):
    out_img_list = [f.replace(".nii.gz", "_reg.nii.gz") for f in flo_img_list]
    with Pool(multiprocessing.cpu_count()) as pool:
        pool.starmap(call_mri_robust_register, zip(flo_img_list, repeat(flo_img_list[10]), out_img_list))

def split_4D_img(processed_cases_dir, patient_name, seq):
    assert seq == "EMR" or seq == "WASSR"
    base_dir = os.path.join(processed_cases_dir, patient_name+"_all")
    img = sitk.ReadImage(os.path.join(base_dir, seq+".nii"))
    # NOTE: [256, 256, 15, n] in sitkImage and [n, 15, 256, 256] in nparray
    for i in range(img.GetSize()[-1]):
        sitk.WriteImage(img[:, :, :, i], os.path.join(base_dir, seq, str(i)+".nii"))

def combine_4d_img(processed_cases_dir, patient_name, seq):
    assert seq == "EMR" or seq == "WASSR"
    base_dir = os.path.join(processed_cases_dir, patient_name+"_all")
    n = 24
    if seq == "WASSR":
        n = 26
    f_list = []
    for i in range(n):
        f_list.append(os.path.join(base_dir, seq, str(i)+"_reg.nii"))
    res = sitk.JoinSeries([sitk.ReadImage(f) for f in f_list])
    sitk.WriteImage(res, os.path.join(base_dir, seq+"_reg.nii"))

def reg_EMR(processed_cases_dir, patient_name):
    base_dir = os.path.join(processed_cases_dir, patient_name+"_all")
    M0_path = os.path.join(base_dir, "EMR", "0.nii")
    for i in range(24):
        img_path = os.path.join(base_dir, "EMR", str(i)+".nii")
        out_path = os.path.join(base_dir, "EMR", str(i)+"_reg.nii")
        if i == 0:
            sitk.WriteImage(sitk.ReadImage(img_path), out_path)
            continue
        call_reg_sitk(M0_path, img_path, out_path)
    combine_4d_img(processed_cases_dir, patient_name, "EMR")
    sitk.WriteImage(sitk.ReadImage(M0_path), os.path.join(base_dir, "M0.nii"))

def reg_WASSR(processed_cases_dir, patient_name):
    split_4D_img(processed_cases_dir, patient_name, "WASSR")
    base_dir = os.path.join(processed_cases_dir, patient_name+"_all")
    M0_path = os.path.join(base_dir, "EMR", "0.nii")
    for i in range(26):
        img_path = os.path.join(base_dir, "WASSR", str(i)+".nii")
        out_path = os.path.join(base_dir, "WASSR", str(i)+"_reg.nii")
        call_reg_sitk(M0_path, img_path, out_path)
    combine_4d_img(processed_cases_dir, patient_name, "WASSR")

def reg_T1T2(processed_cases_dir, patient_name):
    base_dir = os.path.join(processed_cases_dir, patient_name+"_all")
    M0_path = os.path.join(base_dir, "EMR", "0.nii")
    T1_path = os.path.join(base_dir, "T1_map.nii")
    T1_reg_path = os.path.join(base_dir, "T1_map_reg.nii")
    T2_path = os.path.join(base_dir, "T2_map.nii")
    T2_reg_path = os.path.join(base_dir, "T2_map_reg.nii")
    test_aladin(M0_path, T1_path, T1_reg_path)
    # test_aladin(M0_path, T2_path, T2_reg_path)

def reg_HighRes(processed_cases_dir, patient_name):
    base_dir = os.path.join(processed_cases_dir, patient_name+"_all")
    M0_path = os.path.join(base_dir, "EMR", "0.nii")
    HighRes_path = os.path.join(base_dir, "HighRes.nii")
    HighRes_reg_path = os.path.join(base_dir, "HighRes_reg.nii")
    call_reg_sitk(M0_path, HighRes_path, HighRes_reg_path)

def register(processed_cases_dir, patient_name):
    # split_4D_img(processed_cases_dir, patient_name, "EMR")
    # reg_EMR(processed_cases_dir, patient_name)
    # reg_WASSR(processed_cases_dir, patient_name)
    reg_T1T2(processed_cases_dir, patient_name)
    # reg_HighRes(processed_cases_dir, patient_name)
