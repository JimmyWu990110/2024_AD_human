import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk

def formalize_par_rec(new_cases_dir, patient_name):
    # All par/rec files should be named as "[patient_name]_[seq_id].par/rec"
    print("******** formalize par-rec files ********")
    file_path = os.path.join(new_cases_dir, patient_name)
    files = os.listdir(file_path)
    for f in files:
        if f.endswith("par") or f.endswith("rec"):
            prefix = patient_name.lower()
            suffix = f.split("_")[-2] + "_" + f.split("_")[-1]
            new_name = prefix + "_" + suffix
            # print(f, prefix, suffix, new_name)
            if f != new_name:
                print("rename [" + f + "] to [" + new_name + "]")
                src = os.path.join(file_path, f)
                dst = os.path.join(file_path, new_name)
                os.rename(src, dst)
    print("All par-rec file names formalized for [" + file_path + "]")
    return

def read_par(par_path, patient_name, sequence_id):
    with open(par_path, "r") as file:
        data = file.readlines()
        content_line = [patient_name, sequence_id]
        for data_line in data:
            line_parts = data_line[:-1].split(":   ") # 3 spaces
            if len(line_parts) == 2: # header info
                if 'Protocol name' in line_parts[0]:
                    content_line.append(line_parts[1])
                elif 'Examination date/time' in line_parts[0]:
                    content_line.append(line_parts[1][:10])
                elif 'Max. number of slices' in line_parts[0]:
                    content_line.append(line_parts[1])
                elif 'Scan resolution' in line_parts[0]:
                    content_line.append(line_parts[1])
                elif 'FOV (ap,fh,rl)' in line_parts[0]:
                    content_line.append(line_parts[1])
                elif 'Angulation midslice(ap,fh,rl)' in line_parts[0]:
                    content_line.append(line_parts[1])
                elif 'Off Centre midslice(ap,fh,rl)' in line_parts[0]:
                    content_line.append(line_parts[1])
    return content_line

def sort_sequence_id(sequence_ids):
    sequence_numbers = []
    for sequence_id in sequence_ids:
        sequence_numbers.append(int(sequence_id.split('_')[-2]))
    sort_index = np.argsort(sequence_numbers)
    return sort_index
    
def query_par_rec_save_info(new_cases_dir, processed_cases_dir, patient_name):
    print("******** query par-rec files and save info to Excel ********")
    output_dir = os.path.join(processed_cases_dir, patient_name+"_all")
    os.makedirs(output_dir, exist_ok=True)
    files = os.listdir(os.path.join(new_cases_dir, patient_name))
    all_scan_info = []
    for f in files:
        if f.endswith('.par'):
            info = read_par(os.path.join(new_cases_dir, patient_name, f), 
                            patient_name, f.split('.')[0])
            all_scan_info.append(info)
    all_scan_info = np.array(all_scan_info)
    sort_index = sort_sequence_id(all_scan_info[:, 1])
    all_scan_info = all_scan_info[sort_index]
    column_names = ["Patient_id", "Sequence_id", "Protocol", "Examination_date", "slices", 
                    "resolution", "FOV", "Angulation_midslice", "Off_Centre_midslice"]
    df = pd.DataFrame(data=all_scan_info, columns=column_names)
    df.to_excel(os.path.join(output_dir, 'all_scan_info.xlsx'), index=False)
    print("Sucessfully saved to [" + os.path.join(output_dir, 'all_scan_info.xlsx') + "]")

def CovertPARToNIFI(base_dir, out_dir, df):
    os.makedirs(out_dir, exist_ok=True)
    for seq_idx in range(len(df)):
        patient_id = df.iloc[seq_idx]['Patient_id']
        protocol = df.iloc[seq_idx]['Protocol']
        sequence_id = df.iloc[seq_idx]['Sequence_id']
        if 'Look Locker' in protocol:
            print("T1: ", protocol)
            image_name = '%f_%p_%t_%s'
            target_path = os.path.join(base_dir, patient_id, sequence_id + '.rec')
            output_path = os.path.join(out_dir, 'T1')
            os.makedirs(output_path, exist_ok=True)
            os.system('dcm2niix -f ' + image_name + ' -o ' + output_path + ' ' + target_path)
        elif 'T2_ME' in protocol:
            print("T2: ", protocol)
            image_name = '%f_%p_%t_%s'
            target_path = os.path.join(base_dir, patient_id, sequence_id + '.rec')
            output_path = os.path.join(out_dir, 'T2')
            os.makedirs(output_path, exist_ok=True)
            os.system('dcm2niix -f ' + image_name + ' -o ' + output_path + ' ' + target_path)
        elif 'Zspec' in protocol or 'Z_single' in protocol or 'EMR' in protocol:
            print("EMR: ", protocol)
            image_name = '%f_%p_%t_%s'
            target_path = os.path.join(base_dir, patient_id, sequence_id + '.rec')
            output_path = os.path.join(out_dir, 'EMR')
            os.makedirs(output_path, exist_ok=True)
            os.system('dcm2niix -f ' + image_name + ' -o ' + output_path + ' ' + target_path)  
        elif 'WASSR' in protocol:
            print("WASSR: ", protocol)
            image_name = '%f_%p_%t_%s'
            target_path = os.path.join(base_dir, patient_id, sequence_id + '.rec')
            output_path = os.path.join(out_dir, 'WASSR')
            os.makedirs(output_path, exist_ok=True)
            os.system('dcm2niix -f ' + image_name + ' -o ' + output_path + ' ' + target_path) 
        elif 'HighRes' in protocol:
            print("High Resolution: ", protocol)
            image_name = '%f_%p_%t_%s'
            target_path = os.path.join(base_dir, patient_id, sequence_id + '.rec')
            output_path = os.path.join(out_dir, "HighRes")
            os.makedirs(output_path, exist_ok=True)
            os.system('dcm2niix -f ' + image_name + ' -o ' + output_path + ' ' + target_path)

def get_nifti_for_fitting(new_cases_dir, processed_cases_dir, patient_name):
    out_dir = os.path.join(processed_cases_dir, patient_name+"_all")  
    df = pd.read_excel(os.path.join(processed_cases_dir, patient_name+"_all", "all_scan_info.xlsx"))     
    CovertPARToNIFI(new_cases_dir, out_dir, df) 

def set_geometry_EMR(EMR, img):
    # EMR/WASSR (4D) -> M0 (3D)
    spacing = EMR.GetSpacing()[0:3]
    origin = EMR.GetOrigin()[0:3]
    temp = np.array(EMR.GetDirection())
    temp = np.reshape(temp, [4, 4])
    temp = temp[0:3, 0:3]
    direction = temp.flatten()
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection(direction)
    return img

def check_data_helper(img, name):
    print("********", name, "********")
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()
    print("shape:", arr.shape)
    print("spacing:", spacing)
    print("origin:",  origin)
    print("direction:", direction)

def check_data(processed_cases_dir, patient_name):
    base_dir = os.path.join(processed_cases_dir, patient_name+"_all")
    M0 = sitk.ReadImage(os.path.join(base_dir, "M0.nii"), sitk.sitkFloat32)
    EMR = sitk.ReadImage(os.path.join(base_dir, "EMR_reg.nii"), sitk.sitkFloat32)
    WASSR = sitk.ReadImage(os.path.join(base_dir, "WASSR_reg.nii"), sitk.sitkFloat32)
    T1 = sitk.ReadImage(os.path.join(base_dir, "T1_map_reg.nii"), sitk.sitkFloat32)
    T2 = sitk.ReadImage(os.path.join(base_dir, "T2_map_reg.nii"), sitk.sitkFloat32)
    HighRes = sitk.ReadImage(os.path.join(base_dir, "HighRes_reg.nii"), sitk.sitkFloat32)
    check_data_helper(M0, "M0")
    check_data_helper(T1, "T1")
    check_data_helper(T2, "T2")
    check_data_helper(EMR, "EMR")
    check_data_helper(WASSR, "WASSR")
    check_data_helper(HighRes, "HighRes")



        