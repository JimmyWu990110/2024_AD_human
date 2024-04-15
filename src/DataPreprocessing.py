import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk

def formalize_par_rec(new_cases_dir, patient_name):
    """
    All par/rec files should be named as "[patient_name]_[seq_id].par/rec"
    """
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
    output_dir = os.path.join(processed_cases_dir, patient_name+"_all", patient_name)
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
    return

def convert_selected_par_rec_to_nifti(new_cases_dir, processed_cases_dir, patient_name):
    print("******** convert selected par-rec files to nii files in [1_raw_output] ********")
    df_all = pd.read_excel(os.path.join(processed_cases_dir, patient_name+"_all", patient_name, 
                                        "all_scan_info.xlsx"), engine ='openpyxl')
    output_dir = os.path.join(processed_cases_dir, patient_name+"_all", patient_name,
                              "1_raw_output", patient_name) 
    # to avoid generating duplicate nii files by dcm2niix
    os.makedirs(output_dir, exist_ok=False) 
    df = df_all[df_all['Patient_id'] == patient_name] # select data only for this patient
    sequence_not_used = []
    for seq_idx in range(len(df)):
        protocol = df.iloc[seq_idx]['Protocol'].lower()
        sequence_id = df.iloc[seq_idx]['Sequence_id']
        if ('mprage1.1mm' in protocol) or \
        (('mprage' in protocol or 't1' in protocol) and 'pre' in protocol):
            print("T1w: ", sequence_id, protocol)
            image_suffix = 'T1'
            image_name = patient_name + '_' + image_suffix
            target_path = os.path.join(new_cases_dir, patient_name, sequence_id+".rec")
            os.system('dcm2niix -f ' + image_name + ' -o ' + output_dir + ' '  + target_path)
        elif 'mpragegd' in protocol or \
        (('mprage' in protocol or 't1' in protocol) and 'post' in protocol):
            print("T1c: ", sequence_id, protocol)
            image_suffix = 'T1c'
            image_name = patient_name + '_' + image_suffix
            target_path = os.path.join(new_cases_dir, patient_name, sequence_id+".rec")
            os.system('dcm2niix -f ' + image_name + ' -o ' + output_dir + ' '  + target_path)
        elif 'de/t2' in protocol or 't2w_ax' in protocol:
            image_suffix = 'T2'
            print("T2w: ", sequence_id, protocol)
            image_name = patient_name + '_' + image_suffix
            target_path = os.path.join(new_cases_dir, patient_name, sequence_id+".rec")
            os.system('dcm2niix -f ' + image_name + ' -o ' + output_dir + ' '  + target_path)
        elif 'flair_s2' in protocol: # use S2 rather than CS_4
            image_suffix = 'Flair'
            print("FLAIR: ", sequence_id, protocol)
            image_name = patient_name + '_' + image_suffix
            target_path = os.path.join(new_cases_dir, patient_name, sequence_id+".rec")
            os.system('dcm2niix -f ' + image_name + ' -o ' + output_dir + ' '  + target_path)
        elif 'apt' in protocol:
            image_suffix = sequence_id.split('_')[-2] + '_' + 'apt'
            print("APTw: ", sequence_id, protocol)
            image_name = patient_name + '_' + image_suffix
            target_path = os.path.join(new_cases_dir, patient_name, sequence_id+".rec")
            os.system('dcm2niix -f ' + image_name + ' -o ' + output_dir + ' '  + target_path)
        elif 'highres' in protocol:
            image_suffix = sequence_id.split('_')[-2] + '_' + 'highres'
            print("High Resolution: ", sequence_id, protocol)
            image_name = patient_name + '_' + image_suffix
            target_path = os.path.join(new_cases_dir, patient_name, sequence_id+".rec")
            os.system('dcm2niix -f ' + image_name + ' -o ' + output_dir + ' '  + target_path)
        else:
            sequence_not_used.append([sequence_id, protocol])
    print("Sequences not used:")
    for i in range(len(sequence_not_used)):
        print(sequence_not_used[i])
    return

def sepcify_sequence_ids(MTR_1p5uT_id, WASSR_EMR_id, MTR_2uT_id):
    seq_dict = {"MTR_1p5uT_id":MTR_1p5uT_id, "WASSR_EMR_id":WASSR_EMR_id, 
                "MTR_2uT_id":MTR_2uT_id}
    return seq_dict







        