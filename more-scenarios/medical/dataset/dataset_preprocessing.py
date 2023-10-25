import os
import sys
import argparse
import numpy as np
import shutil
from multiprocessing import Pool
import multiprocessing as mp

sys.path.append(os.path.abspath('.'))
from dataset.dataset_transform import ImageCropper
from dataset_functional import load_case_from_list_of_files, resample_and_normalize
from util.tools import *

def prepeocessing(case_identifier,data_dict, output_dir,overwrite_existing):

    print(case_identifier)
    if len(data_dict["label_path"]) != 0:
        output_dir = join(output_dir,"image_labeled")
    else:
        output_dir = join(output_dir,"image_unLabel")
    maybe_mkdir_p(output_dir)
    if len(data_dict["label_path"])==0: 
        seg=None
    else:
        seg=data_dict["label_path"]

    if overwrite_existing \
            or (not os.path.isfile(os.path.join(output_dir, "%s.npz" % case_identifier))
                or not os.path.isfile(os.path.join(output_dir, "%s.pkl" % case_identifier))):
        try:  
            transpose_forward = [0, 1, 2]
            # 非零值裁切
            data, seg, properties = load_case_from_list_of_files(data_dict["image_path"], seg)
            data, seg, properties = ImageCropper.crop(data, properties, seg)
            
            #调整朝向 RAS
            data = data.transpose((0, *[i + 1 for i in transpose_forward]))
            if seg is not None:
                seg = seg.transpose((0, *[i + 1 for i in transpose_forward]))
            # 重采样，调整spacing[2.5,0.8,0.8]
            data, seg, properties = resample_and_normalize(data, [2.5,0.8,0.8],properties, seg)
            # 堆叠图像
            if seg is not None:
                all_data = np.vstack((data, seg)).astype(np.float32)
            else: 
                all_data = data.astype(np.float32)
            # 保存图像文件.npz和属性变换文件.pkl
            np.savez_compressed(os.path.join(output_dir, "%s.npz" % case_identifier), data=all_data)
            with open(os.path.join(output_dir, "%s.pkl" % case_identifier), 'wb') as f:
                pickle.dump(properties, f)
        except Exception as e:
            print("Exception in", case_identifier, ":")
            print(e)
            raise e
        
def collect_preproced_dataset_info(processed_data_root, save_datainfo_json_name):
    dataset_info = load_json(save_datainfo_json_name)
    for item_name,item_lists in dataset_info.items():
        for case_identifier,data_dict in item_lists.items():
            if item_name != "unlabeled":
                center_dir_name = "image_labeled"
            else:
                center_dir_name = "image_unLabel"
            data_dict["precessed_image_npz"] = os.path.join(processed_data_root,center_dir_name,case_identifier+".npz")
            data_dict["precessed_image_pkl"] = os.path.join(processed_data_root,center_dir_name,case_identifier+".pkl")
            if not isfile(data_dict["precessed_image_npz"]) and isfile(data_dict["precessed_image_pkl"]):
                print(f"{case_identifier} was wrong!")
    save_json(dataset_info, save_datainfo_json_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-di", "--dataset_info", type=str, default="/home/fly/datasets/flare22/dataset_info.json",
                        help="this is a original file-path about dataset-information")
    parser.add_argument("-pod", "--preprocessing_out_dir", type=str, default="/home/fly/datasets/flare22/preprocessing_out_dir",
                        help="this is a original file-path about dataset-information")
    parser.add_argument("-tf", type=int, required=False, default=8,
                        help="Number of processes used for preprocessing the full resolution data of the 2D U-Net and "
                             "3D U-Net. Don't overdo it or you will run out of RAM")
    parser.add_argument("--verify_dataset_integrity", required=False, default=False, action="store_true",
                        help="set this flag to check the dataset integrity. This is useful and should be done once for "
                             "each dataset!")
    parser.add_argument("--overwrite", required=False, default=False, action="store_true",
                        help="restruct preprocessing results,it will remove '--preprocessing_out_dir' when this is true"
                             )
    args = parser.parse_args()
    if args.verify_dataset_integrity:
        #verify_dataset_integrity(join(nnUNet_raw_data, task_name))
        pass
    args.dataset_info = os.path.abspath(args.dataset_info)

    dataset_json = load_json(args.dataset_info)
    maybe_mkdir_p(args.preprocessing_out_dir)

    if args.overwrite and isdir(args.preprocessing_out_dir):
        shutil.rmtree(args.preprocessing_out_dir)
        maybe_mkdir_p(args.preprocessing_out_dir)

    args.preprocessing_out_dir = os.path.abspath(args.preprocessing_out_dir)
    list_of_args = []
    
    for _,item_lists in dataset_json.items():
        for case_identifier,data_dict in item_lists.items():
            list_of_args.append((case_identifier,data_dict,args.preprocessing_out_dir, args.overwrite))
    #prepeocessing(*(list_of_args[1]))
    print("Number of processers: ", mp.cpu_count())  
    if args.tf > mp.cpu_count():
        args.tf = mp.cpu_count()
    p = Pool(args.tf)
    p.starmap_async(prepeocessing, list_of_args)
    p.close()
    p.join()
    
    collect_preproced_dataset_info(args.preprocessing_out_dir, args.dataset_info)

    
if __name__ == "__main__":
    main()
    #collect_preproced_dataset_info("/home/grozta/Desktop/Laboratory/semi-supervised/UniMatch/more-scenarios/medical/dataset/flare/preprocessing_out_dir","/home/grozta/Desktop/Laboratory/semi-supervised/UniMatch/more-scenarios/medical/dataset/flare/dataset_info.json")