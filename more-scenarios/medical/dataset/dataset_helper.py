import os
import sys
sys.path.append(os.path.abspath('.'))
import random
from collections import OrderedDict
from util.tools import maybe_mkdir_p,save_json,file_name_walk

def gen_dataset_relative_files(root,imageTr,labelTr,unlabeled,save_file_dirs_path,val_rate = 0.2):
    '''
    root:
    imageTr:
    labelTr:
    unlabel:
    save_file_dirs_path:
    val_rate:验证集占整个有标签图像的比例
    '''
    dct = OrderedDict()
    maybe_mkdir_p(save_file_dirs_path)
    image_list = file_name_walk(os.path.join(root,imageTr))
    label_list = file_name_walk(os.path.join(root,labelTr))
    unlabel_list = file_name_walk(os.path.join(root,unlabeled))

    if len(image_list) != len(label_list): 
        print('iamge cant match label! return...')
        return
    labeled_coll=OrderedDict()
    unlabel_coll=OrderedDict()
    for image ,label in zip(image_list, label_list):
        case_labeled = {"image_path":image,"label_path":label}
        case_identifier = os.path.basename(label).split('.')[0]
        labeled_coll[case_identifier]= case_labeled
    
    for image_u in unlabel_list:
        case_labeled = {"image_path":image_u,"label_path":""}
        case_identifier = os.path.basename(image_u).split('.')[0]
        unlabel_coll[case_identifier]= case_labeled
    train_counts = int(len(image_list) - len(image_list)*val_rate)

    keys_tr = random.sample(labeled_coll.keys(),train_counts)
    keys_val = [i for i in labeled_coll.keys() if i not in keys_tr]
    dct['train_case_list'] = {key:value for key,value in labeled_coll.items() if key in keys_tr}
    dct['val_case_list'] = {key:value for key,value in labeled_coll.items() if key in keys_val}
    dct['unlabeled'] = unlabel_coll
    save_json(dct, os.path.join(save_file_dirs_path, "dataset_info.json"))


if __name__ == '__main__':
    gen_dataset_relative_files("/home/fly/datasets/flare22","imagesTr","labelsTr","unlabeled","/home/fly/datasets/flare22")
