import os
import sys
import argparse
import numpy as np
import shutil
from multiprocessing import Pool
import multiprocessing as mp

from skimage.transform import resize
from collections import OrderedDict
from scipy.ndimage.interpolation import map_coordinates
from batchgenerators.augmentations.utils import resize_segmentation

sys.path.append(os.path.abspath('.'))
from dataset.dataset_crop import ImageCropper, load_case_from_list_of_files
from util.tools import *

RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD = 3  # determines what threshold to use for resampling the low resolution axis
# separately (with NN)

def get_do_separate_z(spacing, anisotropy_threshold=RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis

def resample_data_or_seg(data, new_shape, is_seg, axis=None, order=3, do_separate_z=False, order_z=0):
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param is_seg:
    :param axis:
    :param order:
    :param do_separate_z:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == len(data.shape) - 1
    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_separate_z:
            print("separate z, order in z is", order_z, "order inplane is", order)
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []
            for c in range(data.shape[0]):
                reshaped_data = []
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_data.append(resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                    elif axis == 1:
                        reshaped_data.append(resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                    else:
                        reshaped_data.append(resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                reshaped_data = np.stack(reshaped_data, axis)
                if shape[axis] != new_shape[axis]:

                    # The following few lines are blatantly copied and modified from sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final_data.append(map_coordinates(reshaped_data, coord_map, order=order_z,
                                                                   mode='nearest')[None].astype(dtype_data))
                    else:
                        unique_labels = np.unique(reshaped_data)
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        for i, cl in enumerate(unique_labels):
                            reshaped_multihot = np.round(
                                map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,
                                                mode='nearest'))
                            reshaped[reshaped_multihot > 0.5] = cl
                        reshaped_final_data.append(reshaped[None].astype(dtype_data))
                else:
                    reshaped_final_data.append(reshaped_data[None].astype(dtype_data))
            reshaped_final_data = np.vstack(reshaped_final_data)
        else:
            print("no separate z, order", order)
            reshaped = []
            for c in range(data.shape[0]):
                reshaped.append(resize_fn(data[c], new_shape, order, **kwargs)[None].astype(dtype_data))
            reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        print("no resampling necessary")
        return data


def resample_patient(data, seg, original_spacing, target_spacing, order_data=3, order_seg=0, force_separate_z=False,
                     order_z_data=0, order_z_seg=0,
                     separate_z_anisotropy_threshold=RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD):
    """
    :param data:
    :param seg:
    :param original_spacing:
    :param target_spacing:
    :param order_data:
    :param order_seg:
    :param force_separate_z: if None then we dynamically decide how to resample along z, if True/False then always
    /never resample along z separately
    :param order_z_seg: only applies if do_separate_z is True
    :param order_z_data: only applies if do_separate_z is True
    :param separate_z_anisotropy_threshold: if max_spacing > separate_z_anisotropy_threshold * min_spacing (per axis)
    then resample along lowres axis with order_z_data/order_z_seg instead of order_data/order_seg

    :return:
    """
    assert not ((data is None) and (seg is None))
    if data is not None:
        assert len(data.shape) == 4, "data must be c x y z"
    if seg is not None:
        assert len(seg.shape) == 4, "seg must be c x y z"

    if data is not None:
        shape = np.array(data[0].shape)
    else:
        shape = np.array(seg[0].shape)
    new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(int)

    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(original_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(original_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(original_spacing)
        elif get_do_separate_z(target_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(target_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            # every axis has the spacing, this should never happen, why is this code here?
            do_separate_z = False
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
        else:
            pass

    if data is not None:
        data_reshaped = resample_data_or_seg(data, new_shape, False, axis, order_data, do_separate_z,
                                             order_z=order_z_data)
    else:
        data_reshaped = None
    if seg is not None:
        seg_reshaped = resample_data_or_seg(seg, new_shape, True, axis, order_seg, do_separate_z, order_z=order_z_seg)
    else:
        seg_reshaped = None
    return data_reshaped, seg_reshaped

def resample_and_normalize(data, target_spacing, properties, seg=None, force_separate_z=None):
        """
        data and seg must already have been transposed by transpose_forward. properties are the un-transposed values
        (spacing etc)
        :param data:
        :param target_spacing:
        :param properties:
        :param seg:
        :param force_separate_z:
        :return:
        """

        # target_spacing is already transposed, properties["original_spacing"] is not so we need to transpose it!
        # data, seg are already transposed. Double check this using the properties
        original_spacing_transposed = np.array(properties["original_spacing"])
        before = {
            'spacing': properties["original_spacing"],
            'spacing_transposed': original_spacing_transposed,
            'data.shape (data is transposed)': data.shape
        }

        # remove nans
        data[np.isnan(data)] = 0
        resample_separate_z_anisotropy_threshold = 3
        resample_order_data = 3
        resample_order_seg = 1
        data, seg = resample_patient(data, seg, np.array(original_spacing_transposed), target_spacing,
                                     resample_order_data, resample_order_seg,
                                     force_separate_z=force_separate_z, order_z_data=0, order_z_seg=0,
                                     separate_z_anisotropy_threshold=resample_separate_z_anisotropy_threshold)
        after = {
            'spacing': target_spacing,
            'data.shape (data is resampled)': data.shape
        }
        print("before:", before, "\nafter: ", after, "\n")

        if seg is not None:  # hippocampus 243 has one voxel with -2 as label. wtf?
            seg[seg < -1] = 0

        properties["size_after_resampling"] = data[0].shape
        properties["spacing_after_resampling"] = target_spacing

        return data, seg, properties


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
            transpose_backward = [0, 1, 2]
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-di", "--dataset_info", type=str, default="dataset/flare/dataset_info.json",
                        help="this is a original file-path about dataset-information")
    parser.add_argument("-pod", "--preprocessing_out_dir", type=str, default="dataset/flare/preprocessing_out_dir",
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
    
    dataset_json = load_json(args.dataset_info)
    maybe_mkdir_p(args.preprocessing_out_dir)

    if args.overwrite and isdir(args.preprocessing_out_dir):
        shutil.rmtree(args.preprocessing_out_dir)
        maybe_mkdir_p(args.preprocessing_out_dir)

    
    list_of_args = []
    
    for _,item_lists in dataset_json.items():
        for case_identifier,data_dict in item_lists.items():
            list_of_args.append((case_identifier,data_dict,args.preprocessing_out_dir, args.overwrite))
    #prepeocessing(*(list_of_args[60]))
    print("Number of processers: ", mp.cpu_count())   
    p = Pool(args.tf)
    p.starmap_async(prepeocessing, list_of_args)
    p.close()
    p.join()
    
if __name__ == "__main__":
    main()