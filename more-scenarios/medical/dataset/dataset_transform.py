import abc
import torch
import random
import shutil
import numpy as np
from multiprocessing import Pool
from util.tools import *
from .dataset_functional import random_contrast, random_brightness_multiplicative, random_gamma, \
    load_case_from_list_of_files, crop_to_nonzero, get_case_identifier,random_gaussian_noise,random_gaussian_blur


class AbstractTransform(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, **data_dict):
        raise NotImplementedError("Abstract, so implement")

    def __repr__(self):
        ret_str = str(type(self).__name__) + "( " + ", ".join(
            [key + " = " + repr(val) for key, val in self.__dict__.items()]) + " )"

        return ret_str

class ColorJitter(AbstractTransform):
    """Randomly change the brightness, contrast, and gamma.
    Args:

    """
    def __init__(self, brightness=(0.8, 1.2), contrast=(0.7, 1.3),
                 gamma=(0.5, 1.5), p=0.5):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.gamma = self._check_input(gamma, 'gamma')
        self.p = p

    def _check_input(self, parameters, name):
        if parameters is None:
            parameters = (0.5, 2)
        elif len(parameters) != 2:
            raise ValueError(f'the length of {name} parameter must be two!')

        return parameters

    @staticmethod
    def get_params(parameters):
        if np.random.random() < 0.5 and parameters[0] < 1:
            para = np.random.uniform(parameters[0], 1)
        else:
            para = np.random.uniform(max(parameters[0], 1), parameters[1])

        return para

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        if isinstance(sample, dict):
            image, label = sample['image'], sample['label']
        else:
            image, label = sample, None
        fn_idx = torch.randperm(3)
        for fn_id in fn_idx:
            if fn_id == 0:
                image = random_contrast(image, contrast_range=self.contrast, preserve_range=True, p=1.0)
            elif fn_id == 1:
                image = random_brightness_multiplicative(image, multiplier_range=self.brightness, p=1.0)
            elif fn_id == 2:
                image = random_gamma(image, gamma_range=self.gamma, p=1.0)

        if isinstance(sample, dict):
            return {'image': image, 'label': label}
        else:
            return image

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', gamma={0}'.format(self.gamma)

        return format_string
    
class NoiseJitter(AbstractTransform):
    """Randomly change the gaussian noise and blur.
    Args:

    """
    def __init__(self, noise_sigma=(0, 0.2), blur_sigma=(0.5, 1.0), p=0.5):
        self.noise_sigma = noise_sigma
        self.blur_sigma = blur_sigma
        self.p = p

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        if isinstance(sample, dict):
            image, label = sample['image'], sample['label']
        else:
            image, label = sample, None

        if np.random.random() < 0.5:
            image = random_gaussian_noise(image, self.noise_sigma, p=1.0)
        else:
            image = random_gaussian_blur(image, self.blur_sigma, p=1.0)

        if isinstance(sample, dict):
            return {'image': image, 'label': label}
        else:
            return image


class ImageCropper(object):
    def __init__(self, num_threads, output_folder=None):
        """
        This one finds a mask of nonzero elements (must be nonzero in all modalities) and crops the image to that mask.
        In the case of BRaTS and ISLES data this results in a significant reduction in image size
        :param num_threads:
        :param output_folder: whete to store the cropped data
        :param list_of_files:
        """
        self.output_folder = output_folder
        self.num_threads = num_threads

        if self.output_folder is not None:
            maybe_mkdir_p(self.output_folder)

    @staticmethod
    def crop(data, properties, seg=None):
        shape_before = data.shape
        data, seg, bbox = crop_to_nonzero(data, seg)
        shape_after = data.shape
        print("before crop:", shape_before, "after crop:", shape_after, "spacing:",
              np.array(properties["original_spacing"]), "\n")

        properties["crop_bbox"] = bbox
        if seg is not None:
            properties['classes'] = np.unique(seg)
            seg[seg < -1] = 0

        properties["size_after_cropping"] = data[0].shape
        return data, seg, properties

    @staticmethod
    def crop_from_list_of_files(data_files, seg_file=None):
        data, seg, properties = load_case_from_list_of_files(data_files, seg_file)
        return ImageCropper.crop(data, properties, seg)

    def load_crop_save(self, case, case_identifier, overwrite_existing=False):
        try:
            print(case_identifier)
            if overwrite_existing \
                    or (not os.path.isfile(os.path.join(self.output_folder, "%s.npz" % case_identifier))
                        or not os.path.isfile(os.path.join(self.output_folder, "%s.pkl" % case_identifier))):

                data, seg, properties = self.crop_from_list_of_files(case[:-1], case[-1])

                all_data = np.vstack((data, seg))
                np.savez_compressed(os.path.join(self.output_folder, "%s.npz" % case_identifier), data=all_data)
                with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'wb') as f:
                    pickle.dump(properties, f)
        except Exception as e:
            print("Exception in", case_identifier, ":")
            print(e)
            raise e

    def get_list_of_cropped_files(self):
        return subfiles(self.output_folder, join=True, suffix=".npz")

    def get_patient_identifiers_from_cropped_files(self):
        return [i.split("/")[-1][:-4] for i in self.get_list_of_cropped_files()]

    def run_cropping(self, list_of_files, overwrite_existing=False, output_folder=None):
        """
        also copied ground truth nifti segmentation into the preprocessed folder so that we can use them for evaluation
        on the cluster
        :param list_of_files: list of list of files [[PATIENTID_TIMESTEP_0000.nii.gz], [PATIENTID_TIMESTEP_0000.nii.gz]]
        :param overwrite_existing:
        :param output_folder:
        :return:
        """
        if output_folder is not None:
            self.output_folder = output_folder

        output_folder_gt = os.path.join(self.output_folder, "gt_segmentations")
        maybe_mkdir_p(output_folder_gt)
        for j, case in enumerate(list_of_files):
            if case[-1] is not None:
                shutil.copy(case[-1], output_folder_gt)

        list_of_args = []
        for j, case in enumerate(list_of_files):
            case_identifier = get_case_identifier(case)
            list_of_args.append((case, case_identifier, overwrite_existing))

        p = Pool(self.num_threads)
        p.starmap(self.load_crop_save, list_of_args)
        p.close()
        p.join()

    def load_properties(self, case_identifier):
        with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'rb') as f:
            properties = pickle.load(f)
        return properties

    def save_properties(self, case_identifier, properties):
        with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'wb') as f:
            pickle.dump(properties, f)
