"""
@author: Martin Rogers: marrog@bas.ac.uk

Split training and validation data for ViSual_IceD training.
"""
import os
import numpy as np
from osgeo import gdal
from PIL import Image


def read_file_list(filelist):
    """
    Read in text file list of images.
    Parameters
    ----------
    filelist : object of text file
        DESCRIPTION. Object of text file of file names.

    Returns
    -------
    filenames : List
        DESCRIPTION. File names in list

    """

    pfile = open(filelist)
    filenames = pfile.readlines()
    pfile.close()
    filenames = [f.strip() for f in filenames]
    return filenames


def split_pair_names(filenames, base_dir):
    """
    Split filenames by column to get seperate lists for SAR, MODIS and
    refreence image.
    Parameters
    ----------
    filenames : File Object
        DESCRIPTION. Object of filename text file.
    base_dir : String
        DESCRIPTION. Base directory location

    Returns
    -------
    filenames : List
        DESCRIPTION. Filenames in list

    """
    filenames = [c.split('\t') for c in filenames]  # Split columns by tab.
    filenames = [(os.path.join(base_dir, c[0]), os.path.join(base_dir, c[1]),
                  os.path.join(base_dir, c[2])) for c in filenames]
    return filenames


class DataParser():
    """
    Constructor class. Set defaults hyper-paremeters including batch size,
    training and validation data split and number of epochs.
    """

    def __init__(self, batch_size_train, training_image_fp,
                 training_image_lst):

        self.train_file = os.path.join(str(training_image_fp),
                                       str(training_image_lst))
        self.train_data_dir = str(training_image_fp)
        self.training_pairs = read_file_list(self.train_file)

        self.samples = split_pair_names(self.training_pairs,
                                        self.train_data_dir)

        # Find number of sample pairs (len), then random shuffle.
        self.n_samples = len(self.training_pairs)
        self.all_ids = range(self.n_samples)
        np.random.shuffle(list(self.all_ids))

        # Split training and validation data 80/20.
        train_split = 0.8
        self.training_ids = self.all_ids[:int(train_split * len(self.training_pairs))]
        self.validation_ids = self.all_ids[int(train_split * len(self.training_pairs)):]

        # Divide number of training images by batch size. Assert remainder = 0.
        self.batch_size_train = batch_size_train
        assert len(self.training_ids) % batch_size_train == 0
        self.steps_per_epoch = len(self.training_ids)/batch_size_train

        # Divide number of validation images by 20. Assert remainder = 0.
        assert len(self.validation_ids) % (batch_size_train*2) == 0
        self.validation_steps = len(self.validation_ids)/(batch_size_train*2)

    def get_batch(self, batch):
        """
        Method called by generator object in main.py to get images
        corresponding to id numbers.

        Parameters
        ----------
        batch : TYPE: List
            DESCRIPTION: List of IDs of training or validation images to use.

        Returns
        -------
        images_vis : TYPE: Array
            DESCRIPTION. MODIS training image.
        images_sar : TYPE: Array
            DESCRIPTION: SAR training image.
        edgemaps : TYPE: Array
            DESCRIPTION: Corresponding reference image.
        filenames : TYPE: String
            DESCRIPTION: Filename for training and reference images.

        """

        filenames = []
        images_vis = []
        images_sar = []
        reference_ims = []

        for idx, b in enumerate(batch):

            # Open and preprocess training images
            raster_vis = gdal.Open(self.samples[b][0]).ReadAsArray()
            raster_temp = np.swapaxes(raster_vis, 0, 1)
            raster_norm_vis = np.swapaxes(raster_temp, 1, 2)

            raster_sar = gdal.Open(self.samples[b][1]).ReadAsArray()
            ras_sar = np.expand_dims(raster_sar, 2)
            sar_three_band = np.concatenate((ras_sar, ras_sar, ras_sar),
                                            axis=2)

            # Open corresponding reference image.
            ref_arr = gdal.Open(self.samples[b][2]).ReadAsArray()
            ref_image = Image.fromarray(ref_arr)

            # Resize reference image
            ref_image_resized = ref_image.resize((240, 240),
                                                 resample=Image.NEAREST)
            array_resized_ref = np.asarray(ref_image_resized)
            formatted_ref = np.expand_dims(array_resized_ref, 2)

            # Append all training and reference images per batch to an array.
            images_vis.append(raster_norm_vis)
            images_sar.append(sar_three_band)
            reference_ims.append(formatted_ref)
            filenames.append(self.samples[b])

        images_vis_arr = np.asarray(images_vis)
        images_sar_arr = np.asarray(images_sar)
        reference_ims_arr = np.asarray(reference_ims)

        # Export nump arrays of training and reference images.
        return images_vis_arr, images_sar_arr, reference_ims_arr, filenames
