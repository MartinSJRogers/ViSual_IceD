# -*- coding: utf-8 -*-
"""
Normalise and range normalise SAR input image.

@author: Martin Rogers: marrog@bas.ac.uk
"""

import numpy as np
from osgeo import gdal
from PIL import Image


def normalise_sar(sar_array_in):
    """
    Normalise SAR image to [-1,1]

    Parameters
    ----------
    sar_array_in : TYPE: array
        DESCRIPTION: SAR image resized to 80 m resolution.

    Returns
    -------
    array_out_sar : TYPE: Array
        DESCRIPTION: Normalised SAR image.

    """

    normalized_sar = (sar_array_in - np.nanmin(sar_array_in)) / \
        (np.nanmax(sar_array_in) - np.nanmin(sar_array_in))
    array_out_sar = (2*normalized_sar) - 1
    return array_out_sar


def resize_array_extract_HH_band(arr_sar, image_old_res):
    """
    Resize array to 80 m resolution and extract only HH band.

    Parameters
    ----------
    sar_array_in : TYPE: array
        DESCRIPTION: input Sar image to resize.

    Returns
    -------
    array_out_sar : TYPE: Array
        DESCRIPTION: Resized, single band HH SAR image.
    """

    old_res_sar = image_old_res

    # Remove HV band if image contains it
    if arr_sar.ndim == 2:
        image_sar = Image.fromarray(arr_sar)
        old_height_sar = arr_sar.shape[0]
        old_width_sar = arr_sar.shape[1]
    elif arr_sar.ndim == 3:
        image_sar = Image.fromarray(arr_sar[0, :, :])
        old_height_sar = arr_sar.shape[1]
        old_width_sar = arr_sar.shape[2]

    # Resize image with 80 m resolution pixels
    new_res_sar = 80
    new_height_sar = np.floor(
        (old_height_sar*old_res_sar)/new_res_sar).astype(np.int)
    new_width_sar = np.floor(
        (old_width_sar*old_res_sar)/new_res_sar).astype(np.int)

    # Use default resample interpolation function: bicubic.
    image_resized_sar = image_sar.resize((new_width_sar, new_height_sar),
                                         resample=Image.BILINEAR)
    array_resized_sar = np.asarray(image_resized_sar)

    # Normalise resized array to [-1, 1]
    normalised_sar = normalise_sar(array_resized_sar)

    return normalised_sar


def extract_coords(sar_im, date, MODIS_fp):
    """
    Method called from visualiced.py to resize and normalise SAR image and
    extract bounding coordinates so that concurrent MODIS image
    can be extracted.

    Parameters
    ----------
    im : TYPE@ String
        DESCRIPTION: Filename of input SAR image.
    date : TYPE: String
        DESCRIPTION: Date of image aquisition.
    MODIS_fp : TYPE: String
        DESCRIPTION: Directory to store corresponding MODIS image.

    Returns
    -------
    resized_array : TYPE: numpy array
        DESCRIPTION: Normalised and resized SAR image.
    coords : TYPE: text file object.
        DESCRIPTION: Text file with coordinates of SAR image to extract
        corresponding MODIS image.

    """
    coords = []

    ds = gdal.Open(sar_im)
    lon_ul, res, b, lat_ul, d, e = ds.GetGeoTransform()
    arr = ds.ReadAsArray().astype(np.float32)

    # Resize and normalise array.
    resized_array = resize_array_extract_HH_band(arr, res)
    if arr.ndim == 2:
        lon_br = lon_ul + (res*arr.shape[1])
        lat_br = lat_ul - (res*arr.shape[0])
    elif arr.ndim == 3:
        lon_br = lon_ul + (res*arr.shape[2])
        lat_br = lat_ul - (res*arr.shape[1])

    # Need date in format yyyy-mm-dd for MODIS extraction
    d1 = str(date[:4])
    d2 = str(date[4:6])
    d3 = str(date[6:8])
    date_formatted = d1+'-'+d2+'-'+d3

    fn_out = MODIS_fp + date + '_vis.tiff'
    # Add information on SAR image and MODIS fn to new line in coords.txt file.
    coord_temp = [date_formatted, fn_out, lon_ul, lat_ul, lon_br, lat_br]
    coords.append(coord_temp)
    np.savetxt("coords.txt", coords,
               delimiter=",",
               fmt='% s')

    return resized_array, coords
