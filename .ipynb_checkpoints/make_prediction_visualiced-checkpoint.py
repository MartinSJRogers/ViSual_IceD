# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:29:49 2023

@author: marrog
"""

from osgeo import gdal
from keras.models import model_from_json
import numpy as np
import rasterio


def createImage(original_array_vis, out_array, imageName, ul_x_vis, ul_y_vis):
    """
    Save ViSual_IceD output to file with same coordinate reference system as
    original MODIS image.
    Parameters
    ----------
    original_array_vis : TYPE: numpy array.
        DESCRIPTION. Original MODIS image to extract image properties from
    array : TYPE: Array
        DESCRIPTION: Array of final ViSual_IceD classification.
    imageName : TYPE: String
        DESCRIPTION: Output image name.
    ul_x_vis : TYPE: Double
        DESCRIPTION: Upper left corner x coordinate.
    ul_y_vis : TYPE: Double
        DESCRIPTION: Upper left corner Y coordinate.

    Returns
    -------
    None.

    """
    out_array = out_array[np.newaxis, ...]
    out_array = out_array.astype(np.float32)
    raster_transform = [240, 0, ul_x_vis, 0, -240, ul_y_vis]
    kwargs = ({'driver': "GTiff", 'dtype': 'float32', 'nodata': 0.0,
             'width': original_array_vis.shape[1],
             'height': original_array_vis.shape[0],
             'count': 1, 'crs': "EPSG:3031", 'transform': raster_transform})

    with rasterio.open(imageName, 'w', **kwargs) as dest:
        dest.write(out_array)


def patchImage(arr_vis, arr_sar, visfn, sarfn):
    """
    Patches MODIS and SAR image with height and width 240 and 720 pixels
    respectively.
    Stride = 60 MODIS pixels so there is an overlap of 120 pixels from
    each patch.
    The top left coordinate of every MODIS patch is found. The corresponding
    SAR patch is found using this coordinate to ensure overlap each time.

    Parameters
    ----------
    arr_vis : TYPE: numpy array
        DESCRIPTION: Original MODIS image.
    arr_sar : TYPE: numpy array
        DESCRIPTION: Original SAR image.
    visfn : TYPE: String
        DESCRIPTION: MODIS image filename.
    sarfn : TYPE: String
        DESCRIPTION: Sar image filename.

    Returns
    -------
    arrayList : TYPE: List
        DESCRIPTION. List of all patches
    top_left_pixels : TYPE: List
        DESCRIPTION : List of row and column of every top left pixel.
    ul_x_vis : TYPE: List
        DESCRIPTION: List of x coordinates of all patches.
    ul_y_vis : TYPE: List
        DESCRIPTION.List of y coordinates of all patches.

    """
    col_top_left = []
    row_top_left = []
    visList = []
    sarList = []

    # Original images to get projection info
    im_vis = gdal.Open(visfn)
    im_sar = gdal.Open(sarfn)

    # Stride
    patch_width = 60
    patch_height = 60

    # Extract coordinate and resolution information from SAR and MODIS image.
    ul_x_vis = im_vis.GetGeoTransform()[0]
    ul_y_vis = im_vis.GetGeoTransform()[3]
    ul_x_sar = im_sar.GetGeoTransform()[0]
    ul_y_sar = im_sar.GetGeoTransform()[3]
    res_x_vis = im_vis.GetGeoTransform()[1]
    res_y_vis = im_vis.GetGeoTransform()[5]
    res_x_sar = im_sar.GetGeoTransform()[1]
    res_y_sar = im_sar.GetGeoTransform()[5]

    # Duplicate x and y coordinate of top left hand corner of image. 
    # This will change in iteration    
    x_coord = int(ul_x_vis)
    y_coord = int(ul_y_vis)

    # Calculate the stride in coordinates= 60 MODIS pixels.
    dx = int(patch_width*res_x_vis)
    dy = int(patch_height*res_y_vis)

    # Calculate height and width of image in metres.
    imheight = int(arr_vis.shape[0]*res_y_vis)
    imwidth = int(arr_vis.shape[1]*res_x_vis)

    # Empty list to add coordinates of top left corner of every patch.
    crop_ul_corners = np.empty([1, 2])

    # Make 3 band sar input for parallel network
    arr_sar = np.concatenate((np.expand_dims(arr_sar, axis=2),
                                            np.expand_dims(arr_sar, axis=2),
                                            np.expand_dims(arr_sar, axis=2)), axis=2)
    
    # Run through image and generate patches.
    for y in range(0, imheight, dy):
        for x in range(0,  imwidth, dx):
            new_coord_x = x_coord+x
            new_coord_y = y_coord+y

            newRow = np.array([new_coord_x, new_coord_y])
            crop_ul_corners = np.vstack([crop_ul_corners, newRow])

            # Find the row and column numbers corresponding with the
            # cropped area upper left coordinate in MODIS image.
            col_vis_ul=np.floor(abs((abs(new_coord_x)-abs(ul_x_vis))/abs(res_x_vis))).astype(np.int32())
            row_vis_ul=np.floor(abs((abs(ul_y_vis)-abs(new_coord_y))/abs(res_y_vis))).astype(np.int32())

            # Find corresponding rows and columns for SAR image.
            col_sar_ul=np.floor(abs((abs(new_coord_x)-abs(ul_x_sar))/abs(res_x_sar))).astype(np.int32())
            row_sar_ul=np.floor(abs((abs(ul_y_sar)-abs(new_coord_y))/abs(res_y_sar))).astype(np.int32())

            # Find bottom left rows and columns for every patch.
            br_col_vis = col_vis_ul + 240
            br_row_vis = row_vis_ul + 240
            br_col_sar = col_sar_ul + 720
            br_row_sar = row_sar_ul + 720

            # Subset patches using row and column numbers.
            vis_crop = arr_vis[row_vis_ul:br_row_vis, col_vis_ul:br_col_vis, :]
            sar_crop = arr_sar[row_sar_ul:br_row_sar, col_sar_ul:br_col_sar, :]

            # Append patches and corresponding coordinates to list.
            # Remove edge patches. Only store patch pairs if SAR patch
            # measures 720 by 720 pixels.
            if sar_crop.shape[0] == 720:
                if sar_crop.shape[1] == 720:
                    visList.append(vis_crop)
                    sarList.append(sar_crop)
                    arrayList = list(zip(visList, sarList))

                    col_top_left.append(col_vis_ul)
                    row_top_left.append(row_vis_ul)
                    top_left_pixels = list(zip(col_top_left, row_top_left))

    return arrayList, top_left_pixels, ul_x_vis, ul_y_vis


def classify_patch(in_patch_arrays, model_fp, weights_fn):
    """
    Classify each patch using trained ViSual_IceD network.

    Parameters
    ----------
    in_patch_arrays : TYPE: List
        DESCRIPTION: All patches to be classified.
    model_fp : TYPE: String
        DESCRIPTION: filepath to ViSual_IceD model architecture.
    weights_fn : TYPE: String
        DESCRIPTION: Filepath to trained ViSual_IceD network weights.

    Returns
    -------
    predPatches : List
        DESCRIPTION: Stack of classified patches.

    """
    # load json and create model
    json_file = open(model_fp, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_fn)

    predPatches = []

    # Classify each patch pair and append to predPatches list.
    for patch_in in in_patch_arrays:

        image_vis = patch_in[0]
        image_sar = patch_in[1]

        image_vis_in = np.expand_dims(image_vis, axis=0)
        image_sar_in = np.expand_dims(image_sar, axis=0)

        pred = loaded_model.predict([image_vis_in, image_sar_in])
        squeeze_dims = np.squeeze(pred, axis=1)
        output = squeeze_dims[0, :, :, 1]

        predPatches.append(output)

    return predPatches




def reconstruct_patches(array_list, original_vis_arr, date, ul_x_vis, ul_y_vis, output_name):
    """
    Reconstruct patches into image with same dimensions and resolution as
    original MODIS image. 

    Parameters
    ----------
    array_list : TYPE: List
        DESCRIPTION: List of patches to be reconstructed.
    original_vis_arr : TYPE: Array
        DESCRIPTION: Original MODIS image.
    date : TYPE: String
        DESCRIPTION: Date of image aquisition.
    ul_x_vis : TYPE: Double
        DESCRIPTION: Upper left pixel x coordinate.
    ul_y_vis : TYPE: Double
        DESCRIPTION: Upper left pixel y coordinate.
    output_name : TYPE: String
        DESCRIPTION: Output filename for classified image.

    Returns
    -------
    im_reconstruct : TYPE: array
        DESCRIPTION: Classified image as numpy array.

    """

    # Create new empty array with same dimension as MODIS image.
    WHOLE_IM_HEIGHT = original_vis_arr.shape[0]
    WHOLE_IM_WIDTH = original_vis_arr.shape[1]
    im_reconstruct = np.zeros((WHOLE_IM_HEIGHT, WHOLE_IM_WIDTH))

    # Define patch height and width and stride.
    patch_height = 240
    patch_width = 240
    border = 60

    # Reconstruct image.
    for i in array_list:
        patch_in_parallel = i[0]
        x_corner = int(i[2][1])
        y_corner = int(i[2][0])

        subset_parallel = patch_in_parallel[border: -border, border:-border]
        if subset_parallel.shape[0] == 120:
            if subset_parallel.shape[1] == 120:

                im_reconstruct[x_corner+border:x_corner+patch_width-border,
                               y_corner+border:y_corner+patch_height-border] = subset_parallel

        else:
            q = 1

    createImage(original_vis_arr, im_reconstruct, output_name,
                ul_x_vis, ul_y_vis)

    return im_reconstruct


def make_pred(vis_filename, processed_vis_arr, sar_filename,
              model_filename_parallel, weights_filename_parallel,
              image_date, processed_sar_array):
    """
    Method called from visualiced.py to apply trained model.
    Returns
    -------
    outarray_visualiced : TYPE: numpy array.
        DESCRIPTION: Final ViSual_IceD classification. 

    """
    # Patch all images.
    merged_patched_arrays, upper_left_pixels, ul_x_vis, ul_y_vis = patchImage(processed_vis_arr, processed_sar_array, vis_filename, sar_filename)

    # Classify each patch using the trained ViSual_IceD network
    classified_patches = classify_patch(merged_patched_arrays,
                                        model_filename_parallel,
                                        weights_filename_parallel)

    # Combine classified patch array with coordinates of top left corner
    preds_and_pixels = list(zip(classified_patches, upper_left_pixels))

    # Reconstruct patches into image with same height and width as original.
    outarray_visualiced = reconstruct_patches(preds_and_pixels,
                                              processed_vis_arr,
                                              image_date, ul_x_vis, ul_y_vis)

    return outarray_visualiced
