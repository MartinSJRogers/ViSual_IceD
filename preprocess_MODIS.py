# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:08:50 2023

@author: marrog
"""
from osgeo import gdal
from keras.models import model_from_json
import numpy as np
from PIL import Image
from rasterio.plot import show
import os
import glob
import rasterio

def resize_normalise_vis(vis_fn, arr_vis):
    
    #extract crs details from original image, but then use arr_vis derived from permute or none permute method
    im_vis = gdal.Open(vis_fn)
    ul_x_vis, res_x_vis, distort_x_vis, ul_y_vis, distort_y_vis, res_y_vis = im_vis.GetGeoTransform()
    arr_vis_reshape=np.swapaxes(arr_vis, 0,1)
    arr_vis_reshape=np.swapaxes(arr_vis_reshape, 1,2)
    print(np.shape(arr_vis))
    image_vis=Image.fromarray(arr_vis_reshape)
    
    old_height_vis=arr_vis.shape[1]
    old_width_vis=arr_vis.shape[2]
    old_res_vis=res_x_vis
    new_res_vis= 240

    new_height_vis=np.floor((old_height_vis*old_res_vis)/240).astype(np.int)
    new_width_vis=np.floor((old_width_vis*old_res_vis)/240).astype(np.int)
    
    image_resized=image_vis.resize((new_width_vis, new_height_vis))
    array_resized=np.asarray(image_resized)
    print(np.shape(array_resized))
    array_resized=np.swapaxes(array_resized, 1,2)
    array_resized=np.swapaxes(array_resized, 0,1)
    print(np.shape(array_resized))
    
    #normalise every MODIS band individually, then concatenate again
    normalized_input_0 = (array_resized[0, :, :]- np.amin(array_resized[0, :, :])) / (np.amax(array_resized[0, :, :]) - np.amin(array_resized[0, :, :]))
    normalised_0=(2*normalized_input_0) - 1
    normalized_input_1 = (array_resized[1, :, :]- np.amin(array_resized[1, :, :])) / (np.amax(array_resized[1, :, :]) - np.amin(array_resized[1, :, :]))
    normalised_1=(2*normalized_input_1) - 1
    normalized_input_2 = (array_resized[2, :, :]- np.amin(array_resized[2, :, :])) / (np.amax(array_resized[2, :, :]) - np.amin(array_resized[2, :, :]))
    normalised_2=(2*normalized_input_2) - 1

    array_out_vis=np.concatenate((np.expand_dims(normalised_0, axis=2), 
                                np.expand_dims(normalised_1, axis=2), 
                                    np.expand_dims(normalised_2, axis=2)), axis=2)     

    return array_out_vis 