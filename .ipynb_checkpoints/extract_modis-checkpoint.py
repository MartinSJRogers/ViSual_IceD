# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:19:39 2023

@author: marrog
"""
import subprocess
import os


def create_modis_image():

    subprocess.run(['/data/hpcdata/users/marrog/fuse3/full_pipeline_all/MODIS_im_extraction.sh'])


