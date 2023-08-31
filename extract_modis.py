# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:19:39 2023

@author: marrog
"""
import subprocess
import os

def create_modis_image(shell_script):

    subprocess.run([str(shell_script)])


