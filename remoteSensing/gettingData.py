# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:40:49 2022

@author: vedhs
"""

import ee

# Initialize the Earth Engine module.
ee.Initialize()

# Print earth engine version
print(ee.__version__)

print(ee.ImageCollection('COPERNICUS/S2_SR').filterDate('2021-01-01','2021-01-30').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20)))