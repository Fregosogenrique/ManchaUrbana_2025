#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Urban Growth Analysis for Ameca, Mexico using Landsat Imagery
=============================================================

This script downloads Landsat imagery for Ameca, Mexico for the years 2014 and 2024,
processes the images to highlight urban areas, analyzes the urban growth between
these periods, and visualizes the results.

Dependencies:
- earthengine-api
- geemap
- numpy
- matplotlib
- rasterio (for local file operations)
- geopandas (for handling vector data)
"""

import ee
import geemap
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from matplotlib.colors import ListedColormap

# Set up output directory
output_dir = 'ameca_analysis'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Step 1: Initialize and authenticate Earth Engine
# -------------------------------------------------
# You need to authenticate with Earth Engine. Run this once interactively.
# This will open a web page where you can authenticate with your Google account.

try:
    ee.Initialize(project='proyectocuvallesmanchau25')
    print("Earth Engine inicializado correctamente")
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='proyectocuvallesmanchau25')
    print("Nueva autenticación realizada")

# Step 2: Define the Region of Interest (ROI) - Ameca, Mexico
# ----------------------------------------------------------
# Define the coordinates for Ameca, Mexico
# Ameca is located at approximately 20.55°N, 104.04°W

# Using a buffer around the center of Ameca to create area of interest
ameca_point = ee.Geometry.Point([-104.04, 20.55])
ameca_buffer = ameca_point.buffer(10000)  # 10km buffer around center

# Define the region of interest
roi = ameca_buffer

# Step 3: Function to get Landsat imagery for a specific year
# ----------------------------------------------------------

def get_landsat_collection(year, roi):
    """
    Get Landsat imagery for a specific year and region.
    
    For 2014: Use Landsat 8
    For 2024: Use Landsat 9 if available, otherwise Landsat 8
    
    Parameters:
    year (int): The year to get imagery for
    roi (ee.Geometry): Region of interest
    
    Returns:
    ee.Image: Median composite image for the specified year
    """
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    
    # For 2014, use Landsat 8
    if year == 2014:
        collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
            .filterBounds(roi) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUD_COVER', 20))  # Filter clouds
    
    # For 2024, use Landsat 9 if available, otherwise Landsat 8
    elif year == 2024:
        # Note: As of my knowledge cutoff, use most recent available images
        # Check if we're already in 2024, if not use available data
        current_year = datetime.datetime.now().year
        if current_year < 2024:
            # Use the most recent available data
            start_date = f'{current_year-1}-01-01'
            end_date = f'{current_year}-12-31'
            print(f"Warning: 2024 data not yet available, using {start_date} to {end_date}")
        
        # Try Landsat 9 first
        collection = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') \
            .filterBounds(roi) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUD_COVER', 20))
        
        # If no Landsat 9 images, fall back to Landsat 8
        count = collection.size().getInfo()
        if count == 0:
            print("No Landsat 9 images available, falling back to Landsat 8")
            collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                .filterBounds(roi) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUD_COVER', 20))
    
    # Scale the surface reflectance values
    collection = collection.map(lambda img: img.select(['SR_B.*']).multiply(0.0000275).add(-0.2))
    
    # Create a composite image (median)
    median_image = collection.median()
    
    return median_image

# Step 4: Download and process Landsat imagery for both years
# ----------------------------------------------------------

# Get Landsat composites for 2014 and 2024
landsat_2014 = get_landsat_collection(2014, roi)
landsat_2024 = get_landsat_collection(2024, roi)

# Function to calculate urban index (NDBI - Normalized Difference Built-up Index)
def calculate_urban_indices(image):
    """
    Calculate urban indices for the given image
    
    Parameters:
    image (ee.Image): Landsat image
    
    Returns:
    ee.Image: Image with added urban indices
    """
    # For Landsat 8/9, bands are:
    # B1=Ultra Blue, B2=Blue, B3=Green, B4=Red, B5=NIR, B6=SWIR1, B7=SWIR2
    
    # NDBI (Normalized Difference Built-up Index)
    # NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
    ndbi = image.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
    
    # NDVI (Normalized Difference Vegetation Index) for comparison
    # NDVI = (NIR - Red) / (NIR + Red)
    ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    
    # MNDWI (Modified Normalized Difference Water Index) to remove water bodies
    # MNDWI = (Green - SWIR1) / (Green + SWIR1)
    mndwi = image.normalizedDifference(['SR_B3', 'SR_B6']).rename('MNDWI')
    
    # Add indices to the original image
    return image.addBands([ndbi, ndvi, mndwi])

# Calculate urban indices for both years
landsat_2014_indices = calculate_urban_indices(landsat_2014)
landsat_2024_indices = calculate_urban_indices(landsat_2024)

# Step 5: Create urban area mask for both years
# ---------------------------------------------

def create_urban_mask(image):
    """
    Create a binary mask of urban areas
    
    Urban areas are identified where:
    - NDBI is high (built-up areas)
    - NDVI is low (less vegetation)
    - MNDWI is low (not water)
    
    Parameters:
    image (ee.Image): Image with urban indices
    
    Returns:
    ee.Image: Binary mask of urban areas
    """
    # Get the urban indices
    ndbi = image.select('NDBI')
    ndvi = image.select('NDVI')
    mndwi = image.select('MNDWI')
    
    # Create a mask for urban areas
    # Urban areas have high NDBI, low NDVI, and are not water (low MNDWI)
    urban_mask = ndbi.gt(0).And(ndvi.lt(0.2)).And(mndwi.lt(0)).rename('urban')
    
    return urban_mask

# Create urban masks for both years
urban_2014 = create_urban_mask(landsat_2014_indices)
urban_2024 = create_urban_mask(landsat_2024_indices)

# Step 6: Calculate urban growth between 2014 and 2024
# ---------------------------------------------------

# Calculate the difference between 2024 and 2014 urban areas
# 0 = Non-urban in both years
# 1 = Urban in 2014, non-urban in 2024 (urban loss, rare)
# 2 = Non-urban in 2014, urban in 2024 (urban growth)
# 3 = Urban in both years (stable urban)

urban_change = urban_2014.add(urban_2024.multiply(2))

# Step 7: Visualize the results
# ----------------------------

# True color visualization parameters
rgb_vis_params = {
    'bands': ['SR_B4', 'SR_B3', 'SR_B2'],
    'min': 0,
    'max': 0.3,
    'gamma': 1.4
}

# Urban change visualization parameters
urban_change_vis = {
    'min': 0,
    'max': 3,
    'palette': ['darkgreen', 'orange', 'red', 'darkred']
}

# Create a map for visualization
Map = geemap.Map()
Map.centerObject(roi, 12)

# Add the RGB images
Map.addLayer(landsat_2014, rgb_vis_params, '2014 True Color')
Map.addLayer(landsat_2024, rgb_vis_params, '2024 True Color', False)  # Hidden by default

# Add the urban areas
Map.addLayer(urban_2014.selfMask(), {'palette': 'red'}, '2014 Urban Areas', False)
Map.addLayer(urban_2024.selfMask(), {'palette': 'darkred'}, '2024 Urban Areas', False)

# Add the urban change layer
Map.addLayer(urban_change, urban_change_vis, 'Urban Change 2014-2024')

# Add a legend
legend_colors = [
    ('Non-urban', 'darkgreen'),
    ('Urban loss', 'orange'),
    ('Urban growth', 'red'),
    ('Stable urban', 'darkred')
]
Map.add_legend(title="Urban Change", legend_dict=legend_colors)

# Step 8: Calculate statistics about urban growth
# ----------------------------------------------

# Calculate areas in hectares
area_image = ee.Image.pixelArea().divide(10000)  # Convert to hectares

# Calculate area statistics
stats_2014 = urban_2014.multiply(area_image).reduceRegion(
    reducer=ee.Reducer.sum(),
    geometry=roi,
    scale=30,
    maxPixels=1e9
)

stats_2024 = urban_2024.multiply(area_image).reduceRegion(
    reducer=ee.Reducer.sum(),
    geometry=roi,
    scale=30,
    maxPixels=1e9
)

# Get the area values
urban_area_2014 = stats_2014.get('urban').getInfo()
urban_area_2024 = stats_2024.get('urban').getInfo()

# Calculate growth statistics
growth_percent = ((urban_area_2024 - urban_area_2014) / urban_area_2014) * 100
growth_area = urban_area_2024 - urban_area_2014

print(f"Urban area in 2014: {urban_area_2014:.2f} hectares")
print(f"Urban area in 2024: {urban_area_2024:.2f} hectares")
print(f"Urban growth: {growth_area:.2f} hectares ({growth_percent:.2f}%)")

# Step 9: Export the results
# -------------------------

# Export the RGB images
task_rgb_2014 = ee.batch.Export.image.toDrive(
    image=landsat_2014.select(['SR_B4', 'SR_B3', 'SR_B2']),
    description='ameca_rgb_2014',
    folder='ameca_analysis',
    fileNamePrefix='ameca_rgb_2014',
    region=roi,
    scale=30,
    maxPixels=1e9
)

task_rgb_2024 = ee.batch.Export.image.toDrive(
    image=landsat_2024.select(['SR_B4', 'SR_B3', 'SR_B2']),
    description='ameca_rgb_2024',
    folder='ameca_analysis',
    fileNamePrefix='ameca_rgb_2024',
    region=roi,
    scale=30,
    maxPixels=1e9
)

# Export the urban change image
task_urban_change = ee.batch.Export.image.toDrive(
    image=urban_change,
    description='ameca_urban_change',
    folder='ameca_analysis',
    fileNamePrefix='ameca_urban_change',
    region=roi,
    scale=30,
    maxPixels=1e9
)

# Start export tasks
task_rgb_2014.start()
task_rgb_2024.start()
task_urban_change.start()

print("Export tasks started. Check your Google Drive for results.")

# Display the map
print("Displaying the interactive map...")
display(Map)

# Save the map as an HTML file
Map.save(os.path.join(output_dir, 'ameca_urban_growth_map.html'))
print(f"Map saved to {os.path.join(output_dir, 'ameca_urban_growth_map.html')}")

# Create additional matplotlib visualizations
plt.figure(figsize=(12, 6))

# Bar chart for urban areas
years = ['2014', '2024']
urban_areas = [urban_area_2014, urban_area_2024]

plt.subplot(1, 2, 1)
plt.bar(years, urban_areas, color=['lightblue', 'darkblue'])
plt.title('Urban Area Comparison')
plt.ylabel('Area (hectares)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Pie chart for 2024 composition
plt.subplot(1, 2, 2)
labels = ['Stable Urban', 'New Urban Growth', 'Non-Urban']
sizes = [urban_area_2014, growth_area, 100 - (urban_area_2024)]
colors = ['darkred', 'red', 'darkgreen']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('2024 Land Composition')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ameca_urban_stats.png'), dpi=300)
print(f"Statistics visualization saved to {os.path.join(output_dir, 'ameca_urban_stats.png')}")

print("\nAnalysis complete! The results are saved in the 'ameca_analysis' directory.")

