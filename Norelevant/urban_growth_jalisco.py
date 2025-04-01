"""
Urban Growth Analysis for Jalisco (1985-Present)

This script analyzes urban growth in Jalisco, Mexico from 1985 to the present,
following the methodology described in the Inostroza paper for measuring
urban sprawl as a dynamic process.

The script:
1. Creates classified urban area images for each decade
2. Analyzes urban growth patterns (infill, extension, isolated)
3. Calculates urban sprawl metrics
4. Produces visualizations showing change over time
"""

import ee
import geemap
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import datetime

# Initialize Earth Engine
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# Create output directory if it doesn't exist
output_dir = "urban_growth_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load Jalisco boundary shapefile
def get_jalisco_boundary():
    try:
        # Load the shapefile
        shapefile_path = 'Recursos/MG_Jalisco_2020_EPSG.shp'
        gdf = gpd.read_file(shapefile_path)
        
        # Transform to EPSG:4326 (WGS 84) before extracting coordinates
        gdf_4326 = gdf.to_crs(epsg=4326)
        
        # Extract coordinates in the correct CRS
        coordinates = list(gdf_4326.geometry.values[0].exterior.coords)
        geometry = ee.Geometry.Polygon(coordinates)
        
        return geometry
    except Exception as e:
        print(f"Error loading Jalisco boundary: {e}")
        # Fallback to a simplified boundary if shapefile fails
        return ee.Geometry.Rectangle([-105.7, 18.9, -101.5, 22.8])

# Define the study area
jalisco_boundary = get_jalisco_boundary()

# Cloud masking function for Landsat imagery
def mask_clouds_landsat(image):
    """Function to mask clouds in Landsat imagery using QA band"""
    # Different for Landsat 5, 7, and 8
    if image.get('SATELLITE').getInfo() == 'LANDSAT_8':
        qa_band = image.select('QA_PIXEL')
        cloud_mask = qa_band.bitwiseAnd(1 << 3).eq(0)  # Cloud shadow
        cloud_mask = cloud_mask.And(qa_band.bitwiseAnd(1 << 5).eq(0))  # Cloud
    else:  # LANDSAT 5 or 7
        qa_band = image.select('QA_PIXEL') if 'QA_PIXEL' in image.bandNames().getInfo() else image.select('pixel_qa')
        cloud_mask = qa_band.bitwiseAnd(1 << 5).eq(0)  # Cloud
        
    return image.updateMask(cloud_mask)

# Function to get Landsat imagery for a year range
def get_landsat_collection(start_year, end_year):
    """Get cloud-masked Landsat imagery for the specified year range"""
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    
    # Use appropriate Landsat collection based on the time period
    if start_year >= 2013:
        # Landsat 8
        collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
            .filterDate(start_date, end_date) \
            .filterBounds(jalisco_boundary) \
            .map(mask_clouds_landsat)
        
        # Rename bands for consistency
        collection = collection.map(lambda img: img.select(
            ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
            ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
        ).divide(10000))  # Scale the surface reflectance values
        
    elif start_year >= 1999:
        # Landsat 7
        collection = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2") \
            .filterDate(start_date, end_date) \
            .filterBounds(jalisco_boundary) \
            .map(mask_clouds_landsat)
        
        # Rename bands for consistency
        collection = collection.map(lambda img: img.select(
            ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
            ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
        ).divide(10000))  # Scale the surface reflectance values
        
    else:
        # Landsat 5
        collection = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2") \
            .filterDate(start_date, end_date) \
            .filterBounds(jalisco_boundary) \
            .map(mask_clouds_landsat)
        
        # Rename bands for consistency
        collection = collection.map(lambda img: img.select(
            ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
            ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
        ).divide(10000))  # Scale the surface reflectance values
    
    # Create a composite
    if collection.size().getInfo() > 0:
        return collection.median()
    else:
        print(f"No images found for {start_year}-{end_year}")
        return None

# Calculate spectral indices for urban area detection
def calculate_indices(image):
    """Calculate spectral indices for urban area detection"""
    # NDBI - Normalized Difference Built-up Index
    ndbi = image.normalizedDifference(['swir1', 'nir']).rename('ndbi')
    
    # NDVI - Normalized Difference Vegetation Index
    ndvi = image.normalizedDifference(['nir', 'red']).rename('ndvi')
    
    # UI - Urban Index
    ui = image.normalizedDifference(['swir2', 'nir']).rename('ui')
    
    # MNDWI - Modified Normalized Difference Water Index
    mndwi = image.normalizedDifference(['green', 'swir1']).rename('mndwi')
    
    # Add indices to the image
    return image.addBands(ndbi).addBands(ndvi).addBands(ui).addBands(mndwi)

# Function to classify urban areas
def classify_urban(image):
    """Classify urban areas using multiple indices"""
    # Calculate indices
    image_with_indices = calculate_indices(image)
    
    # Urban area classification based on indices
    # Using thresholds based on literature
    urban = image_with_indices.select('ndbi').gt(0.0) \
        .And(image_with_indices.select('ndvi').lt(0.2)) \
        .And(image_with_indices.select('ui').gt(0.0)) \
        .And(image_with_indices.select('mndwi').lt(0.0))
    
    # Assign value 1 to urban pixels
    urban = urban.selfMask().rename('urban')
    
    return urban

# Function to analyze urban growth between two time periods
def analyze_urban_growth(urban_t1, urban_t2, period_name):
    """
    Analyze urban growth between two time periods following Inostroza methodology
    
    Parameters:
    urban_t1: Urban areas at time 1
    urban_t2: Urban areas at time 2
    period_name: Name of the period (e.g., '1985-1995')
    
    Returns:
    Image with classified urban growth types
    """
    # Make sure urban masks have value 1
    urban_t1_binary = urban_t1.unmask().eq(1)
    urban_t2_binary = urban_t2.unmask().eq(1)
    
    # New urban areas (growth)
    new_urban = urban_t2_binary.And(urban_t1_binary.Not())
    
    # Calculate distance to existing urban areas
    distance = urban_t1_binary.fastDistanceTransform(256).sqrt()
    distance = distance.multiply(ee.Image.pixelArea().sqrt())
    
    # Classify new urban areas by distance (following Inostroza methodology)
    # Class 1: Infill (within 300m of existing urban)
    # Class 2: Extension (between 300m and 900m of existing urban)
    # Class 3: Isolated development (beyond 900m of existing urban)
    infill = new_urban.And(distance.lte(300)).multiply(1)
    extension = new_urban.And(distance.gt(300)).And(distance.lte(900)).multiply(2)
    isolated = new_urban.And(distance.gt(900)).multiply(3)
    
    # Combine growth types into a single image
    growth_types = infill.add(extension).add(isolated).rename('growth_type')
    
    # For visualization and analysis
    growth_types = growth_types.set('period', period_name)
    
    return growth_types

# Function to calculate urban sprawl metrics
def calculate_urban_metrics(urban_t1, urban_t2, years_elapsed):
    """Calculate urban sprawl metrics based on Inostroza methodology"""
    # Convert masked images to binary (1 for urban, 0 for non-urban)
    urban_t1_binary = urban_t1.unmask().eq(1)
    urban_t2_binary = urban_t2.unmask().eq(1)
    
    # Calculate areas
    area_t1 = urban_t1_binary.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=jalisco_boundary,
        scale=30,
        maxPixels=1e12
    ).get('urban')
    
    area_t2 = urban_t2_binary.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=jalisco_boundary,
        scale=30,
        maxPixels=1e12
    ).get('urban')
    
    # Calculate new urban area
    new_urban = urban_t2_binary.And(urban_t1_binary.Not())
    new_area = new_urban.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=jalisco_boundary,
        scale=30,
        maxPixels=1e12
    ).get('urban')
    
    # Calculate metrics
    metrics = {
        'built_up_area_t1': area_t1.getInfo(),
        'built_up_area_t2': area_t2.getInfo(),
        'new_urban_area': new_area.getInfo(),
        'growth_rate': (float(area_t2.getInfo()) / float(area_t1.getInfo()) - 1) * 100,
        'annual_growth_rate': ((float(area_t2.getInfo()) / float(area_t1.getInfo())) ** (1/years_elapsed) - 1) * 100
    }
    
    return metrics

# Function to visualize a decade's urban areas
def visualize_urban_decade(urban_image, decade, filename):
    """Create visualization for urban areas in a specific decade"""
    # Set visualization parameters
    vis_params = {
        'min': 0,
        'max': 1,
        'palette': ['000000', 'FF0000']
    }
    
    # Create a map
    m = geemap.Map()
    m.centerObject(jalisco_boundary, 9)
    m.addLayer(jalisco_boundary, {}, 'Jalisco Boundary', False)
    m.addLayer(urban_image, vis_params, f'Urban Areas {decade}')
    
    # Add a legend
    legend_dict = {
        'Urban Areas': 'FF0000'
    }
    m.add_legend(title=f"Urban Areas {decade}", legend_dict=legend_dict)
    
    # Save the map
    m.to_html(os.path.join(output_dir, filename))
    
    # Return the map for possible embedding
    return m

# Function to visualize urban growth classification
def visualize_urban_growth(growth_image, period, filename):
    """Create visualization for urban growth classification"""
    # Set visualization parameters
    vis_params = {
        'min': 1,
        'max': 3,
        'palette': ['yellow', 'orange', 'red']
    }
    
    # Create a map
    m = geemap.Map()
    m.centerObject(jalisco_boundary, 9)
    m.addLayer(jalisco_boundary, {}, 'Jalisco Boundary', False)
    m.addLayer(growth_image, vis_params, f'Urban Growth {period}')
    
    # Add a legend
    legend_dict = {
        'Infill': 'yellow',
        'Extension': 'orange',
        'Isolated': 'red'
    }
    m.add_legend(title=f"Urban Growth Types {period}", legend_dict=legend_dict)
    
    # Save the map
    m.to_html(os.path.join(output_dir, filename))
    
    # Return the map for possible embedding
    return m

# Function to create final visualization showing all urban growth periods
def create_final_visualization(urban_images, growth_images):
    """Create a final visualization showing all urban areas and growth patterns"""
    # Create a map
    m = geemap.Map()
    m.centerObject(jalisco_boundary, 9)
    
    # Add the Jalisco boundary
    m.addLayer(jalisco_boundary, {}, 'Jalisco Boundary', False)
    
    # Add urban areas for each decade with different colors
    decades = list(urban_images.keys())
    colors = ['0000FF', '00FF00', 'FFFF00', 'FF9900', 'FF0000']
    
    for i, decade in enumerate(decades):
        m.addLayer(
            urban_images[decade], 
            {'min': 0, 'max': 1, 'palette': [colors[i]]}, 
            f'Urban Areas {decade}',
            False
        )
    


