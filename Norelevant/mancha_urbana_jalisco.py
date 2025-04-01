#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Análisis de Mancha Urbana de Jalisco (Urban Sprawl Analysis for Jalisco)

This script downloads satellite imagery from Google Earth Engine for the state of Jalisco, Mexico
from the 1980s to the present day, analyzes the urban sprawl over decades, and visualizes the results.
"""

import os
import ee
import geemap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.cluster import KMeans
import seaborn as sns
from matplotlib.colors import ListedColormap
import folium

# Inicializar Earth Engine
try:
    ee.Initialize(project='proyectocuvallesmanchau25')
    print("Earth Engine inicializado correctamente")
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='proyectocuvallesmanchau25')
    print("Nueva autenticación realizada")

print("Google Earth Engine initialized successfully.")
# Define Jalisco state boundary
jalisco = ee.FeatureCollection('FAO/GAUL/2015/level1').filter(
    ee.Filter.And(
        ee.Filter.eq('ADM0_NAME', 'Mexico'),
        ee.Filter.eq('ADM1_NAME', 'Jalisco')
    )
)

print("Fetching Jalisco boundary...")

# Ensure we have the Jalisco boundary
if jalisco.size().getInfo() == 0:
    # Fallback: Define Jalisco with coordinates (approximate bounding box)
    jalisco_coords = [
        [-105.7, 22.3], [-105.7, 18.9],
        [-101.5, 18.9], [-101.5, 22.3]
    ]
    jalisco = ee.Geometry.Polygon([jalisco_coords])
    print("Using approximate boundary for Jalisco.")
else:
    print("Using official boundary for Jalisco.")
    # Simplify the geometry to reduce complexity and memory usage
    jalisco = jalisco.geometry().simplify(maxError=100)

# Display a map of Jalisco
def display_jalisco_map():
    """Display a map of Jalisco for reference."""
    map_jalisco = geemap.Map()
    map_jalisco.centerObject(jalisco, 7)
    map_jalisco.addLayer(jalisco, {}, 'Jalisco')
    display(map_jalisco)

# Define decades for analysis
decades = {
    "1980s": {"start": "1980-01-01", "end": "1989-12-31", "max_images": 20},
    "1990s": {"start": "1990-01-01", "end": "1999-12-31", "max_images": 20},
    "2000s": {"start": "2000-01-01", "end": "2009-12-31", "max_images": 20},
    "2010s": {"start": "2010-01-01", "end": "2019-12-31", "max_images": 20},
    "2020s": {"start": "2020-01-01", "end": datetime.now().strftime("%Y-%m-%d"), "max_images": 20}
}

# Define global scale parameter for analysis (in meters)
ANALYSIS_SCALE = 100  # Using 100m instead of 30m to reduce computation load
def get_landsat_collection(start_date, end_date):
    """
    Get the appropriate Landsat collection based on the date range.
    
    Args:
        start_date: Start date (string in YYYY-MM-DD format)
        end_date: End date (string in YYYY-MM-DD format)
        
    Returns:
        ee.ImageCollection: Filtered Landsat collection
    """
    # Landsat missions by time period:
    # Landsat 4-5 TM: 1982-2012
    # Landsat 7 ETM+: 1999-present
    # Landsat 8 OLI: 2013-present
    
    start_year = int(start_date.split('-')[0])
    
    # Choose collection based on start date
    if start_year < 1999:
        # Use Landsat 5 for older imagery
        collection = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
    elif start_year < 2013:
        # Use Landsat 7 for middle period
        collection = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
    else:
        # Use Landsat 8 for recent imagery
        collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    
    # Filter by date and location
    # Apply scaling factor for Collection 2 data
    filtered = collection.filterDate(start_date, end_date).filterBounds(jalisco).map(
        lambda img: img.multiply(0.0000275).add(-0.2)  # Apply scale factor for Collection 2 surface reflectance data
    )
    
    # Return a limited number of least cloudy images to reduce memory usage
    return filtered.sort('CLOUD_COVER')
def get_urban_mask(image, start_year):
    """
    Create a mask of urban areas using spectral indices and thresholds.
    
    Args:
        image: ee.Image to process
        start_year: The starting year to determine which bands to use
        
    Returns:
        ee.Image: Binary mask where 1 represents urban areas
    """
    # Normalize band names based on the Landsat satellite
    if start_year < 2013:  # Landsat 5-7
        bands = {
            'blue': 'SR_B1', 
            'green': 'SR_B2', 
            'red': 'SR_B3',
            'nir': 'SR_B4', 
            'swir1': 'SR_B5', 
            'swir2': 'SR_B7'
        }
    else:  # Landsat 8
        bands = {
            'blue': 'SR_B2', 
            'green': 'SR_B3', 
            'red': 'SR_B4',
            'nir': 'SR_B5', 
            'swir1': 'SR_B6', 
            'swir2': 'SR_B7'
        }
    
    # Calculate indices useful for urban detection
    
    # Normalized Difference Built-up Index (NDBI)
    ndbi = image.normalizedDifference([bands['swir1'], bands['nir']]).rename('NDBI')
    
    # Modified Normalized Difference Water Index (MNDWI)
    mndwi = image.normalizedDifference([bands['green'], bands['swir1']]).rename('MNDWI')
    
    # Normalized Difference Vegetation Index (NDVI)
    ndvi = image.normalizedDifference([bands['nir'], bands['red']]).rename('NDVI')
    
    # Urban areas typically have high NDBI, low MNDWI, and low NDVI
    # Create an urban mask based on these characteristics
    urban_mask = ndbi.gt(0.1).And(mndwi.lt(0)).And(ndvi.lt(0.2))
    
    return urban_mask.rename('urban')
def get_decade_urban_area(decade_name, decade_info):
    """
    Get the urban area for a specific decade.
    
    Args:
        decade_name: Name of the decade (e.g., "1980s")
        decade_info: Dictionary with start and end dates
        
    Returns:
        dict: Information about urban areas in the decade
    """
    print(f"Processing {decade_name}...")
    
    try:
        start_date = decade_info['start']
        end_date = decade_info['end']
        start_year = int(start_date.split('-')[0])
        max_images = decade_info.get('max_images', 20)  # Limit number of images
        
        # Get the Landsat collection for this decade
        collection = get_landsat_collection(start_date, end_date)
        
        # Limit to a reasonable number of images to avoid memory issues
        limited_collection = collection.limit(max_images)
        
        # Get the median image for the decade to reduce cloud influence
        median_image = limited_collection.median()
        
        # Create urban mask
        urban_mask = get_urban_mask(median_image, start_year)
        
        # Calculate urban area in square kilometers with optimization
        urban_area_pixels = urban_mask.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=jalisco,
            scale=ANALYSIS_SCALE,  # Use coarser resolution to reduce computation
            maxPixels=1e12,
            tileScale=4  # Use tileScale to distribute computation
        ).get('urban')
        
        # Calculate area considering the scale used
        urban_area_km2 = ee.Number(urban_area_pixels).multiply(ANALYSIS_SCALE * ANALYSIS_SCALE).divide(1e6)
        urban_area_value = urban_area_km2.getInfo()
        
        # Get image for visualization
        if start_year < 2013:
            rgb_viz = median_image.select(['SR_B3', 'SR_B2', 'SR_B1'])
        else:
            rgb_viz = median_image.select(['SR_B4', 'SR_B3', 'SR_B2'])
        
        return {
            'decade': decade_name,
            'urban_mask': urban_mask,
            'urban_area_km2': urban_area_value,
            'median_image': median_image,
            'rgb_viz': rgb_viz,
            'start_year': start_year
        }
    except Exception as e:
        print(f"Error processing {decade_name}: {e}")
        # Return placeholder data to avoid breaking the analysis
        return {
            'decade': decade_name,
            'urban_mask': None,
            'urban_area_km2': 0,
            'median_image': None,
            'rgb_viz': None,
            'start_year': start_year
        }

def analyze_urban_growth(results):
    """
    Analyze the urban growth based on results from each decade.
    
    Args:
        results: List of dictionaries with results for each decade
        
    Returns:
        pd.DataFrame: DataFrame with urban growth analysis
    """
    # Create a DataFrame with the results
    df = pd.DataFrame([
        {
            'Decade': r['decade'],
            'Urban Area (km²)': r['urban_area_km2'],
            'Start Year': r['start_year']
        } for r in results
    ])
    
    # Calculate growth between decades
    df['Growth (km²)'] = df['Urban Area (km²)'].diff()
    df['Growth Rate (%)'] = (df['Growth (km²)'] / df['Urban Area (km²)'].shift(1) * 100)
    
    return df
def visualize_results(results, analysis_df):
    """
    Visualize the results of the urban sprawl analysis.
    
    Args:
        results: List of dictionaries with results for each decade
        analysis_df: DataFrame with urban growth analysis
    """
    # Filter out results with None values (from error handling)
    valid_results = [r for r in results if r['urban_mask'] is not None]
    # 1. Create a map visualizing urban sprawl over time
    m = geemap.Map()
    m.centerObject(jalisco, 8)
    
    # Add Jalisco boundary
    m.addLayer(jalisco, {'color': 'black'}, 'Jalisco Boundary')
    
    # Add urban masks for each decade with different colors
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF']
    
    for idx, result in enumerate(valid_results):
        decade = result['decade']
        urban_mask = result['urban_mask']
        
        # Add the urban mask with a specific color
        m.addLayer(
            urban_mask.selfMask(),
            {'palette': [colors[idx % len(colors)]]},
            f'Urban Area {decade}'
        )
        # Add a legend
    legend_dict = {
        result['decade']: colors[idx % len(colors)]
        for idx, result in enumerate(valid_results)
    }
    m.add_legend(title="Urban Areas by Decade", legend_dict=legend_dict)
    
    # Save the map
    m.save('jalisco_urban_sprawl_map.html')
    
    # 2. Plot urban area by decade
    plt.figure(figsize=(12, 6))
    
    # Bar chart of urban area by decade
    plt.subplot(121)
    sns.barplot(x='Decade', y='Urban Area (km²)', data=analysis_df)
    plt.title('Urban Area in Jalisco by Decade')
    plt.xticks(rotation=45)
    plt.ylabel('Area (km²)')
    
    # Line chart of growth rate
    plt.subplot(122)
    sns.lineplot(x='Decade', y='Growth Rate (%)', data=analysis_df, marker='o')
    plt.title('Urban Growth Rate by Decade')
    plt.xticks(rotation=45)
    plt.ylabel('Growth Rate (%)')
    
    plt.tight_layout()
    plt.savefig('jalisco_urban_growth.png', dpi=300)
    
    print("Visualizations have been saved.")
def main():
    """Main function to run the urban sprawl analysis."""
    print("Starting urban sprawl analysis for Jalisco, Mexico...")
    print(f"Using analysis scale: {ANALYSIS_SCALE}m (coarser resolution to reduce memory usage)")
    
    # Process each decade
    results = []
    for decade_name, decade_info in decades.items():
        try:
            decade_result = get_decade_urban_area(decade_name, decade_info)
            results.append(decade_result)
            print(f"Successfully processed {decade_name}")
        except Exception as e:
            print(f"Error processing {decade_name}: {e}")
            # Add placeholder result to maintain decade sequence
            start_year = int(decade_info['start'].split('-')[0])
            placeholder_result = {
                'decade': decade_name,
                'urban_mask': None,
                'urban_area_km2': 0,
                'median_image': None,
                'rgb_viz': None,
                'start_year': start_year
            }
            results.append(placeholder_result)
            print(f"Added placeholder for {decade_name}")
    
    # Only continue with analysis if we have results
    if results:
        # Analyze urban growth
        analysis_df = analyze_urban_growth(results)
        print("\nUrban Growth Analysis:")
        print(analysis_df)
        
        # Visualize results
        try:
            visualize_results(results, analysis_df)
            print("Visualizations created successfully.")
        except Exception as e:
            print(f"Error creating visualizations: {e}")
    else:
        print("No results to analyze. Check for errors above.")
    
    print("\nAnalysis complete. Check the output files for visualizations.")

if __name__ == "__main__":
    main()

