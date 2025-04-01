import ee
import geemap
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Initialize Earth Engine
try:
    ee.Initialize(project='proyectocuvallesmanchau25')
    print("Earth Engine inicializado correctamente")
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='proyectocuvallesmanchau25')
    print("Nueva autenticación realizada")

# Create output directories if they don't exist
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

images_dir = os.path.join(output_dir, "images")
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

stats_dir = os.path.join(output_dir, "stats")
if not os.path.exists(stats_dir):
    os.makedirs(stats_dir)

# Define a simplified boundary for Jalisco to avoid payload size limitations
def get_jalisco_boundary():
    try:
        # Try to load the shapefile
        shapefile_path = 'Recursos/MG_Jalisco_2020_EPSG.shp'
        gdf = gpd.read_file(shapefile_path)
        
        # Convert to EPSG:4326 (WGS84) for Earth Engine
        gdf_4326 = gdf.to_crs(epsg=4326)
        
        # Create a simplified boundary to avoid payload limitations
        # Method 1: Use the bounding box
        bounds = gdf_4326.total_bounds  # (min_x, min_y, max_x, max_y)
        bbox = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])
        
        # Print debug information
        print(f"Bounding box coordinates: {bounds}")
        print(f"Approximate area: {gdf_4326.area.sum() / 1e6:.2f} km²")
        
        # Method 2: If needed, we can create a simplified polygon
        # For now, using the bounding box approach to ensure small payload
        
        return bbox
    
    except Exception as e:
        print(f"Error loading Jalisco shapefile: {e}")
        # Fallback to a manually defined bounding box for Jalisco
        # Approximate coordinates for Jalisco state
        jalisco_coords = [
            [-105.7, 18.9],  # Southwest
            [-105.7, 22.8],  # Northwest
            [-101.5, 22.8],  # Northeast
            [-101.5, 18.9],  # Southeast
            [-105.7, 18.9]   # Southwest (close the polygon)
        ]
        return ee.Geometry.Polygon([jalisco_coords])

# Function to get Landsat imagery for a specific year range
def get_landsat_imagery(year, geometry):
    start_date = f"{year-2}-01-01"
    end_date = f"{year+2}-12-31"
    
    # Select the appropriate Landsat collection based on the year
    if year < 1984:
        print(f"No Landsat imagery available before 1984. Using earliest available.")
        collection = ee.ImageCollection("LANDSAT/LM01/C02/T1")
    elif year < 1995:
        # Landsat 5
        collection = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
    elif year < 2005:
        # Landsat 7
        collection = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
    elif year < 2015:
        # Landsat 7 (continued)
        collection = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
    else:
        # Landsat 8/9
        collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    
    # Filter by date and location
    images = collection.filterDate(start_date, end_date).filterBounds(geometry)
    
    # Function to cloud mask L7/L8 SR data
    def maskCloudsAndShadows(image):
        # Bits 3 and 5 are cloud shadow and cloud, respectively
        cloudShadowBitMask = (1 << 3)
        cloudsBitMask = (1 << 5)
        
        # Get the quality band
        qa = image.select('QA_PIXEL')
        
        # Both flags should be set to zero, indicating clear conditions
        mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
            qa.bitwiseAnd(cloudsBitMask).eq(0))
        
        return image.updateMask(mask)
    
    # Apply cloud mask
    masked_collection = images.map(maskCloudsAndShadows)
    
    # Take median to get a cloud-free composite
    median = masked_collection.median()
    
    print(f"Found {images.size().getInfo()} images for year {year}")
    
    return median

# Calculate NDBI (Normalized Difference Built-up Index)
def calculate_ndbi(image):
    # For Landsat 5/7
    if 'B5' in image.bandNames().getInfo():
        swir = image.select('B5')  # SWIR1
        nir = image.select('B4')   # NIR
    # For Landsat 8/9
    else:
        swir = image.select('SR_B6')  # SWIR1
        nir = image.select('SR_B5')   # NIR
    
    ndbi = swir.subtract(nir).divide(swir.add(nir)).rename('NDBI')
    
    return ndbi

# Function to detect urban areas using NDBI
def detect_urban_areas(image, year, geometry):
    ndbi = calculate_ndbi(image)
    
    # Threshold NDBI to identify built-up areas (usually positive values)
    # The threshold can be adjusted based on validation
    ndbi_threshold = 0.0
    urban_mask = ndbi.gt(ndbi_threshold)
    
    # Create binary urban/non-urban image
    urban_areas = urban_mask.rename('urban')
    
    return urban_areas

# Function to calculate urban area statistics
def calculate_urban_stats(urban_image, year, geometry):
    # Calculate area of urban pixels
    pixelArea = ee.Image.pixelArea()
    urban_area = urban_image.multiply(pixelArea)
    
    # Sum up all urban pixels to get total urban area
    stats = urban_area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry,
        scale=30,
        maxPixels=1e9
    )
    
    # Get the total area in square kilometers
    urban_area_km2 = ee.Number(stats.get('urban')).divide(1e6).getInfo()  # Convert to km²
    
    print(f"Year {year}: Urban area = {urban_area_km2:.2f} km²")
    
    return {
        'year': year,
        'urban_area_km2': urban_area_km2
    }

# Function to visualize urban areas on a map
def visualize_urban_map(urban_image, year, geometry):
    # Create a map
    map_object = geemap.Map()
    
    # Add the background imagery (for reference)
    map_object.addLayer(
        ee.Image().rgb().paint(geometry, 0, 2),
        {},
        f'Jalisco Boundary'
    )
    
    # Add the urban areas
    map_object.addLayer(
        urban_image.selfMask(),
        {'palette': ['red']},
        f'Urban Areas {year}'
    )
    
    # Center the map on Jalisco
    map_object.centerObject(geometry, 8)
    
    # Save the map as an HTML file
    map_file = os.path.join(images_dir, f"urban_map_{year}.html")
    map_object.save(map_file)
    
    print(f"Urban map for {year} saved to {map_file}")
    
    return map_object

# Function to analyze urban growth between two time periods
def analyze_urban_growth(earlier_urban, later_urban, earlier_year, later_year, geometry):
    # 0 = non-urban in both periods
    # 1 = urban in later period only (new urban growth)
    # 2 = urban in both periods (persistent urban)
    
    # Create a composite image: 
    # - Green: new urban areas (growth)
    # - Red: persistent urban areas
    growth = earlier_urban.add(later_urban)
    
    # Calculate statistics for each category
    area_image = ee.Image.pixelArea().divide(1e6)  # Area in km²
    
    # Persistent urban (value 2)
    persistent_mask = growth.eq(2)
    persistent_area = persistent_mask.multiply(area_image)
    
    # New urban growth (value 1)
    new_urban_mask = growth.eq(1)
    new_urban_area = new_urban_mask.multiply(area_image)
    
    # Calculate statistics
    stats = ee.Dictionary({
        'persistent': persistent_area.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=30,
            maxPixels=1e9
        ),
        'new_urban': new_urban_area.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=30,
            maxPixels=1e9
        )
    })
    
    persistent_km2 = ee.Number(stats.get('persistent')).getInfo()
    new_urban_km2 = ee.Number(stats.get('new_urban')).getInfo()
    
    print(f"Growth {earlier_year}-{later_year}: Persistent urban = {persistent_km2:.2f} km², New urban = {new_urban_km2:.2f} km²")
    
    # Visualize growth
    growth_map = geemap.Map()
    
    # Add the boundary
    growth_map.addLayer(
        ee.Image().rgb().paint(geometry, 0, 2),
        {},
        f'Jalisco Boundary'
    )
    
    # Add persistent urban areas (red)
    growth_map.addLayer(
        persistent_mask.selfMask(),
        {'palette': ['red']},
        f'Persistent Urban {earlier_year}-{later_year}'
    )
    
    # Add new urban growth (green)
    growth_map.addLayer(
        new_urban_mask.selfMask(),
        {'palette': ['green']},
        f'New Urban Growth {earlier_year}-{later_year}'
    )
    
    # Center the map
    growth_map.centerObject(geometry, 8)
    
    # Save the map
    growth_map_file = os.path.join(images_dir, f"urban_growth_{earlier_year}_{later_year}.html")
    growth_map.save(growth_map_file)
    
    print(f"Urban growth map for {earlier_year}-{later_year} saved to {growth_map_file}")
    
    return {
        'period': f"{earlier_year}-{later_year}",
        'persistent_urban_km2': persistent_km2,
        'new_urban_km2': new_urban_km2
    }

# Plot the urban area trends over time
def plot_urban_trends(urban_stats):
    years = [stat['year'] for stat in urban_stats]
    areas = [stat['urban_area_km2'] for stat in urban_stats]
    
    plt.figure(figsize=(12, 8))
    plt.plot(years, areas, 'o-', linewidth=2, markersize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Urban Area Growth in Jalisco, Mexico (1985-2025)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Urban Area (km²)', fontsize=14)
    
    for i, (year, area) in enumerate(zip(years, areas)):
        plt.annotate(f"{area:.1f} km²", (year, area), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    # Calculate and display growth rates
    for i in range(1, len(years)):
        growth_rate = ((areas[i] - areas[i-1]) / areas[i-1]) * 100
        plt.annotate(f"+{growth_rate:.1f}%", 
                    ((years[i] + years[i-1])/2, (areas[i] + areas[i-1])/2),
                    textcoords="offset points", 
                    xytext=(0,-15), 
                    ha='center',
                    fontsize=10,
                    color='green')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'urban_growth_trend.png'), dpi=300)
    print(f"Urban growth trend plot saved to {os.path.join(output_dir, 'urban_growth_trend.png')}")
    
    plt.show()

# Function to create a final visualization showing all decades
def create_final_visualization(urban_images, years, geometry):
    # Create a map with all urban extents by decade
    final_map = geemap.Map()
    
    # Add the Jalisco boundary
    final_map.addLayer(
        ee.Image().rgb().paint(geometry, 0, 2),
        {},
        f'Jalisco Boundary'
    )
    
    # Add each decade with a different color
    colors = ['#4b0082', '#0000ff', '#00ff00', '#ffff00', '#ff0000']  # Purple, Blue, Green, Yellow, Red
    
    for i, (year, urban_image) in enumerate(zip(years, urban_images)):
        final_map.addLayer(
            urban_image.selfMask(),
            {'palette': [colors[i]]},
            f'Urban {year}',
            i == len(years) - 1  # Most recent year is visible by default
        )
    
    # Center the map
    final_map.centerObject(geometry, 8)
    
    # Add a legend
    legend_dict = {}
    for i, year in enumerate(years):
        legend_dict[f'Urban area {year}'] = colors[i]
    
    final_map.add_legend(title="Urban Growth by Decade", legend_dict=legend_dict)
    
    # Save the map
    final_map_file = os.path.join(output_dir, 'urban_growth_all_decades.html')
    final_map.save(final_map_file)

    print(f"Final urban growth visualization saved to {final_map_file}")

    return final_map

# Main execution function
def main():
    print("Starting urban growth analysis for Jalisco, Mexico...")
    
    # Get the Jalisco boundary
    print("Getting Jalisco boundary...")
    jalisco_geometry = get_jalisco_boundary()
    
    # Define the decades to analyze
    decades = [1985, 1995, 2005, 2015, 2025]
    
    # Process each decade
    urban_images = []
    urban_stats = []
    
    print(f"Processing {len(decades)} decades from {decades[0]} to {decades[-1]}...")
    
    # Get imagery and detect urban areas for each decade
    for year in decades:
        print(f"\nProcessing year {year}...")
        
        # Get Landsat imagery
        landsat_image = get_landsat_imagery(year, jalisco_geometry)
        
        # Detect urban areas
        urban_image = detect_urban_areas(landsat_image, year, jalisco_geometry)
        urban_images.append(urban_image)
        
        # Calculate statistics
        stats = calculate_urban_stats(urban_image, year, jalisco_geometry)
        urban_stats.append(stats)
        
        # Visualize urban areas for this decade
        visualize_urban_map(urban_image, year, jalisco_geometry)
    
    # Analyze urban growth between consecutive decades
    growth_stats = []
    for i in range(1, len(decades)):
        earlier_year = decades[i-1]
        later_year = decades[i]
        earlier_urban = urban_images[i-1]
        later_urban = urban_images[i]
        
        print(f"\nAnalyzing urban growth between {earlier_year} and {later_year}...")
        growth_stat = analyze_urban_growth(earlier_urban, later_urban, earlier_year, later_year, jalisco_geometry)
        growth_stats.append(growth_stat)
    
    # Create the final plots and visualizations
    print("\nCreating final visualization...")
    plot_urban_trends(urban_stats)
    create_final_visualization(urban_images, decades, jalisco_geometry)
    
    print("\nUrban growth analysis complete!")

# Execute the main function if the script is run directly
if __name__ == "__main__":
    main()
