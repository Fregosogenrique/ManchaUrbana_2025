"""
Comprehensive 20-Year Analysis of Urban Growth in Ahualulco de Mercado, Jalisco, Mexico (2004-2024)
==================================================================================================

This script creates a comprehensive visualization of urban growth in Ahualulco de Mercado over a
20-year period in a 2x3 grid layout. The top row displays Landsat images for 2004,
2014, and 2024, while the bottom row shows urban change analysis for 2004-2014,
2014-2024, and the full 20-year period (2004-2024). The script produces a visual
comparison with clear labels and a colorbar for the urban change maps.

Dependencies:
- earthengine-api
- geemap
- numpy
- matplotlib
- rasterio (for local file operations)
"""

import ee
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import datetime
import requests
import io
from matplotlib.image import imread
import logging
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
import matplotlib.patches as mpatches

# Set up output directory
output_dir = 'Ahualulco_comparison'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set up a separate directory for individual images
individual_output_dir = 'Ahualulco_individual'
if not os.path.exists(individual_output_dir):
    os.makedirs(individual_output_dir)
    print(f"Created directory for individual images: {individual_output_dir}")

# Initialize and authenticate Earth Engine
try:
    ee.Initialize(project='proyectocuvallesmanchau25')
    print("Earth Engine inicializado correctamente")
except Exception as e:
    print(f"Error initializing Earth Engine: {e}")
    print("Attempting authentication...")
    ee.Authenticate()
    ee.Initialize(project='proyectocuvallesmanchau25')
    print("Nueva autenticación realizada")


def export_to_geotiff(image, filename, region, description):
    """
    Export an Earth Engine image to GeoTIFF format for use in QGIS

    Parameters:
    image (ee.Image): Earth Engine image to export
    filename (str): Name for the output file
    region (ee.Geometry): Region to export
    description (str): Description for the export task

    Returns:
    ee.batch.Task: The export task that can be monitored
    """
    # Configure the export
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder='ManchaUrbana_Ahualulco',
        fileNamePrefix=filename,
        region=region,
        scale=30,
        crs='EPSG:4326',
        maxPixels=1e13
    )

    # Start the export
    task.start()
    print(f"Started GeoTIFF export task for {filename}")
    return task


def get_landsat_collection(year, roi):
    """
    Get Landsat imagery for a specific year and region.

    For 2004: Use Landsat 5 or Landsat 7
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

    # For 2004, use Landsat 5 or Landsat 7
    if year == 2004:
        # Try Landsat 5 first
        collection = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2') \
            .filterBounds(roi) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUD_COVER', 20))  # Filter clouds

        # Check if we have enough Landsat 5 images
        count = collection.size().getInfo()
        if count < 3:  # If fewer than 3 images, try Landsat 7
            print(f"Only {count} Landsat 5 images available for 2004, trying Landsat 7")
            collection = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2') \
                .filterBounds(roi) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUD_COVER', 20))

            # Check if we have enough Landsat 7 images
            count = collection.size().getInfo()
            if count < 3:
                print(f"Warning: Only {count} Landsat 7 images available for 2004. Results may be affected.")

        # For Landsat 5/7, scale the surface reflectance values differently
        collection = collection.map(lambda img: img.select(['SR_B[1-5]', 'SR_B7']).multiply(0.0000275).add(-0.2))

    # For 2014, use Landsat 8
    elif year == 2014:
        collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
            .filterBounds(roi) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUD_COVER', 20))  # Filter clouds

        # Scale the surface reflectance values
        collection = collection.map(lambda img: img.select(['SR_B.*']).multiply(0.0000275).add(-0.2))

    # For 2024, use Landsat 9 if available, otherwise Landsat 8
    elif year == 2024:
        # Note: As of my knowledge cutoff, use most recent available images
        # Check if we're already in 2024, if not use available data
        current_year = datetime.datetime.now().year
        if current_year < 2024:
            # Use the most recent available data
            start_date = f'{current_year - 1}-01-01'
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


def calculate_urban_indices(image, sensor='L8'):
    """
    Calculate urban indices for the given image

    Parameters:
    image (ee.Image): Landsat image
    sensor (str): Sensor type ('L5' for Landsat 5, 'L7' for Landsat 7, 'L8' for Landsat 8/9)

    Returns:
    ee.Image: Image with added urban indices
    """
    if sensor == 'L5' or sensor == 'L7':
        # For Landsat 5/7, bands are:
        # B1=Blue, B2=Green, B3=Red, B4=NIR, B5=SWIR1, B7=SWIR2

        # NDBI (Normalized Difference Built-up Index)
        # NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
        ndbi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDBI')

        # NDVI (Normalized Difference Vegetation Index) for comparison
        # NDVI = (NIR - Red) / (NIR + Red)
        ndvi = image.normalizedDifference(['SR_B4', 'SR_B3']).rename('NDVI')

        # MNDWI (Modified Normalized Difference Water Index) to remove water bodies
        # MNDWI = (Green - SWIR1) / (Green + SWIR1)
        mndwi = image.normalizedDifference(['SR_B2', 'SR_B5']).rename('MNDWI')
    else:
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


def create_comparison_plot():
    """
    Create a comprehensive comparison plot in a 2x3 grid layout showing:
    - Top row: Landsat images for 2004, 2014, and 2024
    - Bottom row: Urban change between 2004-2014, 2014-2024, and full 20-year change (2004-2024)
    """
    # Get the Region of Interest (ROI) - Ahualulco de Mercado, Jalisco coordinates
    ahualulco_point = ee.Geometry.Point([-103.9705, 20.6764])  # Coordenadas del centro de Ahualulco
    roi = ahualulco_point.buffer(7000)  # 7 km de buffer alrededor del centro

    # Get Landsat composites for all three time periods
    print("Fetching Landsat images for 2004, 2014, and 2024...")
    landsat_2004 = get_landsat_collection(2004, roi)
    landsat_2014 = get_landsat_collection(2014, roi)
    landsat_2024 = get_landsat_collection(2024, roi)

    # Calculate urban indices for all three periods
    landsat_2004_indices = calculate_urban_indices(landsat_2004, sensor='L5')  # Use L5 for 2004
    landsat_2014_indices = calculate_urban_indices(landsat_2014)  # Default is L8
    landsat_2024_indices = calculate_urban_indices(landsat_2024)  # Default is L8

    # Create urban masks for all three periods
    urban_2004 = create_urban_mask(landsat_2004_indices)
    urban_2014 = create_urban_mask(landsat_2014_indices)
    urban_2024 = create_urban_mask(landsat_2024_indices)

    # Calculate urban change for different time periods
    # 0 = Non-urban 
    # 1 = Urban growth
    # 2 = Stable urban

    # This mapping ensures we only get the three categories we want
    def calculate_urban_change(earlier, later):
        # Combine the images and create custom mapping
        stable_urban = earlier.And(later).multiply(2)  # Stable urban (2)
        urban_growth = later.And(earlier.Not()).multiply(1)  # Urban growth (1)
        non_urban = earlier.Not().And(later.Not()).multiply(0)  # Non-urban (0)
        return stable_urban.add(urban_growth).add(non_urban)

    # Calculate changes for each period
    urban_change_04_14 = calculate_urban_change(urban_2004, urban_2014)
    urban_change_14_24 = calculate_urban_change(urban_2014, urban_2024)
    urban_change_04_24 = calculate_urban_change(urban_2004, urban_2024)

    # Download the images as NumPy arrays for visualization
    # Scale and region parameters for downloading images
    scale = 30  # 30m resolution
    region = roi
    # RGB visualization parameters for Landsat 5/7 (2004)
    vis_params_l5 = {
        'bands': ['SR_B3', 'SR_B2', 'SR_B1'],  # Different band numbering for Landsat 5/7
        'min': 0,
        'max': 0.3,
        'gamma': 1.4
    }

    # RGB visualization parameters for Landsat 8/9
    vis_params_l8 = {
        'bands': ['SR_B4', 'SR_B3', 'SR_B2'],  # RGB bands for Landsat 8/9
        'min': 0,
        'max': 0.3,
        'gamma': 1.4
    }

    # Get 2004 RGB image
    # Get 2004 RGB image
    rgb_2004 = landsat_2004.visualize(**vis_params_l5).clip(roi)
    try:
        rgb_2004_url = rgb_2004.getThumbURL({
            'region': region,
            'dimensions': 1024,
            'format': 'png'
        })
        print(f"2004 image URL: {rgb_2004_url[:100]}...")  # Print first 100 chars of URL
    except Exception as e:
        print(f"Error generating 2004 image URL: {e}")
        # Fallback: try with a simpler request
        rgb_2004_url = rgb_2004.getThumbURL({
            'dimensions': 1024,
            'format': 'png'
        })

    # Get 2014 RGB image
    rgb_2014 = landsat_2014.visualize(**vis_params_l8).clip(roi)
    try:
        rgb_2014_url = rgb_2014.getThumbURL({
            'region': region,
            'dimensions': 1024,
            'format': 'png'
        })
        print(f"2014 image URL: {rgb_2014_url[:100]}...")  # Print first 100 chars of URL
    except Exception as e:
        print(f"Error generating 2014 image URL: {e}")
        # Fallback: try with a simpler request
        rgb_2014_url = rgb_2014.getThumbURL({
            'dimensions': 1024,
            'format': 'png'
        })

    # Get 2024 RGB image
    rgb_2024 = landsat_2024.visualize(**vis_params_l8).clip(roi)
    try:
        rgb_2024_url = rgb_2024.getThumbURL({
            'region': region,
            'dimensions': 1024,
            'format': 'png'
        })
        print(f"2024 image URL: {rgb_2024_url[:100]}...")  # Print first 100 chars of URL
    except Exception as e:
        print(f"Error generating 2024 image URL: {e}")
        # Fallback: try with a simpler request
        rgb_2024_url = rgb_2024.getThumbURL({
            'dimensions': 1024,
            'format': 'png'
        })

    # Get urban change visualization parameters
    urban_change_vis = {
        'min': 0,
        'max': 2,
        'palette': ['darkgreen', 'red', 'blue']  # [non-urban, urban growth, stable urban]
    }

    # Visualize 2004-2014 change
    change_img_04_14 = urban_change_04_14.visualize(**urban_change_vis).clip(roi)
    try:
        change_url_04_14 = change_img_04_14.getThumbURL({
            'region': region,
            'dimensions': 1024,
            'format': 'png'
        })
        print(f"Change 2004-2014 URL: {change_url_04_14[:100]}...")
    except Exception as e:
        print(f"Error generating 2004-2014 change URL: {e}")
        # Fallback
        change_url_04_14 = change_img_04_14.getThumbURL({
            'dimensions': 1024,
            'format': 'png'
        })

    # Visualize 2014-2024 change
    change_img_14_24 = urban_change_14_24.visualize(**urban_change_vis).clip(roi)
    try:
        change_url_14_24 = change_img_14_24.getThumbURL({
            'region': region,
            'dimensions': 1024,
            'format': 'png'
        })
        print(f"Change 2014-2024 URL: {change_url_14_24[:100]}...")
    except Exception as e:
        print(f"Error generating 2014-2024 change URL: {e}")
        # Fallback
        change_url_14_24 = change_img_14_24.getThumbURL({
            'dimensions': 1024,
            'format': 'png'
        })

    # Visualize full 20-year change (2004-2024)
    change_img_04_24 = urban_change_04_24.visualize(**urban_change_vis).clip(roi)
    try:
        change_url_04_24 = change_img_04_24.getThumbURL({
            'region': region,
            'dimensions': 1024,
            'format': 'png'
        })
        print(f"Change 2004-2024 URL: {change_url_04_24[:100]}...")
    except Exception as e:
        print(f"Error generating 2004-2024 change URL: {e}")
        # Fallback
        change_url_04_24 = change_img_04_24.getThumbURL({
            'dimensions': 1024,
            'format': 'png'
        })

    # Export images to GeoTIFF for QGIS
    print("\nExporting images to GeoTIFF format for QGIS analysis...")

    # Export satellite images with natural color visualization
    vis_satellite_2004 = landsat_2004.visualize(**vis_params_l5)
    vis_satellite_2014 = landsat_2014.visualize(**vis_params_l8)
    vis_satellite_2024 = landsat_2024.visualize(**vis_params_l8)

    # Export raw satellite images (for analysis)
    export_to_geotiff(landsat_2004, 'Ahualulco_2004_raw', roi, 'Ahualulco2004Raw')
    export_to_geotiff(landsat_2014, 'Ahualulco_2014_raw', roi, 'Ahualulco2014Raw')
    export_to_geotiff(landsat_2024, 'Ahualulco_2024_raw', roi, 'Ahualulco2024Raw')

    # Export visualized satellite images (for display)
    export_to_geotiff(vis_satellite_2004, 'Ahualulco_2004_rgb', roi, 'Ahualulco2004RGB')
    export_to_geotiff(vis_satellite_2014, 'Ahualulco_2014_rgb', roi, 'Ahualulco2014RGB')
    export_to_geotiff(vis_satellite_2024, 'Ahualulco_2024_rgb', roi, 'Ahualulco2024RGB')

    # Export urban change maps
    export_to_geotiff(urban_change_04_14, 'Ahualulco_change_2004_2014', roi, 'Change2004_2014')
    export_to_geotiff(urban_change_14_24, 'Ahualulco_change_2014_2024', roi, 'Change2014_2024')
    export_to_geotiff(urban_change_04_24, 'Ahualulco_change_2004_2024', roi, 'Change2004_2024')

    # Export urban masks for each year (for additional analysis)
    export_to_geotiff(urban_2004, 'Ahualulco_urban_2004', roi, 'Urban2004')
    export_to_geotiff(urban_2014, 'Ahualulco_urban_2014', roi, 'Urban2014')
    export_to_geotiff(urban_2024, 'Ahualulco_urban_2024', roi, 'Urban2024')

    print(
        "All GeoTIFF export tasks have been started. Files will be available in your Google Drive folder 'ManchaUrbana_Ahualulco'\n")

    # Now create the plot - 2x3 grid layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Function to download and display image from URL with better error handling
    def display_ee_image(url, ax, title):
        try:
            # Make the request with a timeout
            response = requests.get(url, timeout=30)

            # Check if the request was successful
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code} for {title}")
                print(f"Response content: {response.text[:200]}...")  # Print first 200 chars
                raise Exception(f"Failed to download image: HTTP {response.status_code}")

            # Check content type
            content_type = response.headers.get('Content-Type', '')
            print(f"DEBUG: Content-Type for {title}: {content_type}")

            if 'image/png' not in content_type and 'application/octet-stream' not in content_type:
                print(f"Warning: Expected PNG image but got {content_type}")

            # Try to read the image
            img_data = io.BytesIO(response.content)

            try:
                # First try standard PIL/matplotlib method
                img = imread(img_data, format='png')
            except Exception as e:
                print(f"Error reading image as PNG: {e}")

                # Fallback: Try alternative method to download the image
                print(f"Trying alternative method to download {title} image...")

                # Reset the image data buffer position
                img_data.seek(0)

                # Try using Pillow directly
                from PIL import Image
                try:
                    pil_img = Image.open(img_data)
                    img = np.array(pil_img)
                except Exception as pil_error:
                    print(f"Pillow also failed: {pil_error}")

                    # Last resort: Create a dummy image with error message
                    print("Creating placeholder image...")
                    img = np.ones((512, 512, 3))  # RGB placeholder
                    ax.text(0.5, 0.5, f"Image Load Error\n{title}",
                            ha='center', va='center', transform=ax.transAxes,
                            fontsize=14, color='red')

            # Display the image
            ax.imshow(img)
            ax.set_title(title, fontsize=14)
            ax.axis('off')
            return img

        except Exception as e:
            print(f"Error in display_ee_image for {title}: {e}")
            # Create a blank/error image
            img = np.ones((512, 512, 3))  # RGB placeholder
            ax.imshow(img)
            ax.set_title(f"{title} (Error loading)", fontsize=14, color='red')
            ax.text(0.5, 0.5, f"Failed to load image:\n{str(e)}",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='red')
            ax.axis('off')
            return img

    # Top row: Display satellite images for all three time periods
    img_2004 = display_ee_image(rgb_2004_url, axes[0, 0], 'Ahualulco de Mercado 2004')
    img_2014 = display_ee_image(rgb_2014_url, axes[0, 1], 'Ahualulco de Mercado 2014')
    img_2024 = display_ee_image(rgb_2024_url, axes[0, 2], 'Ahualulco de Mercado2024')

    # Bottom row: Display urban change analyses
    # 2004-2014 change
    img_change_04_14 = display_ee_image(change_url_04_14, axes[1, 0], 'Urban Change 2004-2014')

    # 2014-2024 change
    img_change_14_24 = display_ee_image(change_url_14_24, axes[1, 1], 'Urban Change 2014-2024')

    # 2004-2024 change (full 20-year period)
    img_change_04_24 = display_ee_image(change_url_04_24, axes[1, 2], 'Urban Change 2004-2024')

    # Add colorbar for urban change
    cmap = ListedColormap(['darkgreen', 'red', 'blue'])

    # Create legend patches
    legend_labels = ['Non-urban', 'Urban growth', 'Stable urban']
    legend_colors = ['darkgreen', 'red', 'blue']
    patches = [mpatches.Patch(color=color, label=label)
               for color, label in zip(legend_colors, legend_labels)]

    # Add the legend to each change plot
    for i in range(3):
        axes[1, i].legend(handles=patches, loc='lower right', framealpha=0.7)

    plt.tight_layout()
    fig_title = fig.suptitle('Análisis de Crecimiento Urbano de Ahualulco de Mercado, Jalisco (2004-2024)',
                             fontsize=16, y=1.02)
    plt.savefig(os.path.join(output_dir, 'ahualulco_urban_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'ahualulco_urban_comparison.pdf'), bbox_inches='tight')

    print(
        f"Comprehensive 20-year urban growth analysis saved to {os.path.join(output_dir, 'ahualulco_urban_comparison.png')}")
    print("Top row: Landsat images for 2004, 2014, and 2024")
    print("Bottom row: Urban change analysis for 2004-2014, 2014-2024, and 2004-2024 (full 20-year period)")
    return fig


def main():
    """Main function to run the comparison plot generation"""
    print("Generating comprehensive 20-year urban growth analysis for Ahualulco de Mercado...")
    try:
        fig = create_comparison_plot()
        plt.show()
        print("Comprehensive 20-year urban growth analysis completed successfully!")
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
