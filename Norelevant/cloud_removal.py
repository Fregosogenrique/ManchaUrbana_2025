import os
import numpy as np
import rasterio

def apply_cloud_mask(image):
    # Simple thresholding to simulate cloud masking
    return np.where(image > 200, 0, image)

def process_images(input_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith('.tif'):
            file_path = os.path.join(input_dir, filename)
            with rasterio.open(file_path) as src:
                image = src.read()
                # Apply cloud mask only before saving
                final_image = apply_cloud_mask(image)

                # Save the cleaned image
                with rasterio.open(
                    file_path,
                    'w',
                    driver=src.driver,
                    height=final_image.shape[1],
                    width=final_image.shape[2],
                    count=src.count,
                    dtype=final_image.dtype,
                    crs=src.crs,
                    transform=src.transform,
                ) as dst:
                    dst.write(final_image)

if __name__ == "__main__":
    process_images("descargas")
    print("proceso terminado")

