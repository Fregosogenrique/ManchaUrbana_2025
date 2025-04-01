import ee
import geopandas as gpd
import geemap
import os
import matplotlib.pyplot as plt
import rasterio
import numpy as np
from datetime import datetime
import pandas as pd
from rasterio.plot import show

# Lista de municipios a procesar
municipios = ['Guadalupe', 'Morelos', 'Zacatecas', 'Vetagrande']


# Función para inicializar Earth Engine
def initialize_earth_engine():
    try:
        ee.Initialize(project='proyectocuvallesmanchau25')
        print("Earth Engine inicializado correctamente")
    except Exception as e:
        ee.Authenticate()
        ee.Initialize(project='proyectocuvallesmanchau25')
        print("Nueva autenticación realizada")


# Función para aplicar máscara de nubes
def mask_clouds(image):
    # Función para máscaras de nubes específicas por sensor
    if 'LC08' in image.get('system:id').getInfo():
        # Para Landsat 8, usar el bit de calidad para nubes
        qa = image.select('QA_PIXEL')
        cloud_mask = qa.bitwiseAnd(1 << 3).eq(0)  # Bit 3 es cloud
        return image.updateMask(cloud_mask)
    elif 'LE07' in image.get('system:id').getInfo() or 'LT05' in image.get(
            'system:id').getInfo() or 'LT04' in image.get('system:id').getInfo():
        # Para Landsat 7, 5, 4
        qa = image.select('QA_PIXEL')
        cloud_mask = qa.bitwiseAnd(1 << 5).eq(0)  # Bit 5 es cloud para estos sensores
        return image.updateMask(cloud_mask)
    else:
        return image


# Función para calcular NDBI (Normalized Difference Built-up Index)
def add_ndbi(image):
    # Seleccionar las bandas correctas según el sensor
    if 'LC08' in image.get('system:id').getInfo():
        # Landsat 8
        swir = image.select('B6')  # SWIR1
        nir = image.select('B5')  # NIR
    else:
        # Landsat 7, 5, 4
        swir = image.select('B5')  # SWIR1
        nir = image.select('B4')  # NIR

    ndbi = swir.subtract(nir).divide(swir.add(nir)).rename('NDBI')
    return image.addBands(ndbi)


# Función para obtener la mejor imagen para análisis urbano
def get_urban_analysis_image(year, geometry):
    # Definir fechas para la temporada seca (menos nubes)
    start_date = f'{year}-11-01'
    end_date = f'{year + 1}-04-30'

    # Seleccionar la colección Landsat correcta
    if year >= 2013:
        collection = 'LANDSAT/LC08/C02/T1_L2'  # Landsat 8 Surface Reflectance
        rgb_bands = ['SR_B4', 'SR_B3', 'SR_B2']  # Rojo, Verde, Azul
        swir_band = 'SR_B6'
        nir_band = 'SR_B5'
    elif year >= 1999:
        collection = 'LANDSAT/LE07/C02/T1_L2'  # Landsat 7 Surface Reflectance
        rgb_bands = ['SR_B3', 'SR_B2', 'SR_B1']
        swir_band = 'SR_B5'
        nir_band = 'SR_B4'
    elif year >= 1984:
        collection = 'LANDSAT/LT05/C02/T1_L2'  # Landsat 5 Surface Reflectance
        rgb_bands = ['SR_B3', 'SR_B2', 'SR_B1']
        swir_band = 'SR_B5'
        nir_band = 'SR_B4'
    else:
        collection = 'LANDSAT/LT04/C02/T1_L2'  # Landsat 4 Surface Reflectance
        rgb_bands = ['SR_B3', 'SR_B2', 'SR_B1']
        swir_band = 'SR_B5'
        nir_band = 'SR_B4'

    # Filtrar la colección
    image_collection = (ee.ImageCollection(collection)
                        .filterDate(start_date, end_date)
                        .filterBounds(geometry)
                        .filter(ee.Filter.lt('CLOUD_COVER', 10)))  # Menos del 10% de nubes

    # Verificar si hay imágenes disponibles
    if image_collection.size().getInfo() == 0:
        print(f"No hay imágenes disponibles para el año {year} con nubes < 10%. Intentando con umbral de 20%...")
        image_collection = (ee.ImageCollection(collection)
                            .filterDate(start_date, end_date)
                            .filterBounds(geometry)
                            .filter(ee.Filter.lt('CLOUD_COVER', 20)))

    # Si aún no hay imágenes, expandir el rango de fechas
    if image_collection.size().getInfo() == 0:
        print(f"Expandiendo rango de fechas para el año {year}...")
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
        image_collection = (ee.ImageCollection(collection)
                            .filterDate(start_date, end_date)
                            .filterBounds(geometry)
                            .filter(ee.Filter.lt('CLOUD_COVER', 30)))

    # Si aún no hay imágenes disponibles
    if image_collection.size().getInfo() == 0:
        print(f"No hay imágenes disponibles para el año {year}")
        return None

    # Ordenar por cobertura de nubes y tomar la mejor
    image = ee.Image(image_collection.sort('CLOUD_COVER').first())

    # Aplicar máscara de nubes adicional
    try:
        image = mask_clouds(image)
    except:
        print(f"No se pudo aplicar máscara de nubes para {year}")

    # Calcular NDBI para análisis urbano
    try:
        if 'LC08' in collection:
            swir = image.select(swir_band.replace('SR_', ''))
            nir = image.select(nir_band.replace('SR_', ''))
        else:
            swir = image.select(swir_band.replace('SR_', ''))
            nir = image.select(nir_band.replace('SR_', ''))

        ndbi = swir.subtract(nir).divide(swir.add(nir)).rename('NDBI')
        image = image.addBands(ndbi)
    except:
        print(f"No se pudo calcular NDBI para {year}")

    # Preparar imagen para visualización RGB
    visual_image = image.select(rgb_bands).rename(['R', 'G', 'B'])

    # Crear una imagen compuesta con RGB y NDBI
    composite = ee.Image.cat([visual_image, image.select('NDBI')])

    # Obtener la fecha de la imagen para el nombre del archivo
    img_date = ee.Date(image.get('system:time_start')).format('yyyy-MM-dd').getInfo()

    return {
        'image': composite,
        'date': img_date
    }


# Función principal
def process_municipios():
    # Inicializar Earth Engine
    initialize_earth_engine()

    # Años de interés
    years = [1985, 1995, 2005, 2015, 2023]  # Actualizado a 2023 ya que es más reciente que los datos disponibles

    # Directorio de salida
    output_dir = "descargas_urbanas"
    os.makedirs(output_dir, exist_ok=True)

    # DataFrame para almacenar resultados
    results = []

    for municipio in municipios:
        print(f"\nProcesando municipio: {municipio}")
        try:
            # Cargar el shapefile
            shapefile_path = f"Recursos/2020_{municipio}.shp"
            gdf = gpd.read_file(shapefile_path)

            # Crear geometría para Earth Engine
            coordinates = list(gdf.geometry.values[0].exterior.coords)
            geometry = ee.Geometry.Polygon(coordinates)

            # Descargar imágenes para cada año
            image_paths = []
            image_dates = []

            for year in years:
                try:
                    print(f"Procesando año {year}...")
                    image_data = get_urban_analysis_image(year, geometry)

                    if image_data:
                        # Generar nombre de archivo con fecha
                        file_path = os.path.join(output_dir, f"{municipio}_{year}_{image_data['date']}.tif")

                        # Exportar imagen RGB + NDBI
                        geemap.ee_export_image(
                            image_data['image'],
                            filename=file_path,
                            scale=30,
                            region=geometry,
                            file_per_band=False
                        )

                        image_paths.append(file_path)
                        image_dates.append(image_data['date'])

                        print(f"Imagen {year} ({image_data['date']}) descargada en {file_path}")

                        # Guardar información en resultados
                        results.append({
                            'Municipio': municipio,
                            'Año': year,
                            'Fecha_imagen': image_data['date'],
                            'Ruta_archivo': file_path
                        })
                    else:
                        print(f"No se descargó imagen para {year} debido a falta de datos.")
                except Exception as e:
                    print(f"Error al descargar la imagen del año {year}: {str(e)}")

            # Mostrar las imágenes descargadas si hay al menos una
            if image_paths:
                # Determinar el número de filas y columnas para el gráfico
                n_images = len(image_paths)
                n_cols = min(n_images, 3)  # Máximo 3 columnas
                n_rows = (n_images + n_cols - 1) // n_cols

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
                if n_images == 1:
                    axes = np.array([axes])
                axes = axes.flatten()

                for i, (ax, img_path, year, date) in enumerate(zip(axes, image_paths, years, image_dates)):
                    try:
                        with rasterio.open(img_path) as src:
                            # Mostrar imagen RGB
                            rgb = src.read([1, 2, 3])
                            rgb = np.transpose(rgb, (1, 2, 0))

                            # Normalizar para visualización
                            rgb_norm = np.zeros_like(rgb, dtype=np.float32)
                            for j in range(3):
                                band = rgb[:, :, j]
                                if band.max() > 0:
                                    rgb_norm[:, :, j] = band / band.max()

                            ax.imshow(rgb_norm)
                            ax.set_title(f"{municipio} - {year} ({date})")
                            ax.axis('off')

                            # Crear un segundo eje para el NDBI si está disponible
                            if src.count > 3:
                                ax2 = fig.add_subplot(n_rows, n_cols * 2, i * 2 + 2)
                                ndbi = src.read(4)
                                ndbi_cmap = plt.cm.RdYlGn_r  # Colormap invertido: rojo para áreas urbanas
                                im = ax2.imshow(ndbi, cmap=ndbi_cmap, vmin=-0.3, vmax=0.3)
                                ax2.set_title(f"NDBI {year}")
                                ax2.axis('off')
                                plt.colorbar(im, ax=ax2, shrink=0.7)

                    except Exception as e:
                        print(f"Error al mostrar la imagen {year}: {str(e)}")
                        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                        ax.axis('off')

                # Ocultar ejes vacíos
                for j in range(i + 1, len(axes)):
                    axes[j].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{municipio}_comparacion.png"), dpi=300)
                plt.show()

                # Guardar información del análisis
                df_results = pd.DataFrame(results)
                df_results.to_csv(os.path.join(output_dir, "resultados_analisis.csv"), index=False)

            else:
                print(f"No se pudieron descargar imágenes para {municipio}.")

        except Exception as e:
            print(f"Error procesando el municipio {municipio}: {str(e)}")


# Ejecutar el código
if __name__ == "__main__":
    process_municipios()