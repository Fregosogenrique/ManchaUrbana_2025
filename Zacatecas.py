import ee
import geopandas as gpd
import geemap
import os
import matplotlib.pyplot as plt
import rasterio
import numpy as np
municipios=['Guadalupe','Morelos','Zacatecas','Vetagrande']
for i in municipios:
    try:
        # Inicializar Earth Engine
        try:
            ee.Initialize(project='proyectocuvallesmanchau25')
            print("Earth Engine inicializado correctamente")
        except Exception as e:
            ee.Authenticate()
            ee.Initialize(project='proyectocuvallesmanchau25')
            print("Nueva autenticación realizada")


        # Cargar el shapefile
        shapefile_path = f"Recursos/2020_{i}.shp"
        gdf = gpd.read_file(shapefile_path)
        print("Inicio de descargas")
        coordinates = list(gdf.geometry.values[0].exterior.coords)
        geometry = ee.Geometry.Polygon(coordinates)

        # Definir los años de interés
        years = [1985, 1995, 2005, 2015, 2025]

        # Directorio de salida
        output_dir = "descargas"
        os.makedirs(output_dir, exist_ok=True)


        def get_image(year):
            start_date = f'{year}-01-01'
            end_date = f'{year}-12-31'

            # Seleccionar la colección Landsat correcta
            if year >= 2013:
                collection = 'LANDSAT/LC08/C02/T1_TOA'
            elif year >= 1999:
                collection = 'LANDSAT/LE07/C02/T1_TOA'
            elif year >= 1984:
                collection = 'LANDSAT/LT05/C02/T1_TOA'
            else:
                collection = 'LANDSAT/LT04/C02/T1_TOA'

            bands = ['B4', 'B3', 'B2']  # Rojo, Verde, Azul para imagen a color

            image_collection = (ee.ImageCollection(collection)
                                .filterDate(start_date, end_date)
                                .filterBounds(geometry))

            if image_collection.size().getInfo() == 0:
                print(f"No hay imágenes disponibles para el año {year}")
                return None

            image = image_collection.median().select(bands)
            return image


        # Descargar imágenes para cada año
        image_paths = []
        for year in years:
            try:
                image = get_image(year)
                if image:
                    file_path = os.path.join(output_dir, f"{i}_{year}.tif")
                    geemap.ee_export_image(image, filename=file_path, scale=30, region=geometry, file_per_band=False)
                    image_paths.append(file_path)
                    print(f"Imagen {year} descargada en {file_path}")
                else:
                    print(f"No se descargó imagen para {year} debido a falta de datos.")
            except Exception as e:
                print(f"Error al descargar la imagen del año {year}: {e}")

        # Mostrar las imágenes descargadas si hay al menos una
        if image_paths:
            fig, axes = plt.subplots(1, len(image_paths), figsize=(20, 5))
            if len(image_paths) == 1:
                axes = [axes]  # Asegurar que 'axes' sea iterable si solo hay una imagen

            for ax, img_path, year in zip(axes, image_paths, years):
                try:
                    with rasterio.open(img_path) as src:
                        img = src.read([1, 2, 3])  # Leer bandas en orden RGB
                        img = np.stack(img, axis=-1)  # Reorganizar dimensiones para mostrar como imagen
                    ax.imshow(img)
                    ax.set_title(f"{year}")
                    ax.axis('off')
                except Exception as e:
                    print(f"Error al cargar la imagen {year}: {e}")
            plt.show()
        else:
            print("No se pudieron descargar imágenes.")
    except:
        print("")