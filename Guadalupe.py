import ee
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# Inicializar la API de Earth Engine
try:
    ee.Initialize(project='proyectocuvallesmanchau25')
    print("Earth Engine inicializado correctamente")
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='proyectocuvallesmanchau25')
    print("Nueva autenticación realizada")

# Cargar el shapefile usando geopandas
shapefile_path = 'Recursos/2020_Guadalupe.shp'  # Cambia esto a la ruta de tu archivo .shp
gdf = gpd.read_file(shapefile_path)

# Obtener la geometría del shapefile
geometry = ee.Geometry.Polygon(gdf.geometry.values[0]['coordinates'])

# Definir los años de interés
years = [1985, 1995, 2005, 2015, 2025]

# Lista para almacenar las imágenes
images = []

# Función para obtener imágenes de un año específico
def get_image(year):
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    image_collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR') \
        .filterDate(start_date, end_date) \
        .filterBounds(geometry) \
        .mean()  # Promediar las imágenes del año
    return image_collection

# Obtener imágenes para cada año
for year in years:
    image = get_image(year)
    images.append(image)

# Visualizar las imágenes en un gráfico
fig, axs = plt.subplots(1, 5, figsize=(20, 5))

for ax, image, year in zip(axs, images, years):
    # Convertir la imagen a un array numpy
    img_array = np.array(image.getInfo()['bands'][0]['data'])
    ax.imshow(img_array, cmap='gray')
    ax.set_title(f'Imagen {year}')
    ax.axis('off')

# Mostrar la gráfica
plt.tight_layout()
plt.show()