import re
import glob
from qgis.core import QgsApplication, QgsRasterLayer
import numpy as np
import sys
import os
try:
    sys.path.append('/Applications/QGIS.app/Contents/Resources/python')

    # Ahora intenta importar QGIS
    try:
        from qgis.core import *

        print(QgsApplication.showSettings())
    except ImportError as e:
        print(f"Error al importar QGIS: {e}")
except ImportError:
    print("Error al importar QGIS, verifica la ruta de instalación")
# Iniciar QGIS dentro de un script Python
QgsApplication.setPrefixPath("/usr", True)
qgs = QgsApplication([], False)
qgs.initQgis()
def buscar_imagenes():
    # Ruta de la carpeta de descargas
    ruta_descargas = os.path.join(os.getcwd(), 'descargas')
    # Expresión regular para buscar archivos .tif con "Guadalupe" en el nombre y extraer el año
    patron = re.compile(r'Guadalupe.*?(\d{4}).*?\.tif$', re.IGNORECASE)
    # Buscar archivos que coincidan con el patrón
    imagenes = glob.glob(os.path.join(ruta_descargas, '*.tif'))
    # Filtrar y ordenar imágenes por año
    imagenes_filtradas = []
    for imagen in imagenes:
        nombre_archivo = os.path.basename(imagen)
        coincidencia = patron.search(nombre_archivo)
        if coincidencia:
            year = int(coincidencia.group(1))
            imagenes_filtradas.append((imagen, year))
    # Ordenar por año
    imagenes_filtradas.sort(key=lambda x: x[1])
    return [img[0] for img in imagenes_filtradas]


def cargar_imagen_tif(ruta):
    rl = QgsRasterLayer(ruta, "Capa de Imagen")
    if not rl.isValid():
        print("No se pudo cargar {ruta}")
    else:
        print("Imagen {ruta} cargada exitosamente.")
    return rl
def comparar_imagenes(imagenes):
    cambios_notables = []
    umbral = 50  # Umbral de diferencia de píxeles

    for i in range(len(imagenes) - 1):
        img1 = cargar_imagen_tif(imagenes[i])
        img2 = cargar_imagen_tif(imagenes[i + 1])


        # Convertir imágenes a matrices numpy
        # Convertir imágenes a matrices numpy (suponiendo uso de PyQGIS)
        data1 = img1  # Implementar extracción de datos de raster
        data2 = img2  # Implementar extracción de datos de raster

        # Calcular la diferencia y aplicar el umbral, ahora directamente sobre imágenes RGB
        diferencia = np.abs(data1 - data2)
        if np.any(diferencia > umbral):
            cambios_notables.append((imagenes[i], imagenes[i + 1]))

    return cambios_notables

def generar_informe(cambios):
    if not cambios:
        print("No se encontraron cambios notables entre las imágenes.")
        return

    print("Imágenes con cambios notables:")
    for img1, img2 in cambios:
        print(f"- {os.path.basename(img1)} y {os.path.basename(img2)}")

if __name__ == '__main__':
    imagenes = buscar_imagenes()
    cambios = comparar_imagenes(imagenes)
    generar_informe(cambios)

# Finaliza la instancia de QGIS correctamente
qgs.exitQgis()
