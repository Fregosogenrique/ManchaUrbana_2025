import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap


def detectar_areas_urbanas_rgb(img_path, n_clusters=4, umbral_intensidad=0.3):
    """
    Detecta áreas urbanas utilizando solo bandas RGB mediante clustering.
    Las áreas urbanas tienden a tener mayor reflectancia/brillo y menor vegetación.
    """
    # Leer las bandas RGB
    with rasterio.open(img_path) as src:
        # Verificar cuántas bandas tiene la imagen
        num_bands = src.count
        print(f"La imagen {img_path} tiene {num_bands} bandas")

        # Leer todas las bandas disponibles
        img = src.read()
        meta = src.meta

    # Asegurarse de que tenemos al menos 3 bandas
    if num_bands < 3:
        raise ValueError(f"La imagen {img_path} no tiene suficientes bandas (tiene {num_bands})")

    # Usar las primeras 3 bandas como RGB
    red = img[0].astype(float)
    green = img[1].astype(float)
    blue = img[2].astype(float)

    # Calcular brillo promedio (intensidad)
    intensidad = (red + green + blue) / 3.0

    # Calcular un índice de vegetación sencillo basado en RGB
    # En RGB, la vegetación tiene mayor componente verde que rojo y azul
    denominador = red + green + blue
    mask = denominador > 0

    # Proporción verde normalizada
    proporcion_verde = np.zeros_like(green)
    proporcion_verde[mask] = green[mask] / denominador[mask]

    # Proporción roja normalizada
    proporcion_roja = np.zeros_like(red)
    proporcion_roja[mask] = red[mask] / denominador[mask]

    # Crear características para clustering
    # Aplanar arrays y combinar en una matriz de características
    valid_pixels = ~np.isnan(intensidad) & mask & (intensidad > 0)
    features = np.vstack([
        intensidad[valid_pixels].flatten(),
        proporcion_verde[valid_pixels].flatten(),
        proporcion_roja[valid_pixels].flatten()
    ]).T

    # Aplicar clustering K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features)

    # Reconstruir la imagen de clusters
    cluster_image = np.zeros_like(intensidad, dtype=int)
    cluster_image[valid_pixels] = clusters

    # Identificar clusters urbanos (mayor intensidad y menor verdor)
    centros = kmeans.cluster_centers_
    # Los centros contienen: [intensidad, prop_verde, prop_roja] para cada cluster

    # Calcular un puntaje urbano = intensidad - proporción_verde
    puntajes_urbanos = centros[:, 0] - centros[:, 1]

    # Identificar el cluster con mayor puntaje urbano
    cluster_urbano = np.argmax(puntajes_urbanos)

    # Crear máscara de áreas urbanas
    areas_urbanas = (cluster_image == cluster_urbano)

    # Agrupar píxeles urbanos cercanos y eliminar ruido
    from scipy import ndimage
    # Aplicar operación de cierre morfológico
    areas_urbanas = ndimage.binary_closing(areas_urbanas, structure=np.ones((3, 3)))
    # Eliminar pequeños grupos de píxeles
    areas_urbanas = ndimage.binary_opening(areas_urbanas, structure=np.ones((3, 3)))

    print(f"Cluster identificado como urbano: {cluster_urbano}")
    print(f"Características del cluster urbano: Intensidad={centros[cluster_urbano][0]:.3f}, "
          f"Prop. Verde={centros[cluster_urbano][1]:.3f}, Prop. Roja={centros[cluster_urbano][2]:.3f}")

    return areas_urbanas, meta, img[:3]


def calcular_area_urbana(areas_urbanas, meta):
    """
    Calcula el área urbana en kilómetros cuadrados
    """
    pixel_area = abs(meta['transform'][0] * meta['transform'][4])  # área del pixel en unidades cuadradas
    area_total = np.sum(areas_urbanas) * pixel_area / 1000000  # convertir a km²
    return area_total


def analizar_crecimiento_urbano(dir_imagenes, anios=None):
    """
    Analiza el crecimiento urbano de una serie de imágenes Landsat usando solo RGB
    """
    if anios is None:
        # Buscar todos los archivos .tif
        imagenes = [f for f in os.listdir(dir_imagenes) if f.endswith('.tif')]
        imagenes.sort()  # Ordenar por nombre
    else:
        # Usar los años específicos
        imagenes = [f"Jalisco_{anio}.tif" for anio in anios]

    resultados = []

    # Analizar cada imagen
    for img_filename in imagenes:
        img_path = os.path.join(dir_imagenes, img_filename)
        try:
            # Extraer año del nombre del archivo
            anio = int(img_filename.split('_')[1].split('.')[0])

            # Detectar áreas urbanas usando solo RGB
            print(f"Procesando imagen {img_filename}...")
            areas_urbanas, meta, rgb = detectar_areas_urbanas_rgb(img_path)

            # Calcular área urbana
            area_km2 = calcular_area_urbana(areas_urbanas, meta)

            resultados.append({
                'año': anio,
                'imagen': img_filename,
                'areas_urbanas': areas_urbanas,
                'area_km2': area_km2,
                'meta': meta,
                'rgb': rgb
            })

            print(f"Año {anio}: {area_km2:.2f} km² de área urbana")

        except Exception as e:
            print(f"Error al procesar {img_filename}: {e}")
            import traceback
            traceback.print_exc()

    return resultados


def visualizar_crecimiento(resultados, dir_salida):
    """
    Crea visualizaciones del crecimiento urbano con datos normalizados para mostrar
    correctamente el incremento del área urbana a lo largo del tiempo
    """
    os.makedirs(dir_salida, exist_ok=True)

    # Ordenar resultados por año
    resultados.sort(key=lambda x: x['año'])

    if len(resultados) < 2:
        print("Se requieren al menos dos imágenes para comparar el crecimiento")
        return

    # Obtener la primera y última imagen
    primera = resultados[0]
    ultima = resultados[-1]

    # Crear figura de comparación
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Colores para áreas urbanas
    colors = ['white', 'red']
    cmap = ListedColormap(colors)

    # Visualizar primera imagen - RGB
    rgb_primera = np.transpose(primera['rgb'], (1, 2, 0))
    # Normalizar para visualización
    max_valor = np.percentile(rgb_primera, 98)  # Usar percentil para evitar valores extremos
    rgb_primera_norm = np.clip(rgb_primera / max_valor, 0, 1)
    axes[0, 0].imshow(rgb_primera_norm)
    axes[0, 0].set_title(f"RGB {primera['año']}")
    axes[0, 0].axis('off')

    # Visualizar primera imagen - Áreas urbanas
    axes[0, 1].imshow(primera['areas_urbanas'], cmap=cmap)
    axes[0, 1].set_title(f"Áreas urbanas {primera['año']}")
    axes[0, 1].axis('off')

    # Visualizar última imagen - RGB
    rgb_ultima = np.transpose(ultima['rgb'], (1, 2, 0))
    # Normalizar para visualización
    max_valor = np.percentile(rgb_ultima, 98)  # Usar percentil para evitar valores extremos
    rgb_ultima_norm = np.clip(rgb_ultima / max_valor, 0, 1)
    axes[1, 0].imshow(rgb_ultima_norm)
    axes[1, 0].set_title(f"RGB {ultima['año']}")
    axes[1, 0].axis('off')

    # Visualizar última imagen - Áreas urbanas
    axes[1, 1].imshow(ultima['areas_urbanas'], cmap=cmap)
    axes[1, 1].set_title(f"Áreas urbanas {ultima['año']}")
    axes[1, 1].axis('off')

    # Visualizar cambio
    cambio = ultima['areas_urbanas'].astype(int) - primera['areas_urbanas'].astype(int)
    # 1: nuevo urbano, 0: sin cambio, -1: pérdida de área urbana

    # Crear mapa de colores personalizado
    cmap_cambio = ListedColormap(['blue', 'white', 'red'])  # azul = pérdida, blanco = sin cambio, rojo = ganancia

    axes[0, 2].imshow(cambio + 1, cmap=cmap_cambio, vmin=0, vmax=2)
    axes[0, 2].set_title(f"Cambio {primera['año']} - {ultima['año']}")
    axes[0, 2].axis('off')

    # Gráfico de barras del área urbana a lo largo del tiempo
    ax_bar = axes[1, 2]
    years = [r['año'] for r in resultados]
    areas = [r['area_km2'] for r in resultados]

    # Configurar el eje Y para comenzar en cero o ligeramente por debajo del valor mínimo
    # para mostrar claramente el crecimiento
    min_area = min(areas) * 0.9  # Comenzar un poco por debajo del mínimo
    ax_bar.set_ylim(min_area, max(areas) * 1.1)  # Dar espacio arriba para las etiquetas

    ax_bar.bar(years, areas, color='darkred')
    ax_bar.set_xlabel('Año')
    ax_bar.set_ylabel('Área urbana (km²)')
    ax_bar.set_title('Evolución del área urbana')
    ax_bar.grid(True, linestyle='--', alpha=0.7)

    # Añadir las cifras de crecimiento
    for i, v in enumerate(areas):
        ax_bar.text(years[i], v + (max(areas) - min_area) * 0.02, f"{v:.1f}", ha='center')

    # Mostrar crecimiento en porcentaje y valores absolutos
    if len(areas) > 1:
        crecimiento_total = ((areas[-1] - areas[0]) / areas[0]) * 100
        incremento_absoluto = areas[-1] - areas[0]

        texto_info = f"Crecimiento total: {crecimiento_total:.1f}%\n"
        texto_info += f"Incremento: {incremento_absoluto:.1f} km²"

        ax_bar.text(0.5, 0.9, texto_info,
                    transform=ax_bar.transAxes, ha='center',
                    bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(dir_salida, f"comparacion_{primera['año']}_{ultima['año']}.png"), dpi=300)

    print(f"Imagen de comparación guardada en {dir_salida}")
    return fig


def ejecutar_analisis(dir_imagenes="Jalisco", dir_salida="Resultados"):
    """
    Función principal para ejecutar el análisis de crecimiento urbano
    """
    print("Iniciando análisis de crecimiento urbano...")

    # Analizar imágenes
    resultados = analizar_crecimiento_urbano(dir_imagenes)

    if resultados:
        # Visualizar resultados
        fig = visualizar_crecimiento(resultados, dir_salida)

        # Mostrar figura
        plt.show()

        print("Análisis completado con éxito.")
        return resultados
    else:
        print("No se pudieron analizar las imágenes.")
        return None


# Ejemplo de uso
if __name__ == "__main__":
    ejecutar_analisis()