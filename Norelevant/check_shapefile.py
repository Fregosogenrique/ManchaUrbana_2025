import geopandas as gpd
import folium

# 1. Cargar el shapefile en EPSG:32613
shapefile_path = "Recursos/MG_Jalisco_2020_EPSG.shp"  # Reemplázalo con tu archivo
gdf = gpd.read_file(shapefile_path)

# Verificar el CRS actual
print("CRS original:", gdf.crs)

# Si el CRS no está definido, asignarlo manualmente
if gdf.crs is None or gdf.crs.to_epsg() != 32613:
    gdf.set_crs(epsg=32613, inplace=True)

# 2. Convertir a EPSG:4326
gdf = gdf.to_crs(epsg=4326)
print("CRS convertido:", gdf.crs)

# Guardar el nuevo shapefile convertido
output_path = "shapefile_convertido.shp"
gdf.to_file(output_path)
print(f"Shapefile convertido guardado en: {output_path}")

# 3. Mostrar en un mapa interactivo con Folium
# Obtener el centro del shapefile para centrar el mapa
centro = gdf.geometry.centroid.iloc[0]
m = folium.Map(location=[centro.y, centro.x], zoom_start=12)

# Agregar el shapefile al mapa
for _, row in gdf.iterrows():
    geo_json = row.geometry.__geo_interface__
    folium.GeoJson(geo_json, name="Shapefile Convertido").add_to(m)

# Guardar el mapa en un archivo HTML y mostrarlo
map_path = "mapa_interactivo.html"
m.save(map_path)
print(f"Mapa guardado en: {map_path}")

# Mostrar el mapa en el navegador
import webbrowser
webbrowser.open(map_path)