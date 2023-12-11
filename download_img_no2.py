import ee
import requests
import rasterio
import matplotlib.pyplot as plt

# Inicialize o Earth Engine
ee.Initialize()

# Define a Área de Interesse (AOI)
aoi = ee.Geometry.Polygon([
    [
        [-70.43180413064213, 1.2718437668951026],
        [-70.43180413064213, -10.386980564186288],
        [-45.98334782148049, -10.386980564186288],
        [-45.98334782148049, 1.2718437668951026]
    ]
])

# Seleciona a coleção de imagens do Sentinel-5P OFFL NO2
collection = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")\
    .filterDate('2022-01-01', '2022-12-31')\
    .filterBounds(aoi)\
    .select('tropospheric_NO2_column_number_density')

# Função para verificar a presença de dados
def has_sufficient_data(image):
    total_pixel_count = ee.Number(image.unmask().reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=aoi,
        scale=1000
    ).get('tropospheric_NO2_column_number_density'))

    valid_pixel_count = ee.Number(image.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=aoi,
        scale=1000
    ).get('tropospheric_NO2_column_number_density'))

    valid_data_ratio = valid_pixel_count.divide(total_pixel_count)
    return valid_data_ratio.gte(0.2)

# Encontra a primeira imagem com pelo menos 20% de dados válidos
valid_image = None
image_list = collection.toList(collection.size())

for i in range(image_list.size().getInfo()):
    image = ee.Image(image_list.get(i))
    if has_sufficient_data(image).getInfo():
        valid_image = image
        break

if valid_image:
    # Define os parâmetros para o download
    download_params = {
        'scale': 1000,
        'region': aoi.getInfo()['coordinates'],
        'format': 'GEO_TIFF'
    }

    # Obtém o URL de download
    download_url = valid_image.getDownloadURL(download_params)

    # Baixa o arquivo usando requests
    response = requests.get(download_url)
    with open('temp_image.tif', 'wb') as file:
        file.write(response.content)

    # Lendo a imagem baixada com rasterio
    with rasterio.open('temp_image.tif') as src:
        data = src.read(1)
        plt.figure(figsize=(10, 10))
        plt.imshow(data, cmap='viridis')
        plt.colorbar(label='Concentração de NO2')
        plt.title('Sentinel-5P NO2')
        plt.show()
else:
    print("Não foram encontradas imagens com pelo menos 20% de dados válidos para o período e área especificados.")
