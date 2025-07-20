
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import re
import pandas as pd
import time
from io import BytesIO
import joblib
import pandas as pd
import requests
import numpy as np
import cv2
from sklearn.cluster import KMeans
from io import BytesIO
import base64
import csv

def rgb_to_hsv(rgb_color):
    rgb_color = np.array([rgb_color], dtype=np.uint8)
    hsv_color = cv2.cvtColor(rgb_color.reshape(1, 1, 3), cv2.COLOR_RGB2HSV).reshape(3,)
    return hsv_color

def luminance(color):
    r, g, b = [x / 255.0 for x in color]
    r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def contrast_ratio(color1, color2):
    L1 = luminance(color1)
    L2 = luminance(color2)
    if L1 > L2:
        return (L1 + 0.05) / (L2 + 0.05)
    else:
        return (L2 + 0.05) / (L1 + 0.05)

def find_max_contrast(colors):
    max_contrast = 0
    color_pair = None
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            contrast = contrast_ratio(colors[i], colors[j])
            if contrast > max_contrast:
                max_contrast = contrast
                color_pair = (colors[i], colors[j])
    return max_contrast, color_pair

def average_brightness(colors):
    brightness_values = [luminance(color) for color in colors]
    avg_brightness = np.mean(brightness_values)
    return avg_brightness, brightness_values

def classify_temperature(hsv_colors):
    warm_count = 0
    cool_count = 0

    for color in hsv_colors:
        hue = color[0]  # Получаем оттенок (Hue)
        if hue < 90:  # Тёплые цвета (красный, жёлтый, оранжевый)
            warm_count += 1
        else:  # Холодные цвета (синий, зелёный, фиолетовый)
            cool_count += 1

    return 'Warm' if warm_count >= cool_count else 'Cool'
def find_dominant_color_from_url(img, k=7):

    try:
     
        img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), cv2.IMREAD_COLOR)

        
        if img is None:
            print('error')

            return [-1] * 11

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_resized = cv2.resize(img_rgb, (img_rgb.shape[1] // 2, img_rgb.shape[0] // 2))
        
        pixels = img_resized.reshape(-1, 3)
        
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_
        
        colors = colors.round(0).astype(int)
        
        mean_color = np.mean(colors, axis=0).round(0).astype(int)
        
        hsv_colors = [rgb_to_hsv(color) for color in colors]
        
        temperature = classify_temperature(hsv_colors)
        
        max_contrast, color_pair = find_max_contrast(colors)
        av_contrast = np.mean([contrast_ratio(colors[i], colors[j]) for i in range(len(colors)) for j in range(i + 1, len(colors))])
        
        avg_brightness, brightness_values = average_brightness(colors)
        
        max_brightness = np.max(brightness_values)
        #face_count = count_faces(response)
        #print(face_count)
        mean_data = np.concatenate((mean_color, rgb_to_hsv(mean_color), [max_contrast, av_contrast, avg_brightness, max_brightness, temperature]))
        
        # print(f"Mean data: {mean_data}")
        # print(f"Temperature: {temperature}")
        
        return mean_data
    except Exception as e:
        print(f"Error processing image from {img}: {e}")
        return [-1] * 11

async def parse_site(session, link_inf):
    id, title, poster_path, averageRating = link_inf
    image_data = None
    tmp_list=None
    try:
        async with session.get('https://image.tmdb.org/t/p/w500/' + poster_path) as response:
            if response.status == 200:
                image_data = BytesIO(await response.read())
                tmp_list=find_dominant_color_from_url(image_data)
    except Exception as e:
        print(f"Ошибка загрузки изображения: {e}")
        print(poster_path, "- постер")
        return [id, title, poster_path, averageRating, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

    main_list = [id, title, poster_path, averageRating]
    if tmp_list is not None:

        if isinstance(tmp_list, list):
            main_list.extend(tmp_list)   

        else:
            main_list.extend(tmp_list.tolist())
    else: 
        return [id, title, poster_path, averageRating, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

    return main_list

async def process_links(links):
    async with aiohttp.ClientSession() as session:
        tasks = [parse_site(session, link) for link in links]
        return await asyncio.gather(*tasks)

end = 200  

df = pd.read_csv('poster_data.csv')  
start = 0
result_data = []  


while True:
    with open('picture_data_image.csv', 'a', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        start_time = time.time()
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(process_links(df[start:end+start].values.tolist()))

        for result in results:
            csvwriter.writerow(result)

        end_time = time.time()
        execution_time = end_time - start_time
        start=start+end
        print(start)
        print(f"Время выполнения запросов: {execution_time} секунд")



