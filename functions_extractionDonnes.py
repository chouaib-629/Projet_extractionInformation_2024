import numpy as np 
import cv2
import sqlite3
import json
import os

def load_images(path_datasets: str):
    return [file for file in os.listdir(path_datasets) if os.path.isfile(os.path.join(path_datasets, file)) and not file.startswith('.')]

def lire_images_cv2(path_datasets: str, images_dir):
    # Charger les images
    matrix_images_cv2 = []
    height = np.inf
    width = np.inf
    for image in images_dir:
        path = path_datasets + image
        read_image = cv2.imread(path)
        matrix_images_cv2.append(read_image)

        if read_image.shape[0] < height:
            height = read_image.shape[0]
        
        if read_image.shape[1] < width:
            width = read_image.shape[1]

    # Redimensionner les images pour qu'elles aient la même dimension
    for i in range(len(matrix_images_cv2)):
        matrix_images_cv2[i] = cv2.resize(matrix_images_cv2[i], (width, height))
        
    return matrix_images_cv2, width, height

def gray_scaler(image):
    # Convertir en niveaux de gris  
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Redimensionner les images
    gray_scale = cv2.resize(gray_scale, (500, 500))
    # Appliquer la suppression du bruit
    gray_scale = cv2.GaussianBlur(gray_scale, (9, 9), 1)
    # Égalisation de l'histogramme pour améliorer le contraste
    gray_scale = cv2.equalizeHist(gray_scale)
    # Filtrage de la morphologie mathématique pour supprimer les petits éléments
    kernel = np.ones((5, 5), np.uint8)
    gray_scale = cv2.morphologyEx(gray_scale, cv2.MORPH_OPEN, kernel) 
    
    return gray_scale

def list_gray_scaler(matrix_images_cv2):
    gray_scales = []
    
    for matrix_image in matrix_images_cv2:
        gray_scale = gray_scaler(matrix_image)
        gray_scales.append(gray_scale)
    
    return gray_scales

def extraire_descriptors_keypoints(gray_scales):
    # Initialiser le détecteur SIFT
    sift = cv2.SIFT_create()

    # Trouver les points clés et les descripteurs avec SIFT
    descriptors = []
    keypoints = []
    for gray_scale in gray_scales:
        keypoint, descriptor = sift.detectAndCompute(gray_scale, None)
        descriptors.append(descriptor)
        keypoints.append(keypoint)
        
    return descriptors, keypoints

def table_descriptors(db_name: str, images_dir, descriptors):
    # Creation du base de donnees
    connectionDB = sqlite3.connect(db_name)
    cursor = connectionDB.cursor()
    
    # Supprssion du table if exists
    delete_table_query = f"DROP TABLE IF EXISTS descriptors"
    cursor.execute(delete_table_query)
    
    # Convert each descriptor to a JSON string
    descriptors_json_list = [json.dumps(descriptor.tolist()) for descriptor in descriptors]
    
    # Creation du table "descriptors"
    table_name_descriptors = "descriptors" # IF NOT EXISTS
    
    create_table_query = f"CREATE TABLE {table_name_descriptors} (id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,  image_name TEXT, descriptor TEXT)"
    cursor.execute(create_table_query)

    # insertion des vecteurs
    insert_query = f"INSERT INTO {table_name_descriptors} (image_name, descriptor) VALUES (?, ?)"

    # Execute the INSERT statement for each discriptor
    cursor.executemany(insert_query,
                    [(image, descriptor_json) for image, descriptor_json in zip(images_dir, descriptors_json_list)])
    
    # Comit the changes
    connectionDB.commit()

    # Close the connection
    connectionDB.close()


def table_keypoints(db_name: str, images_dir, keypoints):
    # Creation du base de donnees
    connectionDB = sqlite3.connect(db_name)
    cursor = connectionDB.cursor()
    
    # Supprssion du table if exists
    delete_table_query = f"DROP TABLE IF EXISTS keypoints"
    cursor.execute(delete_table_query)

    # Création d'une table pour stocker les keypoints
    table_name_keypoints = "keypoints"
    
    create_table_query = f"CREATE TABLE {table_name_keypoints} (id INTEGER PRIMARY KEY, image_name TEXT, x REAL, y REAL, size REAL, angle REAL, response REAL, octave INTEGER, class_id INTEGER)"
    cursor.execute(create_table_query)

    insert_query = f"INSERT INTO {table_name_keypoints} (image_name, x, y, size, angle, response, octave, class_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    
    # Insérer les keypoints dans la base de données
    for keypoint, image_name in zip(keypoints, images_dir):
        for kp in keypoint:
            x, y = kp.pt  # Accéder aux coordonnées x et y du keypoint
            size = kp.size
            angle = kp.angle
            response = kp.response
            octave = kp.octave
            class_id = kp.class_id
            
            cursor.execute(insert_query,
                    (image_name, x, y, size, angle, response, octave, class_id))
            
    # Comit the changes
    connectionDB.commit()

    # Close the connection
    connectionDB.close()