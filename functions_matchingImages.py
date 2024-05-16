import numpy as np 
import cv2
import sqlite3
import json

from functions_extractionDonnes import load_images

def extraction_descriptors_keypoints_imagesName(path: str, db_name: str, table_name_descriptors: str, table_name_keypoints: str):
    # Connexion au base de donnees
    connectionDB = sqlite3.connect(db_name)
    cursor = connectionDB.cursor()

    images_name = []
    descriptors = []
    keypoints = []
    for i in range(1, len(load_images(path))): # les dossiers et ficher cachees sont innclus
        # Execute an SQL SELECT statement to retrieve the vector
        select_query = f"SELECT image_name, descriptor FROM {table_name_descriptors} WHERE id = {i}"
        cursor.execute(select_query)
        
        # Fetch the result (vector) from the database
        result_descriptor = cursor.fetchone()
        
        # Check if the result is not None and process the vector
        if result_descriptor:
            image_name, descriptor_str = result_descriptor
            descriptor = np.array(json.loads(descriptor_str), np.float32)
            descriptors.append(descriptor)
            images_name.append(image_name)

            keypoint = []
            # Execute an SQL SELECT statement to retrieve the vector
            select_query = f"SELECT x, y, size, angle, response, octave, class_id FROM {table_name_keypoints} WHERE image_name = '{image_name}'"
            cursor.execute(select_query)
    
            # Fetch the result (vector) from the database
            for row in cursor.fetchall():
                x, y, size, angle, response, octave, class_id = row
                keypoint.append(cv2.KeyPoint(x, y, size, angle, response, octave, class_id))

            keypoints.append(keypoint)            
        else:
            print("Vector descriptor not found in the database.")
            
    # Comit the changes
    connectionDB.commit()

    # Close the connection
    connectionDB.close()
    
    return descriptors, keypoints, images_name

def match(image_path, gray_scale, images_name, descriptors, keypoints):
    # Initialiser le détecteur SIFT
    sift = cv2.SIFT_create()

    # Trouver les points clés et les descripteurs avec SIFT
    keypoint_test, descriptor_test = sift.detectAndCompute(gray_scale, None)

    # Initialiser la correspondance Brute Force avec la distance euclidienne
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Matcher les descripteurs
    grather_than_50 = []
    for image_name, descriptor, keypoint in zip(images_name, descriptors, keypoints):
        # Matcher les descripteurs
        match = bf.match(descriptor_test, descriptor)
        # Trier les correspondances
        match = sorted(match, key=lambda match:match.distance)
        # Calculer le taux de correspondance en pourcentage
        match_rate = len(match) / min(len(keypoint_test), len(keypoint)) * 100
        
        # Detecter les 3 positions
        # Si le taux de correspondance est inférieur à 50%, considérez-le comme non correspondant
        if match_rate > 50:
            image_cv2 = cv2.imread("Datasets/"+image_name)
            # Dessiner les correspondances
            match_image = cv2.drawMatches(image_path, keypoint_test, image_cv2, keypoint, match[:100], None)
            grather_than_50.append((match_image, match_rate, image_name))
        
    if grather_than_50 == None:
        return None
         
    grather_than_50_sorted = sorted(grather_than_50, key=lambda x: x[1], reverse=True)
        
    return grather_than_50_sorted[:3]

def show_match(match_image, match_rate):
    # Redimensionner l'image pour la rendre plus petite
    scale_percent = 50  # pour une réduction de 50%
    width = int(match_image.shape[1] * scale_percent / 100)
    height = int(match_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(match_image, dim, interpolation=cv2.INTER_AREA)

    # Define text parameters
    text = f"Matching Rate: {match_rate:.2f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    text_color = (255, 255, 255)  # White color

    # Get text size
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # Calculate text position
    text_x = (resized_image.shape[1] - text_size[0]) // 2  # Center horizontally
    text_y = resized_image.shape[0] - 20  # 20 pixels above the bottom

    # Add text to the image
    cv2.putText(resized_image, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

    return resized_image