from functions_extractionDonnes import lire_images_cv2, load_images, list_gray_scaler, extraire_descriptors_keypoints, table_descriptors, table_keypoints

def extractionDonnes(db_name: str, path_datasets: str) -> None:
    images_dir = load_images(path_datasets)
    
    images_cv2, width, height = lire_images_cv2(path_datasets, images_dir)

    gray_scales = list_gray_scaler(images_cv2)
    
    descriptors, keypoints = extraire_descriptors_keypoints(gray_scales)
    
    table_descriptors(db_name, images_dir, descriptors)
    
    table_keypoints(db_name, images_dir, keypoints)

if __name__ == "__main__":
    
    # Call for extraction information function
    extractionDonnes(db_name="ma_base.db", path_datasets="Datasets/")