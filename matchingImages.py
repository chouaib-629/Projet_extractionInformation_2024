from functions_matchingImages import extraction_descriptors_keypoints_imagesName, match, show_match
from functions_extractionDonnes import gray_scaler
import cv2

def matchingImages(path, db_name, table_name_descriptors, table_name_keypoints, image_test):
    # tmp value 
    width = 768
    height = 576

    descriptors , keypoints, images_name = extraction_descriptors_keypoints_imagesName(path, db_name, table_name_descriptors, table_name_keypoints)
    
    image_test = cv2.imread(image_test)
    image_test = cv2.resize(image_test, (width, height))
    
    gray_scale = gray_scaler(image_test)    

    tmp = match(image_test, gray_scale, images_name, descriptors, keypoints)

    if tmp:
        matched_image, match_rate, image_name_1 = tmp[0]
        resized_1 = show_match(matched_image, match_rate)
        matched_image, match_rate, image_name_2 = tmp[1]
        resized_2 = show_match(matched_image, match_rate)
        matched_image, match_rate, image_name_3 = tmp[2]
        resized_3 = show_match(matched_image, match_rate)
    else:
        return None
    
    return ((resized_1, image_name_1), (resized_2, image_name_2), (resized_3, image_name_3))
    
if __name__ == "__main__":
    
    # Call for extraction information function
    matchingImages("Datasets/", "ma_base.db", "descriptors", "keypoints", "Datasets/004L_1.png")