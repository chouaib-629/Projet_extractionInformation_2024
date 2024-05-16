import streamlit as st
import os
from extractionDonnees import extractionDonnes
from matchingImages import matchingImages

def interface(path, db_name):
    st.set_page_config(
        page_title="Identification d'individus par l'iris",
        page_icon=":detective:",
        layout="centered",
    )
    
    # Check if the data base is difined
    db_exit = False
    parent_folder = os.path.dirname(__file__)
    for db in os.listdir(parent_folder):
        if db == db_name:
            db_exit = True
    
    # Create the data base if doesn't exist
    if not db_exit:
        extractionDonnes("ma_base.db", "Datasets/")
    
    st.title("Identification d'individus par l'iris")

    st.write("Upload an image to check if it exist in the data base.")

    uploaded_file = st.file_uploader("Upload Image", type=["png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Detect Match"):
            # Add your image forgery detection logic here
            result = matchingImages(path, db_name, "descriptors", "keypoints", "Datasets/"+uploaded_file.name)

            st.subheader("Match Detection Result:")
            if result:
                st.write("Hello Mr: ", (result[0])[1][:3])
                for image in result:
                    caption_text = f'id: {image[1][:3]}  side: {"Left" if image[1][3] == "L" else "Right"}  position: {image[1][5]}'
                    st.image(image[0], caption=caption_text)
            else:
                st.write("Image doesn't exist in the data base")    
        
if __name__ == "__main__":
    interface("Datasets/", "ma_base.db")
