import os
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import openai
import json

# Securely fetch the OpenAI API key from Streamlit secrets or environment variables
openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

# Load the BLIP model for captioning
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model

processor, model = load_blip_model()

st.title("SnapSort Agent")
st.subheader("Effortlessly organize your photos smartly")

# Function to generate a caption for an image
def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt").to(model.device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        st.error(f"Error generating caption for {image_path}: {e}")
        return "Unknown"

# Function to process images and generate captions
def process_images(folder_path):
    metadata = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            #st.write(f"Processing image: {filename}")
            caption = generate_caption(image_path)
            #st.write(f"Generated caption: {caption}")
            metadata[filename] = caption
    return metadata

# Function to generate categories using OpenAI GPT-4 API
def generate_categories_from_openai(metadata, num_categories, logic_selection):
    """
    Generate meaningful category names using OpenAI's GPT-4 model.
    Args:
        metadata (dict): Dictionary with filenames as keys and captions as values.
        num_categories (int): Number of unique categories to generate.
        logic_selection (str): Categorization logic (Place, People, Do it for me).
    Returns:
        str: JSON string containing image names and categories.
    """
    # Combine captions into a formatted string with filenames
    formatted_captions = "\n".join([f"{filename}: {caption}" for filename, caption in metadata.items()])

    # Custom instructions based on logic selection
    logic_instructions = {
        "Place": "Base the categories on the places observed in the images.",
        "People": "Base the categories on the unique face identified in the images.",
        "Do it for me": "Create meaningful and logical categories based on the image descriptions."
    }

    prompt = (
        f"Here are descriptions of images and their filenames:\n\n{formatted_captions}\n\n"
        f"{logic_instructions[logic_selection]}\n"
        f"STRICTLY create {num_categories} unique, concise category names (max 1-2 words each) "
        f"that can logically classify these images. Return only the list of categories."
        f"Output should be a json array with image name mapped to category name generated - the attribute names would imageName, category"
    )

    # Call OpenAI's GPT-4 API
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates JSON mappings for images."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error generating categories: {e}")
        return "[]"


# Function to organize images into category folders based on GPT-4 JSON output
def organize_images_from_json(folder_path, gpt4_output_json):
    """
    Organize images into folders based on categories from GPT-4 JSON output.
    Args:
        folder_path (str): Path to the folder containing images.
        gpt4_output_json (str): JSON string containing image names and categories.
    """
    try:
        # Parse the GPT-4 output JSON
        image_data = json.loads(gpt4_output_json)

        # Create a dictionary to group images by category
        category_map = {}
        for item in image_data:
            category = item["category"]
            image_name = item["imageName"]

            if category not in category_map:
                category_map[category] = []
            category_map[category].append(image_name)

        # Create folders and move images
        for category, images in category_map.items():
            category_folder = os.path.join(folder_path, category)
            os.makedirs(category_folder, exist_ok=True)  # Create category folder if it doesn't exist

            for image_name in images:
                source_path = os.path.join(folder_path, image_name)
                destination_path = os.path.join(category_folder, image_name)

                # Move the image file to the respective category folder
                if os.path.exists(source_path):  # Ensure the file exists before moving
                    os.rename(source_path, destination_path)
                else:
                    st.warning(f"Image file {image_name} not found in {folder_path}. Skipping.")

        st.success("Images have been organized into folders successfully!")

    except Exception as e:
        st.error(f"Error organizing images: {e}")


# Streamlit UI
folder_path = st.text_input("Enter the path to the folder containing images:")
num_categories = st.number_input(
    "Enter the number of categories to generate:",
    min_value=1,
    max_value=20,
    value=5,
    step=1
)

logic_selection = st.radio(
    "Select how to categorize the images:",
    ("Place", "People", "Do it for me"),
    index=2
)

if st.button("Organize"):
    if os.path.isdir(folder_path) and logic_selection:
        st.write("Processing images...")
        metadata = process_images(folder_path)
        st.write("Metadata generated successfully!")
        #st.json(metadata)  # Display captions as JSON

        st.write("Generating categories from captions...")
        gpt4_output_json = generate_categories_from_openai(metadata, num_categories, logic_selection)
        st.write("Generated Categories!")
        #st.json(gpt4_output_json)  # Display GPT-4 JSON response

        st.write("Organizing images into folders based on categories...")
        organize_images_from_json(folder_path, gpt4_output_json)
    else:
        st.error("Please provide a valid folder path and select a categorization logic.")