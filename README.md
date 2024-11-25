# **SnapSort: Image Organizer and Category Generator**

SnapSort is an intelligent image organization tool powered by **OpenAI GPT-4** and **BLIP Image Captioning**. It analyzes images in a folder, generates captions, creates meaningful categories, and organizes the images into corresponding folders—all with a single click!

---

## **Features**
- **AI-Powered Categorization:**
  - Uses BLIP for image captioning.
  - Employs OpenAI GPT-4 to generate logical categories based on user input.
- **Customizable Logic:**
  - Organize images based on:
    - `Place`: Categories focus on locations.
    - `People`: Categories based on people in images.
    - `Do it for me`: Fully AI-driven categorization.
- **Seamless Organization:**
  - Automatically creates folders for categories.
  - Moves images into their respective folders.

---

## **How It Works**
1. **Input**:
   - Provide the folder path containing your images.
   - Select the number of categories to generate.
   - Choose a logic (`Place`, `People`, or `Do it for me`).
2. **Processing**:
   - BLIP generates captions for all images.
   - GPT-4 categorizes images based on the captions and selected logic.
3. **Output**:
   - Categories are created as folders.
   - Images are moved into their respective category folders.

---

### **Install Dependencies**
Ensure you have Python 3.9+ installed.

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### **Set Up OpenAI API Key**
1. Sign up or log in to [OpenAI](https://platform.openai.com/).
2. Obtain your API key.
3. Add the API key securely:
   - Create a `.streamlit/secrets.toml` file:
     ```bash
     mkdir .streamlit
     echo 'OPENAI_API_KEY = "your_openai_api_key_here"' > .streamlit/secrets.toml
     ```

---

## **Usage**
1. Run the app:
   ```bash
   streamlit run image_organizer.py
   ```
2. Open the app in your browser (default: `http://localhost:8501`).
3. Follow these steps in the app:
   - Enter the folder path containing your images.
   - Specify the number of categories to generate.
   - Choose a categorization logic (`Place`, `People`, or `Do it for me`).
   - Click **Organize** to process and organize the images.

---

## **Example Workflow**
1. Input folder path: `/Users/username/Pictures`.
2. Select number of categories: `5`.
3. Choose logic: `Place`.
4. Output:
   - Categories: `Beach`, `Mountains`, `City`, etc.
   - Images organized into corresponding folders.

---

## **Technologies Used**
- **Streamlit:** Interactive user interface.
- **BLIP (Salesforce):** Image captioning.
- **OpenAI GPT-4:** Natural language processing for categorization.
- **Python:** Backend logic and file handling.

---
