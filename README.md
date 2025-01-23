# Semantic Image Search

This project implements a semantic image search application using CLIP (Contrastive Language-Image Pre-training) for feature extraction and FAISS (Facebook AI Similarity Search) for efficient indexing and retrieval. It allows users to search for images using either text queries or image examples.

## Features

*   **Semantic Search:** Searches images based on the semantic content of text or image queries, rather than just keyword matching.
*   **Text-Based Search:** Allows searching for images using a text description.
*   **Image-Based Search:** Allows searching for images similar to a query image.
*   **Efficient Indexing:** Uses FAISS for fast and scalable indexing of image features.
*   **Index Persistence:** Saves the FAISS index and image metadata to disk for reuse, avoiding the need to re-index every time the application starts (unless the image directory or its contents changes).
*   **Multithreaded Indexing:** Utilizes multi-threading to prevent the UI from freezing during feature extraction and indexing.
*  **Batch Feature Extraction**: Processes images in batches to improve the performance of the index.
*   **Tkinter GUI:** Provides a user-friendly graphical interface for easy interaction.
*   **Result Display:** Shows search results in a new browser tab with clickable thumbnails.
*   **Dynamic Image Directory Updates:** Automatically detects changes in the selected image directory and prompts for re-indexing.
*   **About Information**: Displays an information popup with information about the app and developers.
*   **Enter Key Binding:** Allows the search to be triggered with the enter key anywhere.
*   **Clear Progress Bar:** Clear progress bar when the indexing or update is finished.
*   **Improved Index Consistency:** Checks for consistency issues within the stored index (e.g. count of images vs. vectors) and prompts for reindexing.
*   **Icon Loading Handling:** Handles cases where the icon file (`SS.png`) might be missing without breaking the application.
*   **Robust Image Loading:** Uses PIL to verify if an image is valid before index it, and skips the ones which are corrupted or not readable.

## Installation

1.  **Clone the Repository:**

    ```bash
    git clone <your_repository_url>
    cd <your_repository_directory>
    ```
2.  **Create a virtual enviroment**

    ```bash
    python -m venv venv
    ```
3. **Activate enviroment**
    * Windows
      ```bash
       venv\Scripts\activate
      ```
   *  Linux/Mac
      ```bash
       source venv/bin/activate
      ```
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    **Note:**
     *  If you have a CUDA-enabled GPU, install `faiss-gpu` instead of `faiss-cpu` for faster indexing.
     *  Make sure you have the libraries: `clip`, `faiss-cpu`, `numpy`, `opencv_python_headless`,`Pillow`, `torch`.
       ```bash
      clip==1.0
      faiss_cpu==1.9.0.post1
      numpy==2.2.2
      opencv_python_headless==4.11.0.86
      Pillow==11.1.0
      torch==2.5.1
       ```
5.  **Place Icon:** Ensure that the `SS.png` icon file is in the same directory as the Python script.

## Usage

1.  **Run the Application:**

    ```bash
    python SemantikSearch.py
    ```

2.  **Select Image Directory:** Click "Examinar" in the "Directorio de Imágenes" section to choose a directory containing your images.

3.  **Wait for Indexing:** The application will index the images. A progress bar at the bottom will show the progress, also a message will be shown when the index is completed.

4.  **Perform a Search:**
    *   **Text Search:** Select "Buscar por Texto", enter your search query in the "Texto a Buscar" field, and specify the number of results to show ("Resultados a Mostrar (K)").
    *   **Image Search:** Select "Buscar por Imagen", enter the path to a query image in the "Ruta de Imagen" field, and specify the number of results to show ("Resultados a Mostrar (K)").

5.  **Click "Buscar":** This will perform the search and display results in a new browser tab.

6.  **Update Index:** If the image directory changes or images are added, press "Actualizar índice" to re-index the images.

7.  **Use Enter:** You can press enter anywhere in the application to trigger the search.

8. **About Information**: You can press "?" button to get about information.

## Project Structure

*   `SemantikSearch.py`: The main Python script containing the application logic and GUI.
*   `requirements.txt`: Lists the Python dependencies.
*   `SS.png`: Application icon.
*   `image_search_config.json`: Configuration file storing the last image directory used.
*   `image_index.bin`: File where the FAISS index is stored.

## Code Explanation

The core logic of the application resides in `SemantikSearch.py`. Here's a breakdown:

*   **Initialization (`__init__`):**
    *   Sets up the main window.
    *   Initializes logging, CUDA (if available), and the CLIP model.
    *   Loads configuration and sets default variables.
*   **GUI Creation (`_create_widgets`):**
    *   Builds the main GUI layout using Tkinter widgets.
*   **Configuration Management:**
    *   `load_config`, `save_config`: Load and save user configuration (e.g., last used image directory).
*   **Index Validation (`is_index_valid`):**
    *   Checks if the existing index file is valid for the current image directory by comparing metadata and timestamps.
*   **Index Loading and Creation (`load_or_create_index`, `load_index`):**
    *   `load_or_create_index`: Determines whether to load an existing index, re-index the directory or create a new one if not exists.
    *   `load_index`: Loads the FAISS index and related image metadata from disk.
*   **Indexing (`index_images`, `index_images_threaded`):**
    *   `index_images_threaded`: Creates a thread for the indexing process to prevent the UI from freezing.
    *   `index_images`:
        *   Locates all images in the selected directory.
        *   Extracts feature vectors in batches using CLIP.
        *   Normalizes vectors.
        *   Builds the FAISS index.
        *   Saves the index and image metadata.
*   **Feature Extraction (`extract_image_features`, `extract_text_features`):**
    *   Uses the CLIP model to generate feature vectors for images and text.
*   **Search (`search`, `search_by_text`, `search_by_image`):**
    *   Processes text or image queries.
    *   Performs a similarity search using the FAISS index.
    *   Displays the results in an HTML file in a new browser tab.
*  **Batch Feature Extraction**
    *   Images are extracted in batches to improve the index performance.
*  **Image Path Updates**
    *   Uses focus out to trigger and update the image path when is changed directly and not using the browser.
*  **<Enter> key binding**
    *   Added a binding to the root window to allow the user to use <Enter> to search anywhere in the UI.
*  **About Information**
    *   Added the functionality to show information about the app and the developers.

## Important Notes

*   **Image Preprocessing:** This project uses CLIP model preprocessing, please note that there may be some compatibility issues with some non-standard image formats.
*   **Faiss Index:** This project uses `faiss.IndexFlatIP` as the index, for higher scalability and faster search you can change it to `faiss.IndexHNSWFlat` or `faiss.IndexIVFFlat`.
*   **Error Handling:** The project has comprehensive error handling, logs to report any issues, and shows messages to inform the user when an error occurs.
*   **Numerical Stability:** Added a small constant (`1e-8`) during feature normalization to prevent numerical instability problems due to possible divisions by zero.
*  **Performance:**  The images are processed in batches to improve the overall indexing performance.

## Contributing

Feel free to submit pull requests for any bug fixes, improvements, or feature additions.

## License

This project is licensed under the MIT License.