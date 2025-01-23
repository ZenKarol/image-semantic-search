import tkinter as tk
from tkinter import ttk, filedialog, messagebox, PhotoImage
import os
import numpy as np
import faiss
import torch
import clip
from PIL import Image
import webbrowser
import tempfile
import logging
from typing import List, Optional
import threading
import json
import pickle
import io
import base64

CONFIG_FILE = "image_search_config.json"
INDEX_FILE = "image_index.bin"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class AboutWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Acerca de la Búsqueda Semántica de Imágenes")
        self.geometry("600x400")
        self.resizable(False, False)
        self.configure(bg="white")
        self.create_widgets()

    def create_widgets(self):
        # Textos a mostrar
        title = "Aplicación de Búsqueda Semántica de Imágenes"
        description = "Esta aplicación proporciona búsqueda semántica de imágenes utilizando CLIP (Contrastive Language–Image Pre-training) de OpenAi y FAISS de Meta, permitiendo a los usuarios encontrar imágenes relevantes basadas en el significado, no solo en palabras clave."
        developed_by_label = "Desarrollado por:"
        developer_details = "- Karol Jara ZenKarol@gmail.com"
        powered_by_label = "Impulsado por:"
        power_details = "- Google AI, Gemini"
        support_label = "Espero que este proyecto haya sido de gran utilidad para ti. Si es así, agradecería enormemente una pequeña donación. Tu apoyo no solo me ayuda a mantener este proyecto actualizado, sino que también me motiva a seguir creando más herramientas de código abierto para todos. Si deseas contribuir, puedes hacerlo a través de la siguiente dirección de Bitcoin:"
        btc_address = "bc1q57jvz0en5k8drduk4r9h0t5e9p86h25wsxzj2h"
        thank_you = "¡Gracias por tu apoyo!"
        version = "Versión: 1.1 (2025) Licencia MIT"

        # Lista de textos y sus tags
        text_items = [
            (title, "title", {"font": ("Arial", 12, "bold"), "justify": tk.CENTER}),
            ("\n\n", None, {}),
            (description, "description", {"font": ("Arial", 10), "justify": tk.CENTER}),
            ("\n\n", None, {}),
            (developed_by_label, "label", {"font": ("Arial", 10, "bold"), "justify": tk.CENTER}),
            ("\n", None, {}),
            (developer_details, "details", {"font": ("Arial", 10), "justify": tk.CENTER}),
            ("\n\n", None, {}),
            (powered_by_label, "label", {"font": ("Arial", 10, "bold"), "justify": tk.CENTER}),
            ("\n", None, {}),
            (power_details, "details", {"font": ("Arial", 10), "justify": tk.CENTER}),
            ("\n\n", None, {}),
            (support_label, "label", {"font": ("Arial", 10, "bold")}),
            ("\n\n", None, {}),
            (btc_address, "btc_address", {"font": ("Arial", 12), "foreground": "blue", "justify": tk.CENTER}),
            ("\n\n", None, {}),
            (thank_you, "thank_you", {"font": ("Arial", 10, "bold"), "justify": tk.CENTER}),
            ("\n\n", None, {}),
            (version, "version", {"font": ("Arial", 10), "justify": tk.CENTER}),
        ]

        # Creación del widget de texto
        self.text_widget = tk.Text(self, wrap=tk.WORD, bg="white", fg="black", padx=10, pady=10,
                                 borderwidth=0, highlightthickness=0)
        self.text_widget.pack(fill=tk.BOTH, expand=True)
        self.text_widget.config(state=tk.NORMAL)

        # Añadir texto al widget
        for text, tag, config in text_items:
            self.text_widget.insert(tk.END, text, tag)
            if tag and config:
                self.text_widget.tag_config(tag, **config)
            if tag == "btc_address":
                self.text_widget.tag_bind(tag, "<Button-1>", self.copy_btc_address)

        self.text_widget.tag_add("thank_you", "16.0", "16.end")
        self.text_widget.tag_add("version", "18.0", "18.end")
        self.text_widget.config(state=tk.DISABLED)

        # Botón de Cerrar
        close_button = ttk.Button(self, text="Cerrar", command=self.destroy)
        close_button.pack(pady=10)

    def copy_btc_address(self, event):
        self.clipboard_clear()
        self.clipboard_append("bc1q57jvz0en5k8drduk4r9h0t5e9p86h25wsxzj2h")
        self.update()


class ImageSearchWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Búsqueda Semántica de Imágenes")
        self.root.geometry("800x600")
        self.image_dir = None
        self.index = None
        self.image_paths = None
        self.query_type = tk.StringVar(value="text")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        logging.info("Modelo CLIP cargado correctamente.")
        self.feature_dim = 768
        self.config = self.load_config()
        self.index_metadata = {}
        self.k_value = tk.IntVar(value=5)
        self.batch_size = 64
        try:
            self.root.iconphoto(False, PhotoImage(file="SS.png"))
        except tk.TclError:
            logging.warning("Archivo de icono 'SS.png' no encontrado, omitiendo la carga del icono.")
        self._create_widgets()
        self._load_last_paths_and_check_index()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _create_widgets(self):
        """Crea los widgets de la interfaz"""
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.frame_image_dir = ttk.LabelFrame(self.main_frame, text="Directorio de Imágenes", padding=5)
        self.frame_image_dir.pack(fill=tk.X, padx=5, pady=5, expand=False)

        self.label_image_dir = ttk.Label(self.frame_image_dir, text="Ruta:", font=("Arial", "10"))
        self.label_image_dir.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.entry_image_dir = ttk.Entry(self.frame_image_dir, width=60)
        self.entry_image_dir.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.btn_browse_image_dir = ttk.Button(self.frame_image_dir, text="Examinar", command=self.browse_image_dir)
        self.btn_browse_image_dir.grid(row=0, column=2, padx=5, pady=5)
        self.frame_image_dir.columnconfigure(1, weight=1)

        self.frame_query = ttk.LabelFrame(self.main_frame, text="Búsqueda", padding=5)
        self.frame_query.pack(fill=tk.X, padx=5, pady=5, expand=False)

        self.radio_text = ttk.Radiobutton(self.frame_query, text="Buscar por Texto", variable=self.query_type,
                                        value="text")
        self.radio_text.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.radio_image = ttk.Radiobutton(self.frame_query, text="Buscar por Imagen", variable=self.query_type,
                                        value="image")
        self.radio_image.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.label_query_image = ttk.Label(self.frame_query, text="Ruta de Imagen:", font=("Arial", "10"))
        self.label_query_image.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.entry_query_image = ttk.Entry(self.frame_query, width=60)
        self.entry_query_image.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.btn_browse_query_image = ttk.Button(self.frame_query, text="Examinar", command=self.browse_query_image)
        self.btn_browse_query_image.grid(row=1, column=2, padx=5, pady=5)

        self.label_query_text = ttk.Label(self.frame_query, text="Texto a Buscar:", font=("Arial", "10"))
        self.label_query_text.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.entry_query_text = ttk.Entry(self.frame_query, width=60)
        self.entry_query_text.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        self.label_k_value = ttk.Label(self.frame_query, text="Resultados a Mostrar (K):", font=("Arial", "10"))
        self.label_k_value.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.entry_k_value = ttk.Spinbox(self.frame_query, from_=1, to=20, textvariable=self.k_value, width=5)
        self.entry_k_value.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        self.frame_query.columnconfigure(1, weight=1)

        self.frame_buttons = ttk.Frame(self.main_frame, padding=5)
        self.frame_buttons.pack(fill=tk.X, padx=5, pady=5, expand=False)

        self.btn_update_index = ttk.Button(self.frame_buttons, text="Actualizar Índice",width=20, command=self.update_index)
        self.btn_update_index.pack(side=tk.LEFT, padx=(165, 20), pady=5)

        self.btn_search = ttk.Button(self.frame_buttons, text="Buscar",width=20, command=self.search, style="Blue.TButton")
        self.btn_search.pack(side=tk.LEFT, padx=(20, 105), pady=5)
        self.btn_search.bind('<Return>', lambda event: self.search())
        self.root.bind('<Return>', lambda event: self.search())
        self.about_button = ttk.Button(self.frame_buttons, text="Acerca de", width=12, command=self.show_about)
        self.about_button.pack(side=tk.RIGHT, padx=5, pady=5)

        instructions_text = "Instrucciones de Uso:\n\n" \
                            "1. Selecciona un directorio de imágenes usando 'Examinar' en 'Directorio de Imágenes'.\n" \
                            "2. Espera a que se indexen las imágenes (Puede tomar unos minutos).\n" \
                            "3. Selecciona el tipo de búsqueda (texto o imagen).\n" \
                            "4. Ingresa la consulta con lenguaje natural o una imagen y la cantidad de resultados 'K'.\n" \
                            "5. Haz clic en 'Buscar' o presiona la tecla ENTER. Los resultados se mostrarán en una nueva pestaña del Navegador.\n\n" \
                            "Consejos:\n\n" \
                            "*. Prueba con sinónimos e incluso en inglés y otros idiomas\n" \
                            "*. En caso de que cambie el directorio de imágenes, usar botón para actualiza el índice."
        self.instructions_label = ttk.Label(self.main_frame, text=instructions_text, font=("Arial", "11"),
                                        foreground="gray", justify=tk.LEFT)
        self.instructions_label.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.status_label = ttk.Label(self.main_frame, text="", font=("Arial", "10"))
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.progress_bar = ttk.Progressbar(self.main_frame, orient="horizontal", length=200, mode="determinate")
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=0)
        self.main_frame.rowconfigure(1, weight=0)
        self.main_frame.rowconfigure(2, weight=0)

    def show_about(self):
        AboutWindow(self.root)

    def load_config(self):
        """Carga la configuración desde un archivo o crea una configuración por defecto."""
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"image_dir": ""}

    def save_config(self):
        """Guarda la configuración actual en un archivo."""
        config = {
            "image_dir": self.image_dir
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f)

    def _load_last_paths_and_check_index(self):
        """Carga los últimos valores usados desde la configuración y verifica el índice."""
        self.image_dir = self.config.get("image_dir", "")
        self.entry_image_dir.insert(0, self.image_dir)

        if self.image_dir:
            self.load_or_create_index()

    def on_close(self):
        """Guarda los valores de configuración antes de cerrar."""
        self.save_config()
        self.root.destroy()

    def browse_image_dir(self):
        """Abre un diálogo para seleccionar el directorio de imágenes."""
        new_image_dir = filedialog.askdirectory(title="Seleccionar Directorio de Imágenes")
        if new_image_dir:
            self.entry_image_dir.delete(0, tk.END)
            self.entry_image_dir.insert(0, new_image_dir)
            self.image_dir = new_image_dir
            self.load_or_create_index()
            self.save_config()

    def browse_query_image(self):
        """Abre un diálogo para seleccionar la imagen de consulta."""
        query_image_path = filedialog.askopenfilename(title="Seleccionar Imagen de Consulta",
                                                    filetypes=(("Image files", "*.png *.jpg *.jpeg"),
                                                                ("All files", "*.*")))
        if query_image_path:
            self.entry_query_image.delete(0, tk.END)
            self.entry_query_image.insert(0, query_image_path)

    def is_index_valid(self):
        """Verifica si el índice almacenado es válido para el directorio de imágenes actual."""
        try:
            with open(INDEX_FILE, "rb") as f:
                stored_data = pickle.load(f)
                stored_metadata = stored_data.get("metadata", {})
                stored_image_dir = stored_data.get("image_dir", "")

                if not stored_metadata or self.image_dir != stored_image_dir:
                    logging.info("El directorio de imágenes ha cambiado o no hay metadatos almacenados.")
                    return False

                image_paths = [os.path.join(self.image_dir, img) for img in os.listdir(self.image_dir) if
                                img.lower().endswith(('.png', '.jpg', '.jpeg'))]

                current_metadata = {}
                for path in image_paths:
                    try:
                        with Image.open(path) as img:
                            current_metadata[path] = os.path.getmtime(path)
                    except Exception as e:
                        logging.warning(f"No se pudo leer la imagen: {path}. Será ignorada. Error: {e}")
                        continue

                valid_image_count = len([p for p in image_paths if os.path.getmtime(p) in current_metadata.values()])
                if valid_image_count != len(stored_metadata):
                    logging.info("El número de imágenes legibles ha cambiado.")
                    return False

                if current_metadata != stored_metadata:
                    logging.info("Los metadatos de las imágenes han cambiado.")
                    return False

                logging.info("El índice es válido y está actualizado.")
                return True

        except (FileNotFoundError, pickle.UnpicklingError) as e:
            logging.warning(f"No se encontró el archivo del índice o está corrupto: {e}")
            return False

    def load_or_create_index(self):
        """Carga un índice existente o crea uno nuevo si es necesario."""
        current_image_dir = self.entry_image_dir.get()
        if not current_image_dir:
            return

        if self.image_dir != current_image_dir:
            self.image_dir = current_image_dir

        self.status_label.config(text="Verificando cambios del directorio...", foreground="gray")
        self.disable_search_button()

        if self.is_index_valid():
            self.load_index()
            self.status_label.config(text="Índice cargado desde archivo. ", foreground="green")
            self.enable_search_button()
        else:
            self.status_label.config(text="Índice desactualizado. Reindexando imágenes...", foreground="orange")
            self.index_images_threaded()

    def load_index(self):
        """Carga el índice FAISS almacenado en disco."""
        try:
            with open(INDEX_FILE, "rb") as f:
                stored_data = pickle.load(f)
                self.index = stored_data.get("index")
                self.image_paths = stored_data.get("image_paths")
                self.index_metadata = stored_data.get("metadata")

                if self.index is None or self.image_paths is None or self.index_metadata is None:
                    logging.error("Los datos del índice están incompletos o son inválidos. Reindexando...")
                    self.index = None
                    self.image_paths = None
                    self.index_metadata = {}
                    self.status_label.config(text="Datos del índice incompletos. Reindexando...", foreground="red")
                    self.index_images_threaded()
                    return

                if self.index.ntotal != len(self.image_paths):
                    logging.error(
                        f"Inconsistencia en el índice: Número de imágenes ({len(self.image_paths)}) no coincide con el número de vectores en el índice ({self.index.ntotal}). Reindexando...")
                    self.index = None
                    self.image_paths = None
                    self.index_metadata = {}
                    self.status_label.config(text="Inconsistencia en el índice detectada. Reindexando...",
                                            foreground="red")
                    self.index_images_threaded()
                    return

                logging.info(
                    f"Índice cargado correctamente. Imágenes: {len(self.image_paths)}, Vectores en el índice: {self.index.ntotal}")
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            logging.error(f"Error al cargar el archivo del índice: {e}")
            self.index = None
            self.image_paths = None
            self.index_metadata = {}

    def save_index(self):
        """Guarda el índice FAISS en disco junto con los metadatos de las imágenes."""
        try:
            with open(INDEX_FILE, "wb") as f:
                data = {
                    "index": self.index,
                    "image_paths": self.image_paths,
                    "metadata": self.index_metadata,
                    "image_dir": self.image_dir,
                }
                pickle.dump(data, f)
        except Exception as e:
            logging.error(f"Error al guardar el índice en el archivo: {e}")

    def index_images_threaded(self):
        """Inicia el proceso de indexación en un hilo separado."""
        self.status_label.config(text="Indexando imágenes, por favor espere...", foreground="red")
        self.disable_search_button()
        threading.Thread(target=self.index_images, daemon=True).start()

    def index_images(self):
        """Indexa las imágenes en el directorio seleccionado."""
        if not self.image_dir:
            messagebox.showerror("Error", "Por favor, seleccione un directorio de imágenes.")
            self.status_label.config(text="", foreground="gray")
            self.enable_search_button()
            return

        image_paths = [os.path.join(self.image_dir, img) for img in os.listdir(self.image_dir) if
                    img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        num_images = len(image_paths)

        if num_images == 0:
            logging.warning("No se encontraron imágenes en el directorio especificado.")
            messagebox.showerror("Error", "No se encontraron imágenes en el directorio especificado.")
            self.status_label.config(text="", foreground="gray")
            self.enable_search_button()
            return

        self.index = faiss.IndexFlatIP(self.feature_dim)
        features = []
        self.index_metadata = {}

        self.progress_bar["maximum"] = num_images
        self.progress_bar["value"] = 0

        for i in range(0, num_images, self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_features = self.extract_image_features_batch(batch_paths)

            if batch_features is not None:
                features.extend(batch_features)
                for path in batch_paths:
                    try:
                        self.index_metadata[path] = os.path.getmtime(path)
                    except FileNotFoundError:
                        logging.warning(f"Imagen no encontrada, no se indexará: {path}")
                        continue
            else:
                logging.warning(f"Error al procesar el lote de imágenes {batch_paths}. Saltando este lote.")

            self.progress_bar["value"] = i + len(batch_paths)
            self.progress_bar.update()

        self.progress_bar["value"] = num_images
        self.progress_bar.update()

        if not features:
            logging.error("No se pudieron extraer las características de ninguna imagen.")
            messagebox.showerror("Error", "No se pudieron extraer las características de ninguna imagen.")
            self.status_label.config(text="", foreground="gray")
            self.enable_search_button()
            return

        features = np.array(features, dtype=np.float32)
        features = self.normalize_vectors(features)
        self.index.add(features)
        self.image_paths = image_paths
        self.save_index()
        self.status_label.config(text="Imágenes indexadas correctamente. ", foreground="green")
        messagebox.showinfo("Información", "Imágenes indexadas correctamente. ")
        self.enable_search_button()
        self.progress_bar["value"] = 0
        self.progress_bar.update()
        logging.info(
            f"Indexación finalizada. Imágenes: {len(self.image_paths)}, Vectores en el índice: {self.index.ntotal}")

    def extract_image_features_batch(self, image_paths: List[str]) -> Optional[List[np.ndarray]]:
        """Extrae las características de un lote de imágenes utilizando el modelo CLIP."""
        images = []
        valid_paths = []
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    img = img.convert('RGB')
                    img_tensor = self.preprocess(img).unsqueeze(0)
                    images.append(img_tensor)
                    valid_paths.append(path)
            except Exception as e:
                logging.error(f"Error al abrir o procesar imagen: {path}. Error: {e}")
                continue

        if not images:
            logging.warning("No se pudieron cargar imágenes validas del lote.")
            return None

        try:
            batch_images = torch.cat(images).to(self.device)
            with torch.no_grad():
                batch_features = self.model.encode_image(batch_images).cpu().numpy()
                return [feature for feature in batch_features]

        except Exception as e:
            logging.error(f"Error al procesar el lote de imágenes: {e}")
            return None

    def extract_image_features(self, image_path: str) -> Optional[np.ndarray]:
        """Extrae las características de una imagen utilizando el modelo CLIP."""
        try:
            with Image.open(image_path) as img:
                image = img.convert('RGB')
                image = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model.encode_image(image)
                features = features.squeeze().cpu().numpy()

            return features

        except Exception as e:
            logging.error(f"Error al extraer características de la imagen {image_path}: {e}")
            return None

    def extract_text_features(self, text: str) -> Optional[np.ndarray]:
        """Extrae las características de un texto utilizando el modelo CLIP."""
        try:
            text_input = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                features = self.model.encode_text(text_input)
                features = features.squeeze().cpu().numpy()

            return features

        except Exception as e:
            logging.error(f"Error al extraer features de texto: {e}")
            return None

    def normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normaliza un array de vectores para que tenga longitud unitaria."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-8)

    def generate_html(self, query_feature: np.ndarray, k: int = 5, query_type: str = "image",
                        query_text: str = "") -> Optional[str]:
        """Genera un archivo HTML con los resultados de búsqueda y lo abre en un navegador."""
        if query_feature is None:
            logging.error(f"No se pudieron extraer las features de la consulta tipo {query_type}")
            return None

        query_feature = self.normalize_vectors(query_feature.reshape(1, -1))
        D, I = self.index.search(query_feature, k)

        if I.size == 0 or len(I[0]) == 0:
            logging.error("No se encontraron resultados para la búsqueda.")
            return None

        html_content = """
            <html>
            <head>
                <style>
                    .container { display: flex; flex-wrap: wrap; justify-content: flex-start; }
                    .image-item { margin: 10px; text-align: center; }
                    .image-item img { width: 150px; height: auto; border: 1px solid #ddd; }
                    .query-title { margin-bottom: 10px; }
                    .dist-indices { margin-top: 10px; font-size: 0.8em; color: #555;}
                    .image-text-container { display: inline-block; margin: 10px; text-align: left; width: 150px;}
                    .image-item a { text-decoration: none; }
                    .image-item a:hover { text-decoration: underline; }
                    body { font-family: sans-serif; margin: 20px; }
                </style>
            </head>
            <body>
        """

        if query_type == "image":
            html_content += f'<h1 class="query-title">Resultados de la Búsqueda por Imagen</h1>'
        elif query_type == "text":
            html_content += f'<h1 class="query-title">Resultados de la Búsqueda por Texto</h1>'
            html_content += f'<p class="query-title">Texto de la Consulta: {query_text}</p>'

        html_content += f'<p class="dist-indices">Distancias: {D.tolist()}</p>'
        html_content += f'<p class="dist-indices">Indices: {I.tolist()}</p>'
        html_content += '<div class="container">'
        for idx in I[0]:
            if 0 <= idx < len(self.image_paths):
                image_path = self.image_paths[idx]
                try:
                    with Image.open(image_path) as img:
                        if img.mode == 'RGBA':
                            img = img.convert('RGB')
                        img.thumbnail((150, 150))
                        buffered = io.BytesIO()
                        img.save(buffered, format="JPEG")
                        encoded_image = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"
                    html_content += f"""
                        <div class="image-item">
                            <a href="file:///{image_path}">
                                <img src="{encoded_image}" alt="Imagen" />
                            </a>
                            <br>
                        </div>
                    """
                except Exception as e:
                    logging.error(f"No se pudo crear la miniatura para la imagen {image_path} : {e}")
                    html_content += f"""
                        <div class="image-item">
                            <a href="file:///{image_path}">
                                <span>Imagen No Disponible</span>
                            </a>
                            <br>
                        </div>
                    """
            else:
                logging.warning(f"Index {idx} fuera de rango. Número de imagenes: {len(self.image_paths)}")
        html_content += '</div></body></html>'
        return html_content

    def search_and_display(self, query_feature: np.ndarray, k: int = 5, query_type: str = "image",
                        query_text: str = ""):
        """Genera un archivo HTML con los resultados de búsqueda y lo abre en un navegador."""
        html_content = self.generate_html(query_feature, k, query_type, query_text)

        if html_content:
            with tempfile.NamedTemporaryFile(mode='w', suffix=".html", delete=False) as f:
                temp_filename = f.name
                f.write(html_content)
            webbrowser.open_new_tab(f"file:///{temp_filename}")

    def search_by_image(self, query_image_path: str, k: int = 5):
        """Realiza una búsqueda de imágenes basada en una imagen de consulta."""
        query_feature = self.extract_image_features(query_image_path)
        self.search_and_display(query_feature, k, query_type="image")

    def search_by_text(self, query_text: str, k: int = 3):
        """Realiza una búsqueda de imágenes basada en una consulta de texto."""
        query_feature = self.extract_text_features(query_text)
        self.search_and_display(query_feature, k, query_type="text", query_text=query_text)

    def search(self):
        """Realiza una búsqueda basada en la consulta proporcionada (texto o imagen)."""
        if not self.index:
            messagebox.showerror("Error", "Por favor, seleccione un directorio de imágenes e indexe las imágenes.")
            return

        k = self.k_value.get()
        if self.query_type.get() == "image":
            query_image_path = self.entry_query_image.get()
            if query_image_path:
                self.search_by_image(query_image_path, k)
            else:
                messagebox.showerror("Error", "Por favor, seleccione una imagen de consulta.")
        elif self.query_type.get() == "text":
            query_text = self.entry_query_text.get()
            if query_text:
                self.search_by_text(query_text, k)
            else:
                messagebox.showerror("Error", "Por favor, ingrese un texto de consulta.")

    def disable_search_button(self):
        """Desactiva el botón de búsqueda."""
        self.btn_search.config(state="disabled")
        self.btn_update_index.config(state="disabled")

    def enable_search_button(self):
        """Activa el botón de búsqueda."""
        self.btn_search.config(state="normal")
        self.btn_update_index.config(state="normal")

    def update_index(self):
        """Actualiza el índice manualmente."""
        current_image_dir = self.entry_image_dir.get()
        if not current_image_dir:
            messagebox.showerror("Error", "Por favor, seleccione un directorio de imágenes.")
            return

        if self.image_dir != current_image_dir:
            self.image_dir = current_image_dir

        self.status_label.config(text="Verificando cambios en el directorio...", foreground="gray")
        self.disable_search_button()

        if self.is_index_valid():
            self.load_index()
            self.status_label.config(text="Índice está actualizado. ", foreground="green")
            messagebox.showinfo("Información", "El índice ya está actualizado. ")
            self.enable_search_button()
        else:
            self.index_images_threaded()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSearchWindow(root)
    root.mainloop()