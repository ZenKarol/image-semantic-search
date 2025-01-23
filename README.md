# Búsqueda Semántica de Imágenes

Este proyecto implementa una aplicación de búsqueda semántica de imágenes utilizando CLIP (Contrastive Language-Image Pre-training) para la extracción de características y FAISS (Facebook AI Similarity Search) para la indexación y recuperación eficientes. Permite a los usuarios buscar imágenes utilizando consultas de texto o ejemplos de imágenes.

## Características

*   **Búsqueda Semántica:** Busca imágenes basándose en el contenido semántico de las consultas de texto o imágenes, en lugar de solo la coincidencia de palabras clave.
*   **Búsqueda Basada en Texto:** Permite buscar imágenes utilizando una descripción de texto.
*   **Búsqueda Basada en Imágenes:** Permite buscar imágenes similares a una imagen de consulta.
*   **Indexación Eficiente:** Utiliza FAISS para una indexación rápida y escalable de las características de las imágenes.
*   **Persistencia del Índice:** Guarda el índice FAISS y los metadatos de las imágenes en el disco para su reutilización, evitando la necesidad de reindexar cada vez que se inicia la aplicación (a menos que cambie el directorio de imágenes o su contenido).
*   **Indexación Multihilo:** Utiliza subprocesos múltiples para evitar que la interfaz de usuario se congele durante la extracción de características y la indexación.
*   **Extracción de Características por Lotes:** Procesa las imágenes en lotes para mejorar el rendimiento del índice.
*   **GUI Tkinter:** Proporciona una interfaz gráfica de usuario amigable para una fácil interacción.
*   **Visualización de Resultados:** Muestra los resultados de búsqueda en una nueva pestaña del navegador con miniaturas en las que se puede hacer clic.
*   **Actualizaciones Dinámicas del Directorio de Imágenes:** Detecta automáticamente los cambios en el directorio de imágenes seleccionado y solicita la reindexación.
*   **Información "Acerca de":** Muestra una ventana emergente con información sobre la aplicación y los desarrolladores.
*   **Vinculación de la Tecla Enter:** Permite que la búsqueda se active con la tecla Enter en cualquier lugar.
*   **Barra de Progreso Clara:** Borra la barra de progreso cuando finaliza la indexación o la actualización.
*   **Consistencia del Índice Mejorada:** Verifica si hay problemas de consistencia dentro del índice almacenado (por ejemplo, el recuento de imágenes frente a los vectores) y solicita la reindexación.
*   **Manejo de la Carga de Iconos:** Maneja los casos en los que el archivo de icono (`SS.png`) podría faltar sin que la aplicación se interrumpa.
*   **Carga Robusta de Imágenes:** Utiliza PIL para verificar si una imagen es válida antes de indexarla y omite las que están dañadas o no se pueden leer.

## Instalación

1.  **Clona el Repositorio:**

    ```bash
    git clone <tu_url_del_repositorio>
    cd <tu_directorio_del_repositorio>
    ```
2.  **Crea un entorno virtual**

    ```bash
    python -m venv venv
    ```
3.  **Activa el entorno virtual**
    *   Windows
        ```bash
        venv\Scripts\activate
        ```
    *   Linux/Mac
        ```bash
        source venv/bin/activate
        ```
4.  **Instala las Dependencias:**

    ```bash
    pip install -r requirements.txt
    ```

    **Nota:**
    *   Si tienes una GPU habilitada para CUDA, instala `faiss-gpu` en lugar de `faiss-cpu` para una indexación más rápida.
    *   Asegúrate de tener las bibliotecas: `clip`, `faiss-cpu`, `numpy`, `opencv_python_headless`, `Pillow`, `torch`.
        ```bash
        clip==1.0
        faiss_cpu==1.9.0.post1
        numpy==2.2.2
        opencv_python_headless==4.11.0.86
        Pillow==11.1.0
        torch==2.5.1
        ```
5.  **Coloca el Icono:** Asegúrate de que el archivo de icono `SS.png` esté en el mismo directorio que el script de Python.

## Uso

1.  **Ejecuta la Aplicación:**

    ```bash
    python ImageSemanticSearchEs.py
    ```

2.  **Selecciona el Directorio de Imágenes:** Haz clic en "Examinar" en la sección "Directorio de Imágenes" para elegir un directorio que contenga tus imágenes.

3.  **Espera la Indexación:** La aplicación indexará las imágenes. Una barra de progreso en la parte inferior mostrará el progreso, también se mostrará un mensaje cuando se complete el índice.

4.  **Realiza una Búsqueda:**
    *   **Búsqueda por Texto:** Selecciona "Buscar por Texto", introduce tu consulta de búsqueda en el campo "Texto a Buscar" y especifica el número de resultados a mostrar ("Resultados a Mostrar (K)").
    *   **Búsqueda por Imagen:** Selecciona "Buscar por Imagen", introduce la ruta a una imagen de consulta en el campo "Ruta de Imagen" y especifica el número de resultados a mostrar ("Resultados a Mostrar (K)").

5.  **Haz clic en "Buscar":** Esto realizará la búsqueda y mostrará los resultados en una nueva pestaña del navegador.

6.  **Actualiza el Índice:** Si el directorio de imágenes cambia o se añaden imágenes, presiona "Actualizar índice" para reindexar las imágenes.

7.  **Usa Enter:** Puedes presionar Enter en cualquier lugar de la aplicación para activar la búsqueda.

8.  **Información "Acerca de":** Puedes presionar el botón "?" para obtener información sobre la aplicación.

## Estructura del Proyecto

*   `ImageSemanticSearchEs.py`: El script principal de Python que contiene la lógica de la aplicación y la GUI.
*   `requirements.txt`: Lista las dependencias de Python.
*   `SS.png`: Icono de la aplicación.
*   `image_search_config.json`: Archivo de configuración que almacena el último directorio de imágenes utilizado.
*   `image_index.bin`: Archivo donde se almacena el índice FAISS.

## Explicación del Código

La lógica principal de la aplicación reside en `ImageSemanticSearchEs.py`. Aquí tienes un desglose:

*   **Inicialización (`__init__`):**
    *   Configura la ventana principal.
    *   Inicializa el registro, CUDA (si está disponible) y el modelo CLIP.
    *   Carga la configuración y establece las variables predeterminadas.
*   **Creación de la GUI (`_create_widgets`):**
    *   Construye el diseño principal de la GUI utilizando widgets de Tkinter.
*   **Gestión de la Configuración:**
    *   `load_config`, `save_config`: Carga y guarda la configuración del usuario (por ejemplo, el último directorio de imágenes utilizado).
*   **Validación del Índice (`is_index_valid`):**
    *   Verifica si el archivo de índice existente es válido para el directorio de imágenes actual comparando los metadatos y las marcas de tiempo.
*   **Carga y Creación del Índice (`load_or_create_index`, `load_index`):**
    *   `load_or_create_index`: Determina si cargar un índice existente, reindexar el directorio o crear uno nuevo si no existe.
    *   `load_index`: Carga el índice FAISS y los metadatos de imágenes relacionados desde el disco.
*   **Indexación (`index_images`, `index_images_threaded`):**
    *   `index_images_threaded`: Crea un hilo para el proceso de indexación para evitar que la interfaz de usuario se congele.
    *   `index_images`:
        *   Localiza todas las imágenes en el directorio seleccionado.
        *   Extrae vectores de características en lotes utilizando CLIP.
        *   Normaliza los vectores.
        *   Construye el índice FAISS.
        *   Guarda el índice y los metadatos de las imágenes.
*   **Extracción de Características (`extract_image_features`, `extract_text_features`):**
    *   Utiliza el modelo CLIP para generar vectores de características para imágenes y texto.
*   **Búsqueda (`search`, `search_by_text`, `search_by_image`):**
    *   Procesa consultas de texto o imágenes.
    *   Realiza una búsqueda de similitud utilizando el índice FAISS.
    *   Muestra los resultados en un archivo HTML en una nueva pestaña del navegador.
*   **Extracción de Características por Lotes**
    *   Las imágenes se extraen en lotes para mejorar el rendimiento general de la indexación.
*   **Actualizaciones de la Ruta de la Imagen**
    *   Utiliza el enfoque de salida para activar y actualizar la ruta de la imagen cuando se cambia directamente y no utilizando el navegador.
*   **Vinculación de la Tecla <Enter>**
    *   Se ha añadido una vinculación a la ventana raíz para permitir al usuario utilizar <Enter> para buscar en cualquier lugar de la interfaz de usuario.
*   **Información "Acerca de"**
    *   Se ha añadido la funcionalidad para mostrar información sobre la aplicación y los desarrolladores.

## Notas Importantes

*   **Preprocesamiento de Imágenes:** Este proyecto utiliza el preprocesamiento del modelo CLIP, ten en cuenta que puede haber algunos problemas de compatibilidad con algunos formatos de imagen no estándar.
*   **Índice Faiss:** Este proyecto utiliza `faiss.IndexFlatIP` como índice, para una mayor escalabilidad y una búsqueda más rápida puedes cambiarlo a `faiss.IndexHNSWFlat` o `faiss.IndexIVFFlat`.
*   **Manejo de Errores:** El proyecto tiene un manejo de errores integral, registros para informar de cualquier problema y muestra mensajes para informar al usuario cuando se produce un error.
*   **Estabilidad Numérica:** Se ha añadido una pequeña constante (`1e-8`) durante la normalización de características para evitar problemas de inestabilidad numérica debido a posibles divisiones por cero.
*   **Rendimiento:** Las imágenes se procesan en lotes para mejorar el rendimiento general de la indexación.

## Contribución

Siéntete libre de enviar solicitudes de extracción para cualquier corrección de errores, mejoras o adiciones de características.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT.