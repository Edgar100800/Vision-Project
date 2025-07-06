| Fase | Objetivo Principal | Entregables Clave |
|------|-------------------|-------------------|
| **0. Kick-off** | Definir alcance, métricas de éxito y hardware disponible. | Documento de requisitos, lista de KPIs, checklist de hardware. |
| **1. Configuración Inicial** | Levantar entorno y repositorio base. | Entorno `conda/venv`, estructura de carpetas, script de test de GPU. |
| **2. Pipeline Básico** | Detección (YOLO) + Tracking (Deep SORT). | Script que dibuja bounding boxes e IDs en tiempo real desde una cámara o video. |
| **3. Integración Re-ID** | Añadir embeddings visuales para identidades persistentes. | Módulo Re-ID (FastReID) integrado en el tracker, demo con IDs estables tras oclusiones. |
| **4. Zonas de Interés & Eventos** | Medir entrada/salida, cola, atención. | Lógica de cruces de líneas/ROIs, logs CSV/JSON de eventos. |
| **5. Analítica & Dashboard** | Visualizar métricas y reproducir videos anotados. | Dashboard (Streamlit/Dash), reportes PDF/CSV. |
| **6. Optimización & Despliegue** | Acelerar inferencia y empaquetar. | Quantization / TensorRT, contenedor Docker, manual de despliegue. |

> **Nota:** Las duraciones son estimadas y asumen 1-2 desarrolladores dedicados.

| Re-ID | **FastReID** | Embeddings de apariencia para IDs persistentes. |

# Dependencias principales
pip install ultralytics opencv-python deep_sort_realtime torch torchvision torchaudio
# Instalar FastReID (fuente)
git clone https://github.com/JDAI-CV/fast-reid.git
cd fast-reid && pip install -r requirements.txt && python setup.py develop
cd ..
# Opcionales
pip install streamlit dash pandas seaborn ffmpeg-python

### Fase 2 – Pipeline Básico

1. Cargar `/data/video.mp4` con OpenCV.

## 10. Organización y Contribución

### 10.1 Flujo de trabajo Git

* Modelo **GitFlow** simplificado:
  * `main`: branch estable en producción.
  * `develop`: última versión funcional.
  * `feature/<nombre>`: nuevas características.
  * `bugfix/<nombre>`: corrección de errores.
* Pull requests hacia `develop` con revisión obligatoria.

### 10.2 Gestión de tareas

* Se utilizará **tablero Kanban** en GitHub Projects.
* Issues etiquetados por tipo: `feat`, `bug`, `doc`, `test`, `opt`.
* Cada fase (del roadmap) se mapea a un _milestone_.

### 10.3 Convenciones de código

* **Black** + **isort** para formato.
* Hooks de **pre-commit** configurados para linting automático.
* Identificadores en inglés; comentarios en español donde aporte claridad.
