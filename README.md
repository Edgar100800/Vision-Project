# 🏪 Sistema de Monitoreo Inteligente para Tiendas

Sistema completo de seguimiento e identificación de personas para cámaras de seguridad en tiendas, desarrollado con YOLO + Deep SORT + FastReID.

## 📋 Características

- **Detección de personas**: YOLOv8 para detección precisa
- **Seguimiento**: Deep SORT para tracking continuo
- **Re-identificación**: FastReID para identificación visual (Fase 3)
- **Análisis de zonas**: Detección de entrada/salida y análisis de colas
- **Dashboard en tiempo real**: Visualización de métricas y eventos
- **Logging completo**: Registro detallado de eventos y métricas

## 🏗️ Estado del Proyecto

### ✅ Completado
- **Fase 1**: Estructura del proyecto y configuración
- **Fase 2**: Pipeline básico (YOLO + Deep SORT)
  - Detección de personas con YOLOv8
  - Seguimiento con Deep SORT
  - Procesamiento de video completo
  - Sistema de logging
  - Configuración modular

### 🔄 En Progreso
- **Fase 3**: Integración de FastReID para re-identificación

### 📅 Pendiente
- **Fase 4**: Implementación de zonas de interés
- **Fase 5**: Dashboard y análisis en tiempo real
- **Fase 6**: Optimización y deployment

## 🚀 Instalación y Uso

### Configuración del Ambiente Virtual

```bash
# 1. Crear ambiente virtual
python3 -m venv venv

# 2. Activar ambiente virtual
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar instalación
python test_setup.py
```

### Uso Rápido

```bash
# Activar ambiente (script automático)
./activate_env.sh

# Procesar video
python src/main.py --video data/raw/video.mp4 --output data/processed/output_tracking.mp4

# Verificar resultados
python verify_results.py
```

## 📊 Uso

### Pipeline Básico

```bash
python src/main.py --video data/raw/video.mp4
```

### Dashboard

```bash
# Próximamente - Fase 5
streamlit run src/dashboard/app.py
```

## 📁 Estructura del Proyecto

```
Proye3/
├── data/
│   ├── raw/                    # Videos de entrada
│   ├── processed/              # Videos procesados
│   └── models/                 # Modelos entrenados
├── src/
│   ├── detectors/              # Detectores (YOLO)
│   ├── trackers/               # Trackers (Deep SORT)
│   ├── reid/                   # Re-identificación (FastReID)
│   ├── zones/                  # Análisis de zonas
│   ├── dashboard/              # Dashboard web
│   └── utils/                  # Utilidades
├── logs/                       # Archivos de log
├── requirements.txt            # Dependencias
├── activate_env.sh            # Script de activación
└── verify_results.py          # Verificación de resultados
```

## 🔧 Configuración

El sistema utiliza archivos de configuración modulares:

```python
# src/utils/config.py
config = get_config()
config.detection.confidence_threshold = 0.5
config.tracking.max_age = 30
```

## 📈 Resultados Actuales

### Video de Prueba Procesado
- **Duración**: 81 segundos
- **Frames**: 2,430 frames
- **Resolución**: 1920x1080
- **FPS de procesamiento**: ~11.6 FPS
- **Tamaño de salida**: 77.77 MB

### Métricas de Rendimiento
- ✅ Detección de personas funcional
- ✅ Seguimiento básico implementado
- ⚠️ Tracking con algunos errores (en optimización)
- 🔄 Re-identificación en desarrollo

## 🛠️ Comandos Útiles

```bash
# Activar ambiente
source venv/bin/activate

# Ejecutar con video personalizado
python src/main.py --video /path/to/video.mp4

# Ejecutar con visualización en tiempo real
python src/main.py --video data/raw/video.mp4 --display

# Verificar configuración
python test_setup.py

# Verificar resultados
python verify_results.py

# Ejecutar demo sin video
python demo.py
```

## 🎯 Próximos Pasos

1. **Corregir errores de tracking** en Deep SORT
2. **Implementar FastReID** para re-identificación
3. **Agregar zonas de interés** para análisis específico
4. **Crear dashboard interactivo** con Streamlit
5. **Optimizar rendimiento** para tiempo real

## 📚 Documentación

- [`DOCUMENTACION_FASES.md`](DOCUMENTACION_FASES.md) - Plan completo del proyecto
- [`PROJECT_STATUS.md`](PROJECT_STATUS.md) - Estado detallado por fases
- [`logs/`](logs/) - Archivos de log del sistema

## 🤝 Contribución

El proyecto sigue las mejores prácticas de desarrollo:

- **GitFlow** para manejo de ramas
- **Kanban** para seguimiento de tareas
- **Código limpio** con documentación
- **Testing** automatizado
- **Logging** estructurado

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

---

**Desarrollado con ❤️ para sistemas de monitoreo inteligente**