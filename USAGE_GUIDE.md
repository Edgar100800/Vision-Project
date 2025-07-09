# Guía de Uso - Modelo Final Pose2ID

## Descripción
Este modelo integra la metodología Pose2ID con nuestro sistema de tracking para re-identificación avanzada de personas.

## Configuración del Entorno Virtual

### 1. Verificar que el entorno virtual esté configurado
```bash
# El entorno virtual ya está configurado en la carpeta venv/
ls -la venv/
```

### 2. Activar el entorno virtual
```bash
source venv/bin/activate
```

### 3. Verificar que las dependencias estén instaladas
```bash
pip list | grep -E "(torch|ultralytics|torchreid|opencv-python)"
```

## Uso del Modelo

### Opción 1: Usar el script runner (Recomendado)
```bash
# Ejecutar test completo
python run_finalversion.py --test

# Ejecutar con video específico
python run_finalversion.py --video data/raw/video.mp4 --output output/mi_test

# Ejecutar con límite de frames
python run_finalversion.py --video data/raw/video.mp4 --output output/mi_test --max-frames 100

# Ver ayuda
python run_finalversion.py --help-finalversion
```

### Opción 2: Uso directo (con entorno virtual activado)
```bash
# Activar entorno virtual primero
source venv/bin/activate

# Ejecutar el modelo
python finalversion.py --video data/raw/video.mp4 --output output/mi_test

# Con límite de frames
python finalversion.py --video data/raw/video.mp4 --output output/mi_test --max-frames 50
```

## Archivos de Video Disponibles
- `data/raw/video.mp4` - Video principal
- `data/raw/video0.mp4` - Video alternativo
- `data/raw/video2.mp4` - Video alternativo
- `data/raw/video3.mp4` - Video alternativo

## Resultados
Los resultados se guardan en el directorio especificado con:
- `output_video.mp4` - Video procesado con anotaciones
- `frame_results.json` - Datos de detección por frame
- `pose2id_summary.json` - Resumen de métricas Pose2ID
- `pipeline.log` - Log del procesamiento
- `crops/` - Recortes de personas detectadas

## Características del Modelo

### Pose2ID Integration
- **NFC (Neighbor Feature Centralization)**: Mejora las características usando relaciones de vecindad
- **ID2 Metric**: Medición cuantitativa de densidad de identidad
- **Training-free**: No requiere entrenamiento adicional

### Configuración Actual
```python
{
    'bbox_conf_threshold': 0.5,
    'person_conf_threshold': 0.85,
    'nfc_k1': 2,  # Parámetro NFC
    'nfc_k2': 2,  # Parámetro NFC
    'similarity_threshold': 0.6,
    'max_frames_lost': 15,
    'feature_buffer_size': 50,
    'id2_update_interval': 10,
    'min_features_for_nfc': 3,
}
```

## Estadísticas de Rendimiento
En el test con 495 frames (video completo):
- ✅ **6 personas creadas**
- ✅ **528 centralizaciones NFC**
- ✅ **729 detecciones totales**
- ✅ **49 computaciones ID2**
- ✅ **8 tracks perdidos**
- ✅ **Procesamiento: ~40ms por frame**

### Métricas ID2 por Persona:
- **Person_1**: 50 features, ID2: 0.533
- **Person_2**: 50 features, ID2: 0.215
- **Person_3**: 50 features, ID2: 0.566

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'tracking'"
```bash
# Asegúrate de usar el script runner
python run_finalversion.py --video data/raw/video.mp4 --output output/test

# O activar el entorno virtual y agregar src al path
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:src"
python finalversion.py --video data/raw/video.mp4 --output output/test
```

### Error: "CUDA not available"
```bash
# El modelo funciona en CPU, esto es normal
# Se mostrará: "Using device: cpu"
```

### Error: "selected index k out of range"
```bash
# Esto se corrigió automáticamente en la versión actual
# El modelo ajusta k1 y k2 según el número de características disponibles
```

## Comandos Útiles

### Ejecutar test rápido
```bash
python run_finalversion.py --test
```

### Procesar video completo
```bash
python run_finalversion.py --video data/raw/video.mp4 --output output/full_video
```

### Procesar solo primeros 200 frames
```bash
python run_finalversion.py --video data/raw/video.mp4 --output output/quick_test --max-frames 200
```

### Ver logs en tiempo real
```bash
python run_finalversion.py --video data/raw/video.mp4 --output output/test | tee processing.log
```

## Notas Importantes
1. **Entorno Virtual**: Siempre usar el entorno virtual para evitar conflictos de dependencias
2. **Memoria**: El modelo usa CPU por defecto, funciona bien en sistemas sin GPU
3. **Rendimiento**: ~35ms por frame en CPU, escalable según hardware
4. **Compatibilidad**: Funciona con Python 3.13 y todas las dependencias están instaladas

## Contacto
Para problemas o mejoras, consultar la documentación del proyecto o contactar al equipo de desarrollo. 