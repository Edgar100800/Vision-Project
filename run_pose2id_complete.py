#!/usr/bin/env python3
"""
Script de ejemplo para ejecutar el sistema completo POSE2ID
=========================================================

Este script demuestra cómo usar el sistema finalfinal.py con diferentes configuraciones
y proporciona ejemplos de uso práctico.
"""

from finalfinal import CompletePOSE2IDProcessor
import argparse
import os
import sys
from pathlib import Path

# Agregar el directorio del proyecto al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Ejecutar el sistema completo POSE2ID con tracking avanzado",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

1. Procesamiento básico:
   python run_pose2id_complete.py --video data/video.mp4

2. Con configuración personalizada:
   python run_pose2id_complete.py --video data/video.mp4 --output results/ --max-frames 500

3. Con checkpoints personalizados:
   python run_pose2id_complete.py --video data/video.mp4 --pose2id-ckpt /path/to/checkpoints

4. Procesamiento completo con todas las características:
   python run_pose2id_complete.py --video data/video.mp4 --enable-generation --enable-nfc
        """
    )

    # Argumentos obligatorios
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Ruta al archivo de video de entrada"
    )

    # Argumentos opcionales
    parser.add_argument(
        "--output",
        type=str,
        default="output/pose2id_complete",
        help="Directorio para archivos de salida (default: output/pose2id_complete)"
    )

    parser.add_argument(
        "--pose2id-ckpt",
        type=str,
        default="Pose2ID/IPG/pretrained",
        help="Ruta al directorio de checkpoints de POSE2ID (default: Pose2ID/IPG/pretrained)"
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Número máximo de frames a procesar (default: todos)"
    )

    # Configuración de modelos
    parser.add_argument(
        "--bbox-threshold",
        type=float,
        default=0.3,
        help="Umbral de confianza para detección de bounding boxes (default: 0.3)"
    )

    parser.add_argument(
        "--reid-threshold",
        type=float,
        default=0.4,
        help="Umbral de similitud para reidentificación (default: 0.4)"
    )

    parser.add_argument(
        "--max-gallery-size",
        type=int,
        default=50,
        help="Tamaño máximo de galería por persona (default: 50)"
    )

    # Características opcionales
    parser.add_argument(
        "--enable-generation",
        action="store_true",
        help="Habilitar generación de imágenes con IPG"
    )

    parser.add_argument(
        "--enable-nfc",
        action="store_true",
        help="Habilitar Neighbor Feature Centralization"
    )

    parser.add_argument(
        "--generation-interval",
        type=int,
        default=10,
        help="Intervalo de frames para generación de poses (default: 10)"
    )

    parser.add_argument(
        "--nfc-k1",
        type=int,
        default=3,
        help="Parámetro k1 para NFC (vecinos más cercanos) (default: 3)"
    )

    parser.add_argument(
        "--nfc-k2",
        type=int,
        default=2,
        help="Parámetro k2 para NFC (vecinos mutuos) (default: 2)"
    )

    # Configuración de visualización
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Guardar crops de personas detectadas"
    )

    parser.add_argument(
        "--save-features",
        action="store_true",
        help="Guardar características extraídas"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar información detallada durante el procesamiento"
    )

    args = parser.parse_args()

    # Validar argumentos
    if not os.path.exists(args.video):
        print(f"Error: El archivo de video {args.video} no existe")
        sys.exit(1)

    if not os.path.exists(args.pose2id_ckpt):
        print(
            f"Advertencia: El directorio de checkpoints {args.pose2id_ckpt} no existe")
        print("El sistema intentará funcionar sin algunos componentes de POSE2ID")

    # Crear directorio de salida
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configurar el procesador
    print("Inicializando sistema POSE2ID completo...")
    processor = CompletePOSE2IDProcessor(
        input_video=args.video,
        output_dir=args.output,
        pose2id_ckpt_dir=args.pose2id_ckpt
    )

    # Actualizar configuración con argumentos
    processor.config.update({
        'bbox_conf_threshold': args.bbox_threshold,
        'sim_threshold_assign_existing': args.reid_threshold,
        'max_gallery_size': args.max_gallery_size,
        'generate_poses_interval': args.generation_interval if args.enable_generation else float('inf'),
        'nfc_k1': args.nfc_k1,
        'nfc_k2': args.nfc_k2,
        'enable_generation': args.enable_generation,
        'enable_nfc': args.enable_nfc,
        'save_crops': args.save_crops,
        'save_features': args.save_features,
        'verbose': args.verbose
    })

    # Mostrar configuración
    print("\nConfiguración del sistema:")
    print(f"  Video de entrada: {args.video}")
    print(f"  Directorio de salida: {args.output}")
    print(f"  Checkpoints POSE2ID: {args.pose2id_ckpt}")
    print(f"  Frames máximos: {args.max_frames or 'Todos'}")
    print(f"  Umbral bbox: {args.bbox_threshold}")
    print(f"  Umbral ReID: {args.reid_threshold}")
    print(f"  Tamaño galería: {args.max_gallery_size}")
    print(
        f"  Generación IPG: {'Habilitada' if args.enable_generation else 'Deshabilitada'}")
    print(f"  NFC: {'Habilitado' if args.enable_nfc else 'Deshabilitado'}")

    if args.enable_generation:
        print(f"  Intervalo generación: {args.generation_interval} frames")

    if args.enable_nfc:
        print(f"  Parámetros NFC: k1={args.nfc_k1}, k2={args.nfc_k2}")

    print("\nIniciando procesamiento...")

    try:
        # Procesar video
        processor.process_video(max_frames=args.max_frames)

        print("\n" + "="*60)
        print("PROCESAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*60)

        # Mostrar archivos de salida
        output_files = list(output_dir.glob("*"))
        if output_files:
            print("\nArchivos generados:")
            for file in output_files:
                if file.is_file():
                    size_mb = file.stat().st_size / (1024*1024)
                    print(f"  {file.name}: {size_mb:.2f} MB")

        # Mostrar estadísticas finales
        print(f"\nEstadísticas finales:")
        print(f"  Total detecciones: {processor.stats['total_detections']}")
        print(
            f"  Extracciones POSE2ID: {processor.stats['pose2id_extractions']}")
        print(f"  Mejoras NFC: {processor.stats['nfc_enhancements']}")
        print(f"  Imágenes generadas: {processor.stats['image_generations']}")
        print(f"  Nuevas personas: {processor.stats['new_persons']}")
        print(
            f"  Reidentificaciones: {processor.stats['reidentified_persons']}")

        # Mostrar información de galerías
        print(f"\nGalerías de personas:")
        for person_id, gallery in processor.person_galleries.items():
            print(f"  {person_id}: {len(gallery)} características")

    except KeyboardInterrupt:
        print("\nProcesamiento interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\nError durante el procesamiento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
