#!/usr/bin/env python3
"""
Ejemplo de uso del modelo finalversion.py
Muestra diferentes formas de ejecutar el modelo Pose2ID
"""

import subprocess
import sys
import os
from pathlib import Path


def ejemplo_basico():
    """Ejemplo básico de uso del modelo"""
    print("🚀 Ejemplo 1: Uso básico del modelo")
    print("Comando: python run_finalversion.py --video data/raw/video.mp4 --output output/ejemplo1")
    print()

    # Ejecutar el comando
    result = subprocess.run([
        "python", "run_finalversion.py",
        "--video", "data/raw/video.mp4",
        "--output", "output/ejemplo1"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Procesamiento completado exitosamente")
        print(f"📁 Resultados guardados en: output/ejemplo1/")
    else:
        print("❌ Error en el procesamiento:")
        print(result.stderr)


def ejemplo_con_limite_frames():
    """Ejemplo con límite de frames"""
    print("\n🚀 Ejemplo 2: Procesamiento con límite de frames")
    print("Comando: python run_finalversion.py --video data/raw/video.mp4 --output output/ejemplo2 --max-frames 100")
    print()

    # Ejecutar el comando
    result = subprocess.run([
        "python", "run_finalversion.py",
        "--video", "data/raw/video.mp4",
        "--output", "output/ejemplo2",
        "--max-frames", "100"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Procesamiento de 100 frames completado")
        print(f"📁 Resultados guardados en: output/ejemplo2/")
    else:
        print("❌ Error en el procesamiento:")
        print(result.stderr)


def ejemplo_video_diferente():
    """Ejemplo con video diferente"""
    print("\n🚀 Ejemplo 3: Procesamiento de video alternativo")
    print("Comando: python run_finalversion.py --video data/raw/video2.mp4 --output output/ejemplo3")
    print()

    # Ejecutar el comando
    result = subprocess.run([
        "python", "run_finalversion.py",
        "--video", "data/raw/video2.mp4",
        "--output", "output/ejemplo3"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Procesamiento de video2.mp4 completado")
        print(f"📁 Resultados guardados en: output/ejemplo3/")
    else:
        print("❌ Error en el procesamiento:")
        print(result.stderr)


def mostrar_resultados(output_dir):
    """Mostrar los resultados del procesamiento"""
    output_path = Path(output_dir)

    if not output_path.exists():
        print(f"❌ Directorio {output_dir} no existe")
        return

    print(f"\n📊 Resultados en {output_dir}:")

    # Mostrar archivos generados
    files = list(output_path.glob("*"))
    for file in files:
        if file.is_file():
            size = file.stat().st_size
            size_mb = size / (1024 * 1024)
            print(f"  📄 {file.name}: {size_mb:.2f} MB")

    # Leer y mostrar el resumen JSON si existe
    summary_file = output_path / "pose2id_summary.json"
    if summary_file.exists():
        import json
        with open(summary_file, 'r') as f:
            summary = json.load(f)

        print(f"\n📈 Estadísticas del procesamiento:")
        stats = summary['stats']
        print(f"  🎯 Detecciones totales: {stats['total_detections']}")
        print(f"  👥 Personas creadas: {stats['persons_created']}")
        print(f"  🔄 Centralizaciones NFC: {stats['nfc_centralizations']}")
        print(f"  📊 Computaciones ID2: {stats['id2_computations']}")
        print(f"  📉 Tracks perdidos: {stats['tracks_lost']}")
        print(f"  🔁 Tracks reactivados: {stats['tracks_reactivated']}")

        print(f"\n👤 Detalles de clusters:")
        for person_id, cluster in summary['clusters'].items():
            print(
                f"  {person_id}: {cluster['size']} features, ID2: {cluster['id2_density']:.3f}")


def ejecutar_test():
    """Ejecutar la suite de tests"""
    print("\n🧪 Ejecutando suite de tests...")
    print("Comando: python run_finalversion.py --test")
    print()

    result = subprocess.run([
        "python", "run_finalversion.py", "--test"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Todos los tests pasaron exitosamente")
    else:
        print("❌ Algunos tests fallaron:")
        print(result.stderr)


def main():
    """Función principal con ejemplos"""
    print("=" * 60)
    print("🎥 EJEMPLOS DE USO DEL MODELO POSE2ID")
    print("=" * 60)

    # Verificar que estamos en el directorio correcto
    if not Path("finalversion.py").exists():
        print("❌ Error: No se encuentra finalversion.py")
        print("   Asegúrate de ejecutar este script desde el directorio del proyecto")
        return

    # Verificar que el video existe
    if not Path("data/raw/video.mp4").exists():
        print("❌ Error: No se encuentra data/raw/video.mp4")
        print("   Asegúrate de que el video esté en la ubicación correcta")
        return

    try:
        # Ejecutar test primero
        ejecutar_test()

        # Ejemplo básico
        ejemplo_basico()
        mostrar_resultados("output/ejemplo1")

        # Ejemplo con límite de frames
        ejemplo_con_limite_frames()
        mostrar_resultados("output/ejemplo2")

        # Ejemplo con video diferente (si existe)
        if Path("data/raw/video2.mp4").exists():
            ejemplo_video_diferente()
            mostrar_resultados("output/ejemplo3")

        print("\n" + "=" * 60)
        print("✅ TODOS LOS EJEMPLOS COMPLETADOS EXITOSAMENTE")
        print("=" * 60)

        print("\n📚 Comandos útiles:")
        print("  • Test rápido: python run_finalversion.py --test")
        print("  • Procesar video: python run_finalversion.py --video data/raw/video.mp4 --output output/mi_test")
        print("  • Con límite: python run_finalversion.py --video data/raw/video.mp4 --output output/mi_test --max-frames 100")
        print("  • Ver ayuda: python run_finalversion.py --help-finalversion")

    except KeyboardInterrupt:
        print("\n⚠️  Ejecución interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")


if __name__ == "__main__":
    main()
