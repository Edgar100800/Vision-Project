#!/usr/bin/env python3
"""
Script para verificar la configuración de GPU y CUDA.
"""

import sys
import subprocess
import platform

def check_nvidia_smi():
    """Verificar si nvidia-smi está disponible."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ nvidia-smi disponible")
            print(result.stdout)
            return True
        else:
            print("✗ nvidia-smi no disponible")
            return False
    except FileNotFoundError:
        print("✗ nvidia-smi no encontrado")
        return False

def check_cuda_version():
    """Verificar versión de CUDA."""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ CUDA disponible")
            print(result.stdout.strip())
            return True
        else:
            print("✗ CUDA no disponible")
            return False
    except FileNotFoundError:
        print("✗ nvcc no encontrado")
        return False

def check_pytorch_cuda():
    """Verificar PyTorch con CUDA."""
    try:
        import torch
        print(f"✓ PyTorch versión: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA disponible en PyTorch")
            print(f"  - CUDA versión: {torch.version.cuda}")
            print(f"  - Número de GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  - GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Test básico de GPU
            try:
                x = torch.randn(1000, 1000).cuda()
                y = torch.randn(1000, 1000).cuda()
                z = torch.mm(x, y)
                print("✓ Test de operación GPU exitoso")
                return True
            except Exception as e:
                print(f"✗ Error en test de GPU: {e}")
                return False
        else:
            print("✗ CUDA no disponible en PyTorch")
            return False
    except ImportError:
        print("✗ PyTorch no instalado")
        return False

def check_ultralytics():
    """Verificar Ultralytics."""
    try:
        import ultralytics
        print(f"✓ Ultralytics versión: {ultralytics.__version__}")
        
        # Verificar si puede usar GPU
        from ultralytics import YOLO
        print("✓ Ultralytics importado correctamente")
        return True
    except ImportError:
        print("✗ Ultralytics no instalado")
        return False

def check_opencv():
    """Verificar OpenCV."""
    try:
        import cv2
        print(f"✓ OpenCV versión: {cv2.__version__}")
        
        # Verificar si OpenCV fue compilado con CUDA
        build_info = cv2.getBuildInformation()
        if "CUDA" in build_info:
            print("✓ OpenCV compilado con soporte CUDA")
        else:
            print("⚠ OpenCV sin soporte CUDA (normal)")
        
        return True
    except ImportError:
        print("✗ OpenCV no instalado")
        return False

def check_system_info():
    """Mostrar información del sistema."""
    print("=== Información del Sistema ===")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"Arquitectura: {platform.machine()}")
    print()

def main():
    """Función principal."""
    print("Verificación de GPU y CUDA para Person Re-ID System")
    print("=" * 60)
    print()
    
    check_system_info()
    
    print("=== Verificación de Hardware ===")
    nvidia_ok = check_nvidia_smi()
    cuda_ok = check_cuda_version()
    print()
    
    print("=== Verificación de Software ===")
    pytorch_ok = check_pytorch_cuda()
    ultralytics_ok = check_ultralytics()
    opencv_ok = check_opencv()
    print()
    
    print("=== Resumen ===")
    if nvidia_ok and cuda_ok and pytorch_ok and ultralytics_ok and opencv_ok:
        print("🎉 ¡Todo configurado correctamente! El sistema está listo para usar GPU.")
        print("\nPara probar el sistema completo:")
        print("python test_system.py")
    else:
        print("⚠ Algunos componentes no están configurados correctamente.")
        print("\nPasos para solucionar:")
        print("1. Ejecuta: install_cuda.bat")
        print("2. Reinicia la terminal")
        print("3. Ejecuta este script nuevamente")
    
    print()

if __name__ == "__main__":
    main() 