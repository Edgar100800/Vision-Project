@echo off
echo ========================================
echo Configurando Entorno Conda con CUDA
echo ========================================
echo.

echo Creando entorno conda...
conda create -n person_reid python=3.10 -y

echo.
echo Activando entorno...
call conda activate person_reid

echo.
echo Instalando PyTorch con CUDA...
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo.
echo Instalando dependencias adicionales...
pip install ultralytics opencv-python pillow
pip install timm transformers scikit-learn scipy
pip install ffmpeg-python moviepy
pip install pandas h5py matplotlib seaborn
pip install tqdm PyYAML

echo.
echo Instalando aceleración GPU...
pip install cupy-cuda12x

echo.
echo ========================================
echo Entorno configurado exitosamente!
echo ========================================
echo.
echo Para activar el entorno:
echo conda activate person_reid
echo.
echo Para verificar la instalación:
echo python check_gpu.py
echo.
pause 