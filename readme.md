# VideoLLaMA3 - Установка на сервере
[VideoLLaMA3](https://github.com/DAMO-NLP-SG/VideoLLaMA3)
```
python -m venv venvname
source venvname/bin/activate
export CUDA_HOME=/deepstore/software/nvidia/cuda-12.4
pip install ninja
pip install transformers accelerate
pip install decord ffmpeg-python imageio opencv-python
pip install packaging wheel ninja
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn --no-build-isolation
```  

Если нет интернета при использовании GPU на кластере, то необходимо сначала скачать модуль в конкретную папку, а потом его установить.

Скачивание в папку:
```
mkdir packages
pip download -d packages name_module
```


Установка из папки:  
```
pip install --no-index --find-links packages name_module
```