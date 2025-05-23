

# VideoLLaMA3 - Установка на сервере
[Github VideoLLaMA3](https://github.com/DAMO-NLP-SG/VideoLLaMA3)
```
python -m venv venvname
source venvname/bin/activate
export CUDA_HOME=/deepstore/software/nvidia/cuda-12.4
pip install ninja
pip install transformers accelerate
pip install decord ffmpeg-python imageio opencv-python
pip install packaging wheel ninja
```  

`mkdir packages`  
Скачиваем torch и прочее:
`pip download -d packages torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124`  
Установка их на кластере с GPU:
`pip install --no-index --find-links packages torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0`  

## Если flash-attn не устанавливается, то попробовать так же как и torch, на кластере с GPU:
pip download -d packages flash-attn --no-build-isolation
pip install --no-index --find-links packages flash-attn --no-build-isolation

# Video-LLaVA - Установка на сервере
[Github Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA)
```

```