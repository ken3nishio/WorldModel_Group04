# the session is need to restart the kernel.
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
!pip install -r requirements.txt
# transformers, diffusers, peft, accelerate を最新に更新
# Google colab uses this lines. but other env may not need to use this lines.
!pip install -U transformers diffusers peft accelerate

!python demo_gradio_f1.py --share
