# import tensorflow as tf
# print(tf.config.list_physical_devices('GPU'))
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
