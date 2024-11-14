import tensorflow as tf
import tf2onnx

# Load the trained model
model_path = r'D:\Documents\guekece\TUGAS AKHIR\koding\segmentasi\unet_model.h5'

model = tf.keras.models.load_model(model_path)

# Define output path
onnx_model_path = 'D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/unet_model.onnx'

img_height = 480  # Ganti dengan tinggi gambar yang digunakan model Anda
img_width = 848   # Ganti dengan lebar gambar yang digunakan model Anda


# Convert the model
spec = (tf.TensorSpec((None, img_height, img_width, 3), tf.float32, name="input"),)
output_path = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_model_path)

print("Model berhasil dikonversi ke ONNX dan disimpan di", onnx_model_path)
