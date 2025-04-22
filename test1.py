#
#   Test Python script
#
print("Hello World!")

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

tf.sysconfig.get_build_info()