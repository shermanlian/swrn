# Copyright 2021 by Andrey Ignatov. All Rights Reserved.

# The following instructions will show you how to test your converted (quantized / floating-point) TFLite model
# on the real images

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import numpy as np
import os
import cv2

from tensorflow.lite.python import interpreter as interpreter_wrapper
from tqdm import tqdm
import time

if __name__ == "__main__":

    # Specify the name of your TFLite model and the location of the sample test images

    model_file = "tflite/model.tflite"

    # Load your TFLite model
    interpreter = interpreter_wrapper.Interpreter(model_path=model_file, num_threads=32)

    input_details = interpreter.get_input_details()
    print(input_details)

    output_details = interpreter.get_output_details()
    print(output_details)

