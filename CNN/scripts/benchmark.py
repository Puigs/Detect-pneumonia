#!/usr/bin/env python 

import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import glob
import sys

if len(sys.argv) < 3:
    print("This script take in argument two inputs")
    print("A directory witch contains model onnx")
    print("A directory which contains images")
    exit()

if sys.argv[1][-1:] == "/":
    sys.argv[1] += "*"
else:
    sys.argv[1] += "/*"

if sys.argv[2][-1:] == "/":
    sys.argv[2] += "*"
else:
    sys.argv[2] += "/*"

result = dict()
list_model_path = glob.glob(sys.argv[1])
for model in list_model_path:
    result[model.split("/")[-1]] = [0, 0]

list_image_path = glob.glob(sys.argv[2])

for image_path in list_image_path:
    img = np.array(Image.open(image_path),dtype=np.float32)
    img = img.reshape(1, 500, 500, 1)
    name = image_path.split("/")
    for model in list_model_path:
        ort_sess = ort.InferenceSession(model)
        result[model.split("/")[-1]][1] += 1
        outputs = ort_sess.run(None, {'conv2d_input': img})
        if outputs[0] == 1 and name[-1][0:4] == "pneu":
            result[model.split("/")[-1]][0] += 1
        elif outputs[0] == 0 and name[-1][0:4] == "norm":
            result[model.split("/")[-1]][0] += 1

for line in result:
    print("For the model " + line)
    print("Success : ", str(result[line][0]), " on ", str(result[line][1]), " évaluations")
    print("Accuracy : " + str(result[line][0]/result[line][1]*100) + "%\n")
