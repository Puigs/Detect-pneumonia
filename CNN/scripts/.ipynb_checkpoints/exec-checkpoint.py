#!/usr/bin/env python 

import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import glob
import sys
from datetime import datetime

if len(sys.argv) < 3:
    print("This script take in argument two inputs")
    print("A model onnx")
    print("A directory which contains images")
    exit()

if sys.argv[2][-1:] == "/":
    sys.argv[2] += "*"
else:
    sys.argv[2] += "/*"

list_image_path = glob.glob(sys.argv[2])
ort_sess = ort.InferenceSession(sys.argv[1])
n_total = 0
n_succed = 0
for image_path in list_image_path:
    n_total += 1
    img = np.array(Image.open(image_path),dtype=np.float32)
    img = img.reshape(1, 500, 500, 1)
    outputs = ort_sess.run(None, {'conv2d_input': img})
    name = image_path.split("/")
    if outputs[0] == 1 and name[-1][0:4] == "pneu":
        n_succed += 1
    elif outputs[0] == 0 and name[-1][0:4] == "norm":
        n_succed += 1

print("Success : ", str(n_succed), " on ", str(n_total), " évaluations")
print("Accuracy : " + str(n_succed/n_total*100) + "%")

line = datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " - Model : " + sys.argv[1] + " - Dataset : " + sys.argv[2] + " - Success : " + str(n_succed) + " on " + str(n_total) + " valuations" + "Accuracy : " + str(n_succed/n_total*100) + "%" + "\n"
with open('results.txt', 'a') as f:
    f.write(line)
