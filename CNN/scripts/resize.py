#!/usr/bin/env python 

from PIL import Image, ImageOps
import glob
import sys

if len(sys.argv) < 6:
    print("This script take in argument 5 inputs:")
    print("A directory path where are the images")
    print("A directory path for save the resized images")
    print("The new width of every image")
    print("The new heigth of every image")
    print("Prefix of the image name")
    exit()


list_image_path = glob.glob(sys.argv[1] + "/*")
index = 0
for file_path in list_image_path:
    index += 1
    image = Image.open(file_path)
    gray_image = ImageOps.grayscale(image)
    new_image = gray_image.resize((int(sys.argv[3]), int(sys.argv[4])))
    new_image.save(sys.argv[2] + "/" + str(sys.argv[5]) + "-" + str(index) + ".jpeg")