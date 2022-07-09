#!/bin/python3

import os
from glob import glob
from PIL import Image
from PIL import ImageEnhance
import argparse

def max_images(path_data):
    list_folder = glob(os.path.join(path_data, "*/"))
    max_found = 0
    max_len_class = 0
    for folder in list_folder:
        nb_imgs = len(glob(os.path.join(folder, '*')))
        len_class = len(folder.split('/')[-2])
        if nb_imgs > max_found:
            max_found = nb_imgs
        if len_class > max_len_class:
            max_len_class = len_class
    return max_found, max_len_class


def preprocess(path_data, path_result, luminosity, contraste, sharpness, orientation, size):
    max_found, max_len_class = max_images(path_data)
    list_class = glob(os.path.join(path_data, "*/"))
    for sub_path in list_class:
        classe = sub_path.split('/')[-2]
        classe_folder = os.path.join(path_result, classe)
        os.mkdir(classe_folder)
        imgs = glob(os.path.join(sub_path, '*'))
        for img in imgs:
            img_name = img.split('/')[-1]
            dest = os.path.join(classe_folder, img_name)
            image = Image.open(img).convert('L')
            image = ImageEnhance.Brightness(image).enhance(luminosity)
            image = ImageEnhance.Contrast(image).enhance(contraste)
            image = ImageEnhance.Sharpness(image).enhance(sharpness)
            image = image.rotate(orientation, expand=True)
            image = image.resize(size)
            image.save(dest)
        if max_found > len(imgs):
            for i in range (0, max_found - len(imgs), 1):
                img_name = imgs[i].split('/')[-1]
                dest = os.path.join(classe_folder, "complete_" + img_name)
                image = Image.open(imgs[i]).convert('L')
                image = image.resize(size)
                image.save(dest)
        classe += (' ' * (max_len_class - len(classe)))
        print(f"Folder {classe} done ({len(imgs)} elements", end="")
        if max_found > len(imgs):
            print(f", {max_found - len(imgs)} raw inputs added to reach {max_found} elements", end="")
        print(')')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use to make preprocessing on folder")
    parser.add_argument(
        '--path_data',
        '-d',
        type=str,
        help="The path to the folder where inputs are",
        required=True
    )
    parser.add_argument(
        '--path_result',
        '-r',
        type=str,
        help="The path where the datas will be saved",
        required=True
    )
    parser.add_argument(
        "--luminosity",
        '-l',
        type=float,
        help="The luminosity to apply",
        default=1.0
    )
    parser.add_argument(
        "--contraste",
        '-c',
        type=float,
        help="The contraste to apply",
        default=1.0
    )
    parser.add_argument(
        "--sharpness",
        '-s',
        type=float,
        help="The sharpness to apply",
        default=1.0
    )
    parser.add_argument(
        "--orientation",
        '-o',
        type=int,
        help="Orientation to apply : 0 = original, 90 = 90° anti-clockwise, 180 = reversed, 270 = 90° clockwise",
        choices=[0, 90, 180, 270],
        default=0
    )
    parser.add_argument(
        "--resize",
        help="Size to resize. First is X, second is Y",
        default=['500', '500'],
        nargs='+'
    )
    args = parser.parse_args()
    if os.path.exists(args.path_result):
        print(f"{args.path_result} already exist, please select another")
        exit(1)
    if not os.path.exists(args.path_data) or len(glob(os.path.join(args.path_data, "*/"))) == 0:
        print(f"{args.path_data} do not exist, please select another")
        exit(1)
    if len(args.resize) != 2 or not args.resize[0].isnumeric() or not args.resize[1].isnumeric() :
        print("Resize has not exactly 2 arguments. Cannot proceed")
        exit(1)
    os.mkdir(args.path_result)
    preprocess(args.path_data, args.path_result, args.luminosity, args.contraste, args.sharpness, args.orientation, list(map(int, args.resize)))
