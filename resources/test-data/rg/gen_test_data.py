import numpy as np
from PIL import Image


def gen_uniform_color_img(n, path, dim=(1000, 1000, 3)):
    for i in range(n):
        imarray = np.random.rand(*dim) * 255
        im = Image.fromarray(imarray.astype("uint8")).convert("RGBA")
        im.save(path + "/" + str(i) + ".png")


def gen_uniform_depth_img(n, path, dim=(1000, 1000)):
    for i in range(n):
        imarray = np.random.rand(*dim) * 255
        np.save(path + "/" + str(i) + ".npy", imarray)


if __name__ == "__main__":
    gen_uniform_color_img(16, "./rgb")
    gen_uniform_depth_img(16, "./depth")
