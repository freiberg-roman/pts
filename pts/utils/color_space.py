import numpy as np

color_space = np.asarray(
    [
        [154, 205, 50],  # yellow green
        [85, 107, 47],  # dark olive green
        [107, 142, 35],  # olive drab
        [124, 252, 0],  # lawn green
        [127, 255, 0],  # chart reuse
        [173, 255, 47],  # green yellow
        [0, 100, 0],  # dark green
        [0, 128, 0],  # green
        [34, 139, 34],  # forest green
        [0, 255, 0],  # lime
        [50, 205, 50],  # lime green
        [144, 238, 144],  # light green
        [152, 251, 152],  # pale green
        [143, 188, 143],  # dark sea green
        [0, 250, 154],  # medium spring green
        [0, 255, 127],  # spring green
        [46, 139, 87],  # sea green
        [0, 255, 255],  # cyan
        [224, 255, 255],  # light cyan
        [0, 206, 209],  # dark turquoise
        [64, 224, 208],  # turquoise
        [72, 209, 204],  # medium turquoise
        [175, 238, 238],  # pale turquoise
        [127, 255, 212],  # aqua marine
        [176, 224, 230],  # powder blue
        [95, 158, 160],  # cadet blue
        [70, 130, 180],  # steel blue
        [100, 149, 237],  # corn flower blue
        [0, 191, 255],  # deep sky blue
        [30, 144, 255],  # dodger blue
        [173, 216, 230],  # light blue
        [135, 206, 235],  # sky blue
        [135, 206, 250],  # light sky blue
        [25, 25, 112],  # midnight blue
        [0, 0, 128],  # navy
        [0, 0, 139],  # dark blue
        [0, 0, 205],  # medium blue
        [0, 0, 255],  # blue
        [65, 105, 225],  # royal blue
        [138, 43, 226],  # blue violet
        [75, 0, 130],  # indigo
        [72, 61, 139],  # dark slate blue
    ],
    dtype=np.uint8,
)

color_space_red = np.asarray(
    [
        [128, 0, 0],  # maroon
        [139, 0, 0],  # dark red
        [165, 42, 42],  # brown
        [178, 34, 34],  # firebrick
        [220, 20, 60],  # crimson
        [255, 0, 0],  # red
        [255, 99, 71],  # tomato
        [255, 127, 80],  # coral
        [205, 92, 92],  # indian red
        [240, 128, 128],  # light coral
        [233, 150, 122],  # dark salmon
        [250, 128, 114],  # salmon
        [255, 160, 122],  # light salmon
        [255, 69, 0],  # orange red
        [255, 140, 0],  # dark orange
        [255, 165, 0],  # orange
        [160, 82, 45],  # sienna
        [219, 112, 147],  # pale viaolet red
    ],
    dtype=np.uint8,
)


class BColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
