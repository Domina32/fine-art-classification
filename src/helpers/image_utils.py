import cv2
import numpy as np
import torchvision
import torchvision.transforms.functional as fn
from skimage import io

############ COMMON


# def resize_img(img, new_width=300, new_height=300):
#     """Resize an image using new  width and height"""
#     new_points = (new_width, new_height)
#     return cv2.resize(img, new_points, interpolation=cv2.INTER_LINEAR)


def resize_img(img, new_width=300, new_height=300):
    """Resize an image using new  width and height"""
    new_points = [new_width, new_height]

    return fn.resize(
        img,
        new_points,
        torchvision.transforms.InterpolationMode.BILINEAR,
        antialias=True,
    )


def change_channels(img):
    """Change image color channels from BGR to RGB"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)


############ WGA


def label_mapping(label):
    lbl = 0

    if label == "portrait":
        lbl = 1
    elif label == "landscape":
        lbl = 2
    elif label == "mythological":
        lbl = 3
    elif label == "genre":
        lbl = 4
    elif label == "still-life":
        lbl = 5
    elif label == "historical":
        lbl = 6
    elif label == "interior":
        lbl = 7
    elif label == "study":
        lbl = 8
    else:
        lbl = 9

    return lbl


def transform_URL(URL):
    """Transform URL from old format to new format in order to access jpg."""
    return "https://www.wga.hu/art/" + "/".join(URL.split("/")[-3:-1] + [URL.split("/")[-1].split(".")[0] + ".jpg"])


def load_img(URL):
    """Load an image from an URL, if found, and turn into RGB format."""
    image = io.imread(URL)
    return change_channels(image)


def url_to_numpy(url):
    try:
        img = load_img(url)
        resized_img = resize_img(img)
        numpy_img = np.expand_dims(resized_img[:, :, :-1], axis=0)[:, :, :, ::-1]

        assert numpy_img.shape[-1] == 3

        return numpy_img

    except Exception:
        pass


############ WIKIART
