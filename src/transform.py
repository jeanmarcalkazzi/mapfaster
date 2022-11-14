import albumentations as A
import numpy as np
from fastai.vision.all import DisplayedTransform, PILImage, store_attr


class AlbumentationsTransformAll(DisplayedTransform):
    # Apply Data Augmentations to all the images in the dataset (train and val)
    order = 0

    def __init__(self, train_aug):
        store_attr()

    def encodes(self, img: PILImage):
        aug_img = self.train_aug(image=np.array(img))["image"]
        return PILImage.create(aug_img)


def get_aug_all(im_size):
    return A.Compose(
        [
            A.LongestMaxSize(
                max_size=im_size[0], interpolation=0, always_apply=True, p=1
            ),
            A.PadIfNeeded(
                min_height=im_size[0],
                min_width=im_size[1],
                border_mode=0,
                value=0,
                always_apply=True,
                p=1.0,
            ),
        ]
    )


def get_item_tfms(im_size):
    return [AlbumentationsTransformAll(get_aug_all(im_size=im_size))]
