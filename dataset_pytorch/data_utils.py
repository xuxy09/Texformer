import numpy as np
import cv2
import torch



def rescale_image(image):
    """
        Rescales image from [0, 1] to [-1, 1]
        Resnet v2 style preprocessing.
    """
    image = image - 0.5
    image *= 2.0
    return image


def normalize_image(image):
    """
    normalize the numpy image(HWC)
    :param image: numpy image
    :return: normalized image
    """
    image = 2.0 * ((image / 255.) - 0.5)
    return image


def jitter_center(center, trans_max):
    """
    randomly shift human center for future crop
    :param center: human center
    :param trans_max: max trans margin
    :return: shifted center
    """
    rand_trans = np.random.uniform(low=-trans_max, high=trans_max, size=[2])
    return center + rand_trans


def jitter_scale(image, image_size, masks, center, scale_low, scale_high):
    """
    rescale the image and gt
    :param image:
    :param image_size:
    :param masks:
    :param center:
    :param scale_low:
    :param scale_high:
    :return:
    """
    assert image.shape == masks.shape
    scale_factor = np.random.uniform(low=scale_low, high=scale_high, size=[1])
    new_size = image_size * scale_factor
    new_image = cv2.resize(image, new_size)
    new_masks = cv2.resize(masks, new_size)

    actual_factor = new_image.shape[:2] / image_size

    new_center_x = center[0] * actual_factor[1]
    new_center_y = center[1] * actual_factor[0]

    return new_image, new_masks, [new_center_x, new_center_y]


def resize_img(img, scale_factor):
    """
    resize image with scale factor
    :param img:
    :param scale_factor:
    :return:
    """
    new_size = (np.floor(np.array(img.shape[0:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])
    ]
    return new_img, actual_factor


def reshape_img(img, new_shape):
    """
    resize image with new shape
    :param img:
    :param new_shape:
    :return:
    """
    h = new_shape[1]
    w = new_shape[0]
    if img.shape[0] == w and img.shape[1] == h:
        return img
    else:
        return cv2.resize(img, (h, w))


def scale_and_crop(image, scale, center, img_size, safe_margin):
    image_scaled, scale_factors = resize_img(image, scale)
    # Swap so it's [x, y]
    scale_factors = [scale_factors[1], scale_factors[0]]
    center_scaled = np.round(center * scale_factors).astype(np.int)

    margin = int(img_size / 2)
    safe_margin = margin + safe_margin
    image_pad = np.pad(
        image_scaled, ((safe_margin,), (safe_margin,), (0,)), mode='edge')
    center_pad = center_scaled + safe_margin
    # figure out starting point
    start_pt = center_pad - margin
    end_pt = center_pad + margin
    # crop:
    crop = image_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
    proc_param = {
        'scale': scale,
        'start_pt': start_pt,
        'end_pt': end_pt,
        'img_size': img_size
    }

    return crop, proc_param

class Resize(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (int or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))


        return img

class Resize_pose(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (int or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        
        self.output_size = output_size

    def __call__(self, image):

        
        

        new_h, new_w = self.output_size

        img = cv2.resize(image, (new_w, new_h))


        return img


class RandomCrop(object):
    """Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return image


class ResizeCrop:
    def __init__(self, center_trans_max, output_size, safe_margin=40):
        self.output_size = output_size
        self.center_trans_max = center_trans_max
        self.safe_margin = safe_margin

    def __call__(self, image, center):

        if np.max(image.shape[:2]) != self.output_size:
            scale = (float(self.output_size) / np.max(image.shape[:2]))
        else:
            scale = 1.

        center = jitter_center(center, self.center_trans_max)

        image, _ = scale_and_crop(image, scale, center, self.output_size, safe_margin=self.safe_margin)
        image = reshape_img(image, new_shape=(self.output_size, self.output_size))
        return image


class RandomFlip:
    """
    random horizontally flip the image
    """

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image):
        if np.random.random() < self.flip_prob:
            image = cv2.flip(image, 1)
        return image

class RandomFlip_deepfashion:
    """
    random horizontally flip the image
    """

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image):
        if np.random.random() < self.flip_prob:
            image = cv2.flip(image, 1)
        return image

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, normalize):
        self.normalize = normalize

    def __call__(self, image):


        # unnormalized image for hmr
        if self.normalize:
            image = normalize_image(image)

        # swap HWC axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))

        return torch.from_numpy(image).float()
