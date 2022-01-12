from PIL import Image
from torchvision.transforms import (Resize, Compose, ToTensor, Normalize, CenterCrop, RandomCrop, ColorJitter,
                                    RandomResizedCrop, RandomHorizontalFlip)

AVAI_CHOICES = ['random_flip', 'random_resized_crop', 'normalize', 'random_crop', 'center_crop', 'colorjitter']

INTERPOLATION_MODES = {
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'nearest': Image.NEAREST
}


def build_transform(cfg, is_train=True, choices=None):
    """Build transformation function.

    Args:
        cfg (CfgNode): config.
        is_train (bool, optional): for training (True) or test (False).
            Default is True.
        choices (list, optional): list of strings which will overwrite
            cfg.INPUT.TRANSFORMS if given. Default is None.
    """

    if choices is None:
        choices = cfg.INPUT.TEST.TRANSFORMS

    for choice in choices:
        assert choice in AVAI_CHOICES, \
            'Invalid transform choice ({}), ' \
            'expected to be one of {}'.format(choice, AVAI_CHOICES)

    expected_size = '{}x{}'.format(cfg.INPUT.SIZE[0], cfg.INPUT.SIZE[1])

    normalize = Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

    if is_train:
        return _build_transform_train(cfg, choices, expected_size, normalize)
    else:
        return _build_transform_test(cfg, choices, expected_size, normalize)


def _build_transform_train(cfg, choices, expected_size, normalize):
    print('Building transform_train')
    tfm_train = []

    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]

    print('+ resize to {}'.format(expected_size))
    tfm_train += [Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]

    if 'random_flip' in choices:
        print('+ random flip')
        tfm_train += [RandomHorizontalFlip(p=0.5)]

    if 'random_crop' in choices:
        crop_padding = cfg.INPUT.CROP_PADDING
        print('+ random crop (padding = {})'.format(crop_padding))
        tfm_train += [RandomCrop(cfg.INPUT.CROP_SIZE, padding=crop_padding)]

    if 'random_resized_crop' in choices:
        print('+ random resized crop')
        tfm_train += [
            RandomResizedCrop(cfg.INPUT.CROP_SIZE, interpolation=interp_mode)
        ]

    if 'center_crop' in choices:
        print('+ center crop (on 1.125x enlarged input)')
        enlarged_size = [int(x * 1.125) for x in cfg.INPUT.SIZE]
        tfm_train += [Resize(enlarged_size, interpolation=interp_mode)]
        tfm_train += [CenterCrop(cfg.INPUT.CROP_SIZE)]

    if 'colorjitter' in choices:
        print('+ color jitter')
        tfm_train += [
            ColorJitter(
                brightness=cfg.INPUT.COLORJITTER_SCALAR * 0.8,
                contrast=cfg.INPUT.COLORJITTER_SCALAR * 0.8,
                saturation=cfg.INPUT.COLORJITTER_SCALAR * 0.8,
                hue=cfg.INPUT.COLORJITTER_SCALAR * 0.2
            )
        ]

    ####### transformation before to tensor of range [0, 1]
    print('+ to torch tensor of range [0, 1]')
    tfm_train += [ToTensor()]
    ####### transformation after to tensor of range [0, 1]

    if 'normalize' in choices:
        print(
            '+ normalization (mean={}, '
            'std={})'.format(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
        )
        tfm_train += [normalize]

    tfm_train = Compose(tfm_train)

    return tfm_train


def _build_transform_test(cfg, choices, expected_size, normalize):
    print('Building transform_test')
    tfm_test = []

    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]

    print('+ resize to {}'.format(expected_size))
    tfm_test += [Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]

    if 'center_crop' in choices:
        print('+ center crop (on 1.125x enlarged input)')
        enlarged_size = [int(x * 1.125) for x in cfg.INPUT.SIZE]
        tfm_test += [Resize(enlarged_size, interpolation=interp_mode)]
        tfm_test += [CenterCrop(cfg.INPUT.CROP_SIZE)]

    print('+ to torch tensor of range [0, 1]')
    tfm_test += [ToTensor()]

    if 'normalize' in choices:
        print(
            '+ normalization (mean={}, '
            'std={})'.format(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
        )
        tfm_test += [normalize]

    tfm_test = Compose(tfm_test)

    return tfm_test
