import numpy as np
import typing
from PIL import Image

from scale.scaler import Algorithm, Scaler


class PillowScaler(Scaler):

    def __init__(self, algorithm: typing.Union[int, Algorithm],
                 src_image_shape: typing.Union[typing.Tuple[int, int], typing.Tuple[int, int, int]],
                 tar_image_shape: typing.Union[typing.Tuple[int, int], typing.Tuple[int, int, int]]):
        super().__init__(algorithm, src_image_shape, tar_image_shape)

    def scale_image_with(self, xin, tr, tc):

        if np.max(xin) < 1:
            raise Exception('Warning. Image might be not between 0 and 255')
        img = np.clip(xin, 0, 255)
        img = Image.fromarray(img)
        # The size of resize method in PIL.Image lib is (width x height).
        img_resized = img.resize((tc, tr), self.algorithm)
        img_resized = np.asarray(img_resized)
        return img_resized

    def _convert_scaling_algorithm(self, algorithm: Algorithm):
        if algorithm == Algorithm.NEAREST:
            return Image.NEAREST
        elif algorithm == Algorithm.LINEAR:
            return Image.BILINEAR
        elif algorithm == Algorithm.CUBIC:
            return Image.CUBIC
        elif algorithm == Algorithm.LANCZOS:
            return Image.LANCZOS
        elif algorithm == Algorithm.AREA:
            return Image.BOX
        else:
            raise NotImplementedError()
