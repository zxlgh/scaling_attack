from abc import ABC, abstractmethod
import typing
import enum
import numpy as np


class Algorithm(enum.Enum):
    """
    The supported scaling algorithms, so that we have one unique
    way to select NEAREST, or LINEAR, CUBIC for the different
    scaling libraries.
    """
    NEAREST = 100
    LINEAR = 200
    CUBIC = 300
    LANCZOS = 400
    AREA = 500


class Scaler(ABC):
    """
    The setup for a scaling approach.
    Includes the library (e.g. PIL or OpenCV) and the interpolation algorithm.
    This class saves the CL, CR, interpolation algorithm and the image sizes before and after.
    The reason is that CL and CR are only valid for a fixed size in combination with the right interpolation algorithm,
    so that we should not save them separately.
    """

    def __init__(self, algorithm: typing.Union[int, Algorithm],
                 src_image_shape: typing.Union[typing.Tuple[int, int], typing.Tuple[int, int, int]],
                 tar_image_shape: typing.Union[typing.Tuple[int, int], typing.Tuple[int, int, int]]):
        assert isinstance(algorithm, int) or isinstance(algorithm, Algorithm)
        if isinstance(algorithm, Algorithm):
            algorithm = self._convert_scaling_algorithm(algorithm)
        self.algorithm = algorithm

        self.src_image_shape = src_image_shape
        self.tar_image_shape = tar_image_shape

        self.cl_matrix, self.cr_matrix = self._get_matrix_cr_cl()

    @abstractmethod
    def scale_image_with(self, xin: np.ndarray, tr, tc) -> np.ndarray:
        pass

    @abstractmethod
    def _convert_scaling_algorithm(self, algorithm: Algorithm):
        """
        Convert enum for common algorithm specification to scaling library value.
        :param algorithm:
        :return:
        """
        pass

    def __get_scale_cr_cl(self, sh, tr, tc):
        """
        Helper Function for Coefficients Recovery (to get CL and CR matrix)
        :param sh: shape
        :return:
        """
        mint = 255
        im_max = mint * np.identity(sh)

        im_max_scaled = self.scale_image_with(xin=im_max, tr=tr, tc=tc)

        cm = im_max_scaled / 255
        return cm

    def _get_matrix_cr_cl(self):
        """
        Coefficients Recovery
        :return: CL and CR matrix
        """

        cl = self.__get_scale_cr_cl(sh=self.src_image_shape[0],
                                    tr=self.tar_image_shape[0],
                                    tc=self.src_image_shape[0])
        cr = self.__get_scale_cr_cl(sh=self.src_image_shape[1],
                                    tr=self.src_image_shape[1],
                                    tc=self.tar_image_shape[1])
        # print(cl.shape, cr.shape)
        # normalize
        cl = cl / cl.sum(axis=1)[:, np.newaxis]
        cr = cr / cr.sum(axis=0)[np.newaxis, :]

        return cl, cr
