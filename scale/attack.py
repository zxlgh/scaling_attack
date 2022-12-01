import numpy as np
import cvxpy as cp
import typing

from scale.scaler import Scaler


class Attack:

    def __init__(self, src: np.ndarray,
                 tar: typing.List[np.ndarray],
                 scaler: typing.List[Scaler]):
        self.src = src
        self.tar = tar
        self.scaler = scaler

    def attack(self):
        if self.src.dtype != np.uint8:
            raise Exception('Source or target image have dtype != np.uint8')

        att_image = np.zeros(self.src.shape)
        # This is not work for gray image.
        for ch in range(self.src.shape[2]):
            print(f'Channel {ch}...')
            att_ch = self._attack_on_one_channel(ch=ch)
            att_image[:, :, ch] = att_ch

        att_image = att_image.astype(np.uint8)
        return att_image

    def _attack_on_one_channel(self, ch):
        """
        solving the optimization problem by cvxpy lib.
        :param ch: channel of image
        :return:
        """

        src_image = self.src[:, :, ch]

        delta = cp.Variable(src_image.shape)
        att_img = (src_image + delta)

        obj = cp.pnorm(delta, 2)
        constr = []
        for tar, scaler in zip(self.tar, self.scaler):
            target_image = tar[:, :, ch]
            cl = scaler.cl_matrix
            cr = scaler.cr_matrix
            obj += 2 * cp.pnorm(cl @ att_img @ cr - target_image, 2)

        constr.append(att_img <= 255)
        constr.append(att_img >= 0)

        prob = cp.Problem(cp.Minimize(obj), constr)

        try:
            prob.solve()
        except prob.status != cp.OPTIMAL:
            print('Can not find the optim answer with default solver.')
        else:
            try:
                prob.solve(solver=cp.ECOS)
            except prob.status != cp.OPTIMAL:
                print('Can not find the optim answer with ECOS solver.')
        assert delta.value is not None
        attack_image_all = src_image + delta.value
        attack_image_all = np.clip(np.round(attack_image_all), 0, 255)
        return attack_image_all
