import numpy as np
import cvxpy as cp


class Attack:

    def attack(self, src, tar, scaler):
        if src.dtype != np.uint8:
            raise Exception('Source or target image have dtype != np.uint8')

        att_image = np.zeros(src.shape)
        # There is not work for gray image.
        for ch in range(src.shape[2]):
            print(f'Channel {ch}...')
            att_ch = self._attack_on_one_channel(ch=ch, src=src, tar=tar, scaler=scaler)
            att_image[:, :, ch] = att_ch

        att_image = att_image.astype(np.uint8)
        return att_image

    @staticmethod
    def _attack_on_one_channel(ch, src,  tar, scaler):

        src_image = src[:, :, ch]
        target_image = tar[:, :, ch]
        delta = cp.Variable(src_image.shape)
        att_img = (src_image + delta)

        cl = scaler.cl_matrix
        cr = scaler.cr_matrix
        obj = cp.pnorm(cl @ att_img @ cr - target_image, 2) + 0.3 * cp.pnorm(delta, 2) + 0.5 * cp.norm(delta, 'inf')

        constr1 = att_img <= 255
        constr2 = att_img >= 0

        prob = cp.Problem(cp.Minimize(obj), [constr1, constr2])

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
