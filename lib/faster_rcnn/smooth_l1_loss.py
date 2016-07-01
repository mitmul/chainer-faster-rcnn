from chainer import cuda
from chainer import function
from chainer.utils import type_check

import numpy


class SmoothL1Loss(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 4)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x0, x1, self.inside_weights, self.outside_weights = inputs
        self.diff = self.inside_weights * (x0 - x1)
        y = xp.square(self.diff)
        mask = y > (self.delta ** 2)
        y -= mask * xp.square(abs(self.diff) - self.delta)
        y *= 0.5
        y *= self.outside_weights
        return y.sum(axis=1),

    def backward(self, inputs, gy):
        xp = cuda.get_array_module(*inputs)
        mask = xp.abs(self.diff) <= self.delta
        gx = gy[0].reshape(gy[0].shape + (1,) * (self.diff.ndim - 1)) * \
            xp.where(mask, self.diff, self.delta * xp.sign(self.diff))
        gx *= self.inside_weights
        gx *= self.outside_weights
        return gx, -gx


def smooth_l1_loss(x, t):
    return SmoothL1Loss()(x, t, inside_weights, outside_weights)
