"""One-render opacity attenuation control for independent Bernoulli dropout.

For ideal alpha compositing with fixed depth order, independent masks and no
clipping/early termination, rendering q*alpha equals expected dropped render.
CUDA alpha caps, culling and termination can break exact equivalence.
This is a diagnostic/inference control, not a novel learned module.
"""
class OpacityExpectation:
    def __init__(self, gaussians, keep_probability=0.85):
        if not 0 < keep_probability <= 1:
            raise ValueError('keep_probability must be in (0,1]')
        self.gaussians=gaussians
        self.keep_probability=keep_probability

    def __getattr__(self,name):
        return getattr(self.gaussians,name)

    @property
    def get_opacity(self):
        return self.gaussians.get_opacity*self.keep_probability
