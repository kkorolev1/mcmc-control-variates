import jax.numpy as np
from folx import forward_laplacian


class Generator:
    def __init__(self, grad_log_p, g):
        self.grad_log_p = grad_log_p
        self.fwd_g = forward_laplacian(g)
    
    def __call__(self, x):
        output_g = self.fwd_g(x)
        return np.dot(self.grad_log_p(x), output_g.jacobian.dense_array) + output_g.laplacian
