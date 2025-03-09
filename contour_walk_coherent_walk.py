import numpy as np
from numba import njit

@njit
def act(grad,d,x,eta):
    grad_norm = np.linalg.norm(grad)
    if grad_norm < 1e-8:
        # print(f"Step {t}: gradient is nearly zero, stopping.")
        return x,d
    # Generate a random direction in R^n

    # Project d onto the null space of grad: d_proj is orthogonal to grad
    d_proj = d - (np.dot(grad, d) / (grad_norm ** 2)) * grad
    norm_d_proj = np.linalg.norm(d_proj)
    if norm_d_proj < 1e-8:
        # If the projected direction is almost zero, try a different random vector.
        return x,d
    d_proj = d_proj / norm_d_proj  # normalize the direction

    # Update x by taking a step of size eta in the d_proj direction
    x = x + eta * d_proj
    return x,d_proj
def go_on_contour_coherent(x_init, f, f_grad, eta, T):
    """
    Moves along the level set (contour) of f by taking T steps.
    At each step, the update is in a random direction projected to be orthogonal to the gradient of f,
    so that to first order, f(x) does not change.

    Parameters:
      x_init: np.array, initial point
      f: callable, function f: R^n -> R (assumed convex)
      f_grad: callable, gradient of f: R^n -> R^n
      eta: float, step size
      T: int, number of steps

    Returns:
      x: np.array, final point after T steps.
    """
    x = np.copy(x_init)
    d = np.random.randn(*x.shape)
    for t in range(T):
        grad = f_grad(x)
        x, d_proj = act(grad,d,x,eta)
        d = d_proj
        # print(f"Step {t}: f(x) = {f(x)}")
    return x


# Example usage:
if __name__ == "__main__":
    # For demonstration, we use f(x) = ||x||^2, which is convex.
    # Its gradient is f_grad(x) = 2*x.
    # Note: For f(x)=||x||^2, the level sets are spheres, so any step orthogonal to x (which equals the gradient up to a factor)
    # will keep f(x) constant (to first order, and exactly in this quadratic case).

    def f(x):
        return np.sum(x ** 4)


    def f_grad(x):
        return 4 * (x ** 3)


    x_init = np.array([1.0, 2.0])  # starting point in R^2
    L = 2
    eta = 0.000001  # step size
    T = int(L / eta)  # number of steps

    final_x = go_on_contour_coherent(x_init, f, f_grad, eta, T)
    print("Final x:", final_x)
    print("Distance from origin:", np.linalg.norm(final_x-x_init))
    print(f"At the beginning f(x_init) = {f(x_init)} and in the end f(final_x) = {f(final_x)}")
