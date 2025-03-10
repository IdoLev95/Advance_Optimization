import numpy as np
import matplotlib.pyplot as plt
from contour_walk_coherent_walk import go_on_contour_coherent

def go_on_contour_random_direction(x_init, f, f_grad, eta, T):
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
    all_values = list()
    all_locs = list()
    for t in range(T):
        grad = f_grad(x)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-8:
            # print(f"Step {t}: gradient is nearly zero, stopping.")
            break
        # Generate a random direction in R^n
        d = np.random.randn(*x.shape)
        # Project d onto the null space of grad: d_proj is orthogonal to grad
        d_proj = d - (np.dot(grad, d) / (grad_norm ** 2)) * grad
        norm_d_proj = np.linalg.norm(d_proj)
        if norm_d_proj < 1e-8:
            # If the projected direction is almost zero, try a different random vector.
            continue
        d_proj = d_proj / norm_d_proj  # normalize the direction

        # Update x by taking a step of size eta in the d_proj direction
        all_values.append(f(x))
        all_locs.append(x)
        x = x + eta * d_proj
        # print(f"Step {t}: f(x) = {f(x)}")
    return x, all_values, all_locs


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

    np.random.seed(0)
    x_init = np.array([1.0, 2.0])  # starting point in R^2
    L = 2
    eta = 0.00001  # step size
    T = int(L / eta)  # number of steps


    final_x,all_values_first, all_locs_first= go_on_contour_random_direction(x_init, f, f_grad, eta, T)

    print("Final x:", final_x)
    print(f"Distance from origin: {np.linalg.norm(final_x - x_init)} and f(x) = {f(final_x)}")
    final_x, all_locs_second, all_values_second = go_on_contour_coherent(x_init, f, f_grad, eta, T)
    print("Final x:", final_x)
    print(f"Distance from origin: {np.linalg.norm(final_x - x_init)} and f(x) = {f(final_x)}")
    figure,axes = plt.subplots(1,2,figsize=(16,8))
    axes[0].plot(range(len(all_values_first)),all_values_first,label="Random Direction")
    axes[0].plot(range(len(all_values_second)),all_values_second,label="Coherent Walk")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("f(x)")
    axes[0].set_title("f(x) vs Iteration for coherent and non coherent walks")
    axes[0].legend()
    axes[1].plot(range(len(all_locs_first)),[np.linalg.norm(x-x_init) for x in all_locs_first],label="Random Direction")
    axes[1].plot(range(len(all_locs_second)),[np.linalg.norm(x-x_init) for x in all_locs_second],label="Coherent Walk")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Distance from init")
    axes[1].set_title("Distance from init vs Iteration for coherent and non coherent walks")
    axes[1].legend()
    plt.tight_layout()
    plt.show()