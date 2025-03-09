import torch
from torch.optim.optimizer import Optimizer, required

class ContourWalkingOptimizer(Optimizer):
    def __init__(self, params, lr=required):
        # Initialize the optimizer with the parameters and hyperparameters (e.g., learning rate)
        defaults = dict(lr=lr)
        super(ContourWalkingOptimizer, self).__init__(params, defaults)
        self._is_first_time_calc = True
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Loop over parameter groups (usually one group, but could be more)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Get gradient data for the parameter
                grad = p.grad.data
                # If this is the first call, initialize a random direction for p
                if self._is_first_time_calc:
                    d = torch.randn_like(grad)
                    self.state.setdefault(p, {})['d'] = d
                else:
                    d = self.state[p]['d']

                ## take here d from self
                d_proj = self.__act(grad,d)
                ## store here d_proj into self
                self.state[p]['d'] = d_proj.clone()
                # Custom update: Here, we perform a simple gradient descent step.
                # This is equivalent to: p = p - lr * grad
                p.data.add_(-group['lr'], d_proj)
        if self._is_first_time_calc:
            self._is_first_time_calc = False
        return loss

    def __act(self, grad, d):
        grad_norm = torch.norm(grad)
        if grad_norm < 1e-8:
            return d  # or return d unchanged if gradient is nearly zero

        # Compute dot product (flatten the tensors in case they are not 1D)
        dot = torch.dot(grad.view(-1), d.view(-1))

        # Project d onto the null space of grad
        d_proj = d - (dot / (grad_norm ** 2)) * grad
        norm_d_proj = torch.norm(d_proj)
        if norm_d_proj < 1e-8:
            return d  # or return d if the projection is negligible

        d_proj = d_proj / norm_d_proj  # normalize the projected direction
        return d_proj


# Example usage:
if __name__ == '__main__':
    # A simple linear model for demonstration
    model = torch.nn.Linear(10, 1)
    # Use Mean Squared Error as our loss function
    criterion = torch.nn.MSELoss()

    # Instantiate our custom optimizer
    L = 0.2
    eta = 0.00001  # step size
    T = int(L / eta)  # number of steps
    optimizer_contour = ContourWalkingOptimizer(model.parameters(), lr=eta)
    optimizer_score = torch.optim.SGD(model.parameters(), lr=0.01)

    num_samples = 100
    # Dummy data: input x and target y
    x = torch.randn(num_samples, 10)
    y = torch.randn(num_samples, 1)

    # Training loop (simplified)
    for epoch in range(100):
        optimizer_score.zero_grad()  # reset gradients to zero
        output = model(x)
        loss = criterion(output, y)
        loss.backward()  # compute gradients

        optimizer_score.step()  # update parameters
        print(f"Epoch {epoch}: loss = {loss.item()}")
    optimized_params = [p.detach().clone() for p in model.parameters()]
    for t in range(T):
        optimizer_contour.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer_contour.step()
        if t % 1000 == 0:
            print(f"Step {t}: loss = {loss.item()}")
    curr_params = model.parameters()
    for i, (param_score, param_contour) in enumerate(zip(optimized_params, curr_params)):
        print(f"Parameter {i}: score = {torch.norm(param_score - param_contour)}")
