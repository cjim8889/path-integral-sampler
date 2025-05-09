import sys
from dataclasses import dataclass, field
from typing import Callable, Tuple

import diffrax as dfx
import jax.numpy as jnp
import lineax
import numpyro.distributions as dist
from jax.random import PRNGKey
from jaxtyping import Array, PyTree  # type: ignore

# Disable host callbacks for errors since it leads to this bug:
# https://github.com/patrick-kidger/diffrax/pull/104
for module_name, module in sys.modules.items():
    if module_name.startswith("diffrax"):
        if hasattr(module, "branched_error_if"):
            module.branched_error_if = lambda *a, **kw: None  # type: ignore


@dataclass
class PathIntegralSampler:
    """
    Class defining loss and sampling functions for the path integral sampler.

    This approach consists of a training objective and sampling procedure for optimal
    control of the stochastic process

    .. math:: \\mathrm{d}\\mathbf{x}_t = \\mathbf{u}_t \\mathrm{d}t + \\mathrm{d}\\mathbf{w}_t ,

    where :math:`\\mathbf{w}_t` is a Wiener process. A network trained to find the
    control policy :math:`\\mathbf{u}_t(t, \\mathbf{x})` such that the loss function
    is minimized causes the above process to yield samples at time :math:`T` with
    the prespecified distribution :math:`\\mu(\\cdot)`. (Distributions and quantities
    at time :math:`t=T` are often referred to as "terminal".) The procedure also
    yields importance sampling weights :math:`w`.

    Notes:
        As explained in the paper, the control policy network is trained by constructing
        an SDE augmented by the trajectory's cost. This implementation uses a similar
        trick to simultaneously sample and compute importance sampling weights using
        any SDE solver.

    """

    get_log_mu: Callable[[Array], Array]
    """:math:`\\log \\mu(x)`, the log of the (unnormalized) terminal density to
    be sampled from.
    """
    x_size: int
    """size of :math:`x` vector.
    """
    t1: float
    """duration of diffusion.
    """
    dt0: float
    """initial timestep size for solver.
    """
    t0_eps: float = 1e-5
    """Small epsilon for the start time of the diffusion, for numerical stability."""
    solver: dfx.AbstractSolver = dfx.Euler()
    """SDE solver.
    """
    brownian_motion_tol: float = 1e-3
    """tolerance for `dfx.VirtualBrownianTree`.
    """
    y0: Array = field(init=False)
    """point at which diffusion begins (the origin).
    """

    sigma: float = 1.0

    def __post_init__(self):
        self.y0 = jnp.zeros(self.x_size + 1)

    def get_log_mu_0(self, x: Array) -> Array:
        """
        Gets log probability for the terminal distribution of the uncontrolled process.
        """
        effective_duration = self.t1 - self.t0_eps
        return dist.Normal(loc=0., scale=self.sigma * jnp.sqrt(effective_duration)).log_prob(x).sum()

    def get_drift(
        self, t: Array, x: Array, model: Callable[[Array, Array], Array]
    ) -> Array:
        """
        Gets the drift coefficient for augmented SDE.

        Args:
            t: time.
            x: state variable, with `x[:-1]` corresponding to :math:`x_t` and `x[-1]`
                corresponding to the trajectory's cost (:math:`y_t` in the paper).
            model: control policy network taking :math:`t` and :math:`x_t` as arguments.
        """
        # model output is u_pt (control before scaling by sigma, analogous to official impl.)
        u_pt = model(t, x[:-1])
        # Effective control applied to the SDE for x
        u_effective = u_pt * self.sigma
        # Cost rate matches 0.5 * ||u_pt||^2
        cost_rate = 0.5 * jnp.sum(u_pt**2)
        return jnp.append(u_effective, cost_rate)

    def get_diffusion_train(self, t: Array, x: Array, _) -> lineax.AbstractLinearOperator:
        """
        Gets the diffusion coefficient for the training SDE, returning a diagonal linear operator.

        Args:
            t: time.
            x: state variable, with `x[:-1]` corresponding to :math:`x_t` and `x[-1]`
                corresponding to the trajectory's cost (:math:`y_t` in the paper).
            _: unused argument required by diffrax.
        """
        diagonal_vector = jnp.append(self.sigma * jnp.ones(self.x_size), jnp.zeros(1))
        return lineax.DiagonalLinearOperator(diagonal_vector)

    def get_loss(self, model: PyTree, key: PRNGKey):
        """
        Gets loss for a single trajectory.

        Args:
            model: control policy network taking `t` and `x` as arguments.
            key: PRNG key for the trajectory.

        Returns:
            cost: approximation to :math:`\\int_{t_0}^{t_1} \\mathrm{d}t \\frac{1}{2} \\mathbf{u}_t(t, \\mathbf{x}_t ; \\theta) + \\Psi(\\mathbf{x}_T)`,
                where the second term is the terminal cost specified by the training
                procedure.
        """
        brownian_motion = dfx.VirtualBrownianTree(
            self.t0_eps, self.t1, self.brownian_motion_tol, (self.x_size + 1,), key
        )
        terms = dfx.MultiTerm(
            dfx.ODETerm(self.get_drift),
            dfx.ControlTerm(self.get_diffusion_train, brownian_motion),
        )
        return self._sample_x_cost(terms, model)[1]

    def get_diffusion_sampling(
        self, t: Array, x: Array, model: Callable[[Array, Array], Array]
    ) -> Array:
        """
        Gets the diffusion coefficient for sampling.

        Args:
            t: time.
            x: position.
            model: control policy network taking `t` and `x` as arguments.
        """
        # model output is u_pt (control before scaling by sigma)
        u_pt = model(t, x[:-1])
        # The diffusion term for the cost/log-weight accumulator is u_pt
        # This corresponds to (u_effective / sigma) dW_t = (u_pt * sigma / sigma) dW_t = u_pt dW_t
        return jnp.append(
            self.sigma * jnp.eye(self.x_size),
            u_pt[None, :],
            axis=0
        )

    def _sample_x_cost(self, terms: dfx.MultiTerm, model: PyTree) -> Tuple[Array, Array]:
        """
        Helper to get sample and its cost.
        """
        # TODO: custom control term! f is the identity stacked on u.
        y1 = dfx.diffeqsolve(
            terms,
            self.solver,
            self.t0_eps,
            self.t1,
            self.dt0,
            self.y0,
            args=model,
            saveat=dfx.SaveAt(t1=True),
        ).ys[-1]
        # Split up augmented state
        x_T = y1[:-1]
        x_T = jnp.nan_to_num(x_T)  # Handle NaN/Inf for numerical stability
        y_T = y1[-1]
        # Add terminal cost
        Psi_T = self.get_log_mu_0(x_T) - self.get_log_mu(x_T)
        cost = y_T + Psi_T
        return x_T, cost

    def get_sample(self, model: PyTree, key: PRNGKey) -> Tuple[Array, Array]:
        """
        Generates a sample. To generate multiple samples, `vmap` over `key`.

        Args:
            model: control policy network taking `t` and `x` as arguments.
            key: PRNG key for the trajectory.

        Returns:
            x_T: sample.
            log_w: log of the importance sampling weight.
        """
        # TODO: custom control term! f is the identity stacked on u.
        brownian_motion = dfx.VirtualBrownianTree(
            self.t0_eps, self.t1, self.brownian_motion_tol, (self.x_size,), key
        )
        terms = dfx.MultiTerm(
            dfx.ODETerm(self.get_drift),
            dfx.ControlTerm(self.get_diffusion_sampling, brownian_motion),
        )
        x_T, cost = self._sample_x_cost(terms, model)
        log_w = -cost
        return x_T, log_w
