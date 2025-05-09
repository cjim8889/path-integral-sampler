from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.random import PRNGKey, split
from jaxtyping import Array  # type: ignore

# from .composed import MLP
from .init import init_linear_weights, xavier_init, zero_init
from .positional_encoding import PositionalEncoding


class ControlNet(eqx.Module):
    """
    Affine transformation of score parametrized by two neural networks, using the
    architecture close to the one from the `path integral sampler paper <https://arxiv.org/abs/2111.15141>`_.
    This is of the form :math:`a_\\theta(t, x) + b_\\theta(t, x) \\nabla \\log \\mu(x)`,
    with the output initialized to zero using an overall learn multiplicative factor.
    """

    get_score_mu: Callable[[Array, Array], Array] = eqx.static_field()
    T: float = eqx.static_field()
    t_pos_encoding: Callable
    t_emb: Callable
    x_emb: Callable
    const_net: Callable
    coeff_net: Callable
    nn_clip: float = eqx.static_field()
    score_clip: float = eqx.static_field()
 
    def __init__(
        self,
        x_dim: int,
        get_score_mu: Callable[[Array, Array], Array],
        T: float = 1.0,
        L_max: int = 64,
        emb_dim: int = 64,
        embed_width_size: int = 64,
        embed_depth: int = 2,
        width_size: int = 64,
        depth: int = 3,
        scalar_coeff_net: bool = True,
        act: Callable[[Array], Array] = jax.nn.relu,
        weight_init=xavier_init,
        # bias_init=xavier_init,
        nn_clip: float = 100.0,
        score_clip: float = 100.0,
        *,
        key: PRNGKey,
    ):
        """
        Args:
            x_dim: size of :math:`x` vector.
            get_score_mu: score of the target density, :math:`\\nabla \\log \\mu(x)`.
            T: duration of diffusion.
            L_max: :math:`L` parameter for positional encoding of :math:`t`.
            emb_dim: dimension for embedding of :math:`t` and :math:`x`.
            embed_width_size: hidden layer dimensionality of embedding networks.
            embed_depth: depth of embedding networks.
            width_size: hidden layer dimensionality for MLPs mapping embeddings to
                :math:`a_\\theta(t, x)` and :math:`b_\\theta(t, x)`.
            depth: depth of MLPs mapping embeddings to :math:`a_\\theta(t, x)`
                and :math:`b_\\theta(t, x)`.
            # init_const_scale: initial scaling of constant network.
            # init_coeff_scale: initial scaling of score coefficient network.
            scalar_coeff_net: if `True`, :math:`b_\\theta(t, x)` will output a scalar
                instead of a vector to be multiplied elementwise with :math:`\\nabla \\log \\mu(x)`.
            act: activation used in all networks.
            weight_init: function for initializing weights.
            bias_init: function for initializing biases.
            key: PRNG key for initializing layers.
        """
        super().__init__()
        self.get_score_mu = get_score_mu
        self.T = T
        self.nn_clip = nn_clip
        self.score_clip = score_clip
        # self.const_scale = init_const_scale
        # self.coeff_scale = init_coeff_scale

        # Build layers
        key_t, key_x, key_const, key_coeff = split(key, 4)
        self.t_pos_encoding = PositionalEncoding(L_max)
        self.t_emb = eqx.nn.MLP(
            2 * L_max,
            emb_dim,
            embed_width_size,
            embed_depth,
            activation=act,
            key=key_t,
        )
        self.x_emb = eqx.nn.MLP(
            x_dim,
            emb_dim,
            embed_width_size,
            embed_depth,
            activation=act,
            key=key_x,
        )
        self.const_net = eqx.nn.MLP(
            2 * emb_dim,
            x_dim,
            width_size,
            depth,
            activation=act,
            key=key_const,
        )
        coeff_net_out_size = 1 if scalar_coeff_net else x_dim
        self.coeff_net = eqx.nn.MLP(
            emb_dim,
            coeff_net_out_size,
            width_size,
            depth,
            activation=act,
            key=key_coeff,
        )

        # Reinitialize weights
        self.t_emb = init_linear_weights(
            self.t_emb, weight_init, key=key_t, scale=0.1,
        )
        self.x_emb = init_linear_weights(
            self.x_emb, weight_init, key=key_x, scale=0.1,
        )
        self.const_net = init_linear_weights(
            self.const_net, weight_init, key=key_const, scale=0.1,
        )
        self.coeff_net = init_linear_weights(
            self.coeff_net, weight_init, key=key_coeff, scale=0.1,
        )

        # Initialize last layers to zero
        self.const_net = init_linear_weights(
            self.const_net, zero_init, key=key_const, scale=0.0,
        )

    def __call__(self, t: Array, x: Array) -> Array:
        t_emb = t # / self.T - 0.5
        t_emb = self.t_pos_encoding(t_emb)
        t_emb = self.t_emb(t_emb)

        # Normalize to Gaussian sample for uncontrolled process
        x_norm = x# / jnp.sqrt(self.T)
        x_emb = self.x_emb(x_norm)
        tx_emb = jnp.concatenate((t_emb, x_emb))

        const = self.const_net(tx_emb)
        const = jnp.clip(const, -self.nn_clip, self.nn_clip)

        coeff = self.coeff_net(t_emb)  # Coefficient network now only depends on time
        
        score = self.get_score_mu(t, x)
        score = jnp.clip(score, -self.score_clip, self.score_clip)
 
        # return self.const_scale * const + self.coeff_scale * coeff * score
        return const - coeff * score  # Changed to subtraction to match t_net_grad form


class FourierControlNet(eqx.Module):
    """
    Affine transformation of score parametrized by two neural networks, using Fourier
    features for time embedding, and an architecture inspired by `path integral sampler paper`
    and `FourierMLP`.
    This is of the form :math:`a_\\theta(t, x) + b_\\theta(t, x) \\nabla \\log \\mu(x)`,
    with the output initialized to zero using an overall learn multiplicative factor.
    """

    get_score_mu: Callable[[Array, Array], Array] = eqx.static_field()
    T: float = eqx.static_field()
    timestep_coeff: Array = eqx.static_field()  # For Fourier features
    nn_clip: float = eqx.static_field()
    score_clip: float = eqx.static_field()

    # Learnable parameters (implicitly via MLPs, and explicitly for timestep_phase)
    timestep_phase: Array  # For Fourier features
    time_fourier_mlp: eqx.nn.MLP
    x_emb: eqx.nn.MLP
    const_net: eqx.nn.MLP
    coeff_net: eqx.nn.MLP

    def __init__(
        self,
        x_dim: int,
        get_score_mu: Callable[[Array, Array], Array],
        T: float = 1.0,
        channels: int = 128,  # For Fourier features, replaces L_max
        emb_dim: int = 64,
        embed_width_size: int = 64,
        embed_depth: int = 2,  # Number of hidden layers in embedding MLPs
        width_size: int = 64,
        depth: int = 3,  # Number of hidden layers in main const/coeff MLPs
        scalar_coeff_net: bool = True,
        act: Callable[[Array], Array] = jax.nn.relu,
        weight_init=xavier_init,
        # bias_init=xavier_init, # Not used directly by init_linear_weights, assumed handled by weight_init or default
        nn_clip: float = 100.0,
        score_clip: float = 100.0,
        *,
        key: PRNGKey,
    ):
        """
        Args:
            x_dim: size of :math:`x` vector.
            get_score_mu: score of the target density, :math:`\\nabla \\log \\mu(x)`.
            T: duration of diffusion.
            channels: number of Fourier frequency channels for time embedding.
            emb_dim: dimension for embedding of :math:`t` (post-Fourier MLP) and :math:`x`.
            embed_width_size: hidden layer dimensionality of embedding networks.
            embed_depth: depth (number of hidden layers) of embedding networks.
            width_size: hidden layer dimensionality for MLPs mapping embeddings to
                :math:`a_\\theta(t, x)` and :math:`b_\\theta(t, x)`.
            depth: depth (number of hidden layers) of MLPs mapping embeddings to :math:`a_\\theta(t, x)`
                and :math:`b_\\theta(t, x)`.
            scalar_coeff_net: if `True`, :math:`b_\\theta(t, x)` will output a scalar
                instead of a vector to be multiplied elementwise with :math:`\\nabla \\log \\mu(x)`.
            act: activation used in all networks.
            weight_init: function for initializing weights.
            nn_clip: clipping value for network outputs.
            score_clip: clipping value for the score of mu.
            key: PRNG key for initializing layers.
        """
        super().__init__()
        self.get_score_mu = get_score_mu
        self.T = T
        self.nn_clip = nn_clip
        self.score_clip = score_clip

        key_phase, key_t_fourier, key_x, key_const, key_coeff = split(key, 5)

        # Fourier features parameters
        self.timestep_coeff = jnp.linspace(start=0.1, end=100.0, num=channels)[
            None, :
        ]  # Shape (1, channels)
        self.timestep_phase = jax.random.normal(
            key_phase, (1, channels)
        )  # Shape (1, channels)

        # Time embedding MLP (processes Fourier features)
        # Input: 2 * channels (sin and cos), Output: emb_dim
        self.time_fourier_mlp = eqx.nn.MLP(
            in_size=2 * channels,
            out_size=emb_dim,
            width_size=embed_width_size,
            depth=embed_depth,
            activation=act,
            key=key_t_fourier,
        )

        # X embedding MLP
        self.x_emb = eqx.nn.MLP(
            in_size=x_dim,
            out_size=emb_dim,
            width_size=embed_width_size,
            depth=embed_depth,
            activation=act,
            key=key_x,
        )

        # Constant part of the control
        self.const_net = eqx.nn.MLP(
            in_size=emb_dim + emb_dim,  # Concatenated time_emb and x_emb
            out_size=x_dim,
            width_size=width_size,
            depth=depth,
            activation=act,
            key=key_const,
        )

        # Coefficient part of the control (multiplies score_mu)
        coeff_net_out_size = 1 if scalar_coeff_net else x_dim
        self.coeff_net = eqx.nn.MLP(
            in_size=emb_dim,  # Only time embedding
            out_size=coeff_net_out_size,
            width_size=width_size,
            depth=depth,
            activation=act,
            key=key_coeff,
        )

        # Reinitialize weights (as in original ControlNet)
        self.time_fourier_mlp = init_linear_weights(
            self.time_fourier_mlp, weight_init, key=key_t_fourier, scale=0.1,
        )
        self.x_emb = init_linear_weights(
            self.x_emb, weight_init, key=key_x, scale=0.1,
        )
        self.const_net = init_linear_weights(
            self.const_net, weight_init, key=key_const, scale=0.1,
        )
        self.coeff_net = init_linear_weights(
            self.coeff_net, weight_init, key=key_coeff, scale=0.1,
        )

        # Initialize last layers of const_net and coeff_net to zero
        # (assuming init_linear_weights with zero_init and scale=0.0 targets the last layer)
        self.const_net = init_linear_weights(
            self.const_net, zero_init, key=key_const, scale=0.0,
        )
        self.coeff_net = init_linear_weights(
            self.coeff_net, zero_init, key=key_coeff, scale=0.0,
        )

    def __call__(self, t: Array, x: Array) -> Array:
        # Ensure t is float and correctly shaped for broadcasting, e.g., (batch_size, 1)
        if t.ndim == 0: # scalar t
            t_arr = jnp.array([t], dtype=jnp.float32)
        elif t.ndim == 1: # (batch_size,)
            t_arr = t.astype(jnp.float32)
        else: # Should not happen if t is scalar or (batch_size,)
            t_arr = t.astype(jnp.float32)
        
        t_normalized = t_arr / self.T  # Normalize t to [0, 1]
        if t_normalized.ndim == 1:
            t_normalized = t_normalized[:, None] # Ensure (batch_size, 1) for broadcasting

        # Compute Fourier features for time
        sin_features = jnp.sin(
            (self.timestep_coeff * t_normalized) + self.timestep_phase
        )
        cos_features = jnp.cos(
            (self.timestep_coeff * t_normalized) + self.timestep_phase
        )
        # Concatenate sin and cos features: shape (batch_size, 2 * channels)
        fourier_t = jnp.concatenate((sin_features, cos_features), axis=-1)

        t_emb = self.time_fourier_mlp(fourier_t)

        # X embedding
        # x_norm = x # Original ControlNet had x / jnp.sqrt(self.T) commented out
        x_e = self.x_emb(x)  # Use x directly

        # Concatenate time and x embeddings for const_net
        tx_emb = jnp.concatenate((t_emb, x_e), axis=-1)

        const = self.const_net(tx_emb)
        const = jnp.clip(const, -self.nn_clip, self.nn_clip)

        # Coeff_net only depends on time embedding
        coeff = self.coeff_net(t_emb)
        # coeff = jnp.clip(coeff, -self.nn_clip, self.nn_clip) # Clipping coeff? Original doesn't.

        score = self.get_score_mu(t_arr, x) # Use original t for score function
        score = jnp.clip(score, -self.score_clip, self.score_clip)

        return const - coeff * score
