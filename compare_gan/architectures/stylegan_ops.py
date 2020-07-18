import gin

import numpy as np
import tensorflow as tf

from typing import Any, List, Tuple, Union

TfExpression = Union[tf.Tensor, tf.Variable, tf.Operation]
"""A type that represents a valid Tensorflow expression."""

TfExpressionEx = Union[TfExpression, int, float, np.ndarray]
"""A type that can be converted to a valid Tensorflow expression."""


def is_tf_expression(x: Any) -> bool:
    """Check whether the input is a valid Tensorflow expression, i.e., Tensorflow Tensor, Variable, or Operation."""
    return isinstance(x, (tf.Tensor, tf.Variable, tf.Operation))


def lerp(a: TfExpressionEx, b: TfExpressionEx, t: TfExpressionEx) -> TfExpressionEx:
    """Linear interpolation."""
    with tf.name_scope("Lerp"):
        return a + (b - a) * t


# Util classes
# ------------------------------------------------------------------------------------------

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]




activation_funcs = {
    'linear':   EasyDict(func=lambda x, **_:        x,                          def_alpha=None, def_gain=1.0,           cuda_idx=1, ref='y', zero_2nd_grad=True),
    'relu':     EasyDict(func=lambda x, **_:        tf.nn.relu(x),              def_alpha=None, def_gain=np.sqrt(2),    cuda_idx=2, ref='y', zero_2nd_grad=True),
    'lrelu':    EasyDict(func=lambda x, alpha, **_: tf.nn.leaky_relu(x, alpha), def_alpha=0.2,  def_gain=np.sqrt(2),    cuda_idx=3, ref='y', zero_2nd_grad=True),
    'tanh':     EasyDict(func=lambda x, **_:        tf.nn.tanh(x),              def_alpha=None, def_gain=1.0,           cuda_idx=4, ref='y', zero_2nd_grad=False),
    'sigmoid':  EasyDict(func=lambda x, **_:        tf.nn.sigmoid(x),           def_alpha=None, def_gain=1.0,           cuda_idx=5, ref='y', zero_2nd_grad=False),
    'elu':      EasyDict(func=lambda x, **_:        tf.nn.elu(x),               def_alpha=None, def_gain=1.0,           cuda_idx=6, ref='y', zero_2nd_grad=False),
    'selu':     EasyDict(func=lambda x, **_:        tf.nn.selu(x),              def_alpha=None, def_gain=1.0,           cuda_idx=7, ref='y', zero_2nd_grad=False),
    'softplus': EasyDict(func=lambda x, **_:        tf.nn.softplus(x),          def_alpha=None, def_gain=1.0,           cuda_idx=8, ref='y', zero_2nd_grad=False),
    'swish':    EasyDict(func=lambda x, **_:        tf.nn.sigmoid(x) * x,       def_alpha=None, def_gain=np.sqrt(2),    cuda_idx=9, ref='x', zero_2nd_grad=False),
}


def fused_bias_act(x, b=None, axis=1, act='linear', alpha=None, gain=None, impl='ref'):
    r"""Fused bias and activation function.

    Adds bias `b` to activation tensor `x`, evaluates activation function `act`,
    and scales the result by `gain`. Each of the steps is optional. In most cases,
    the fused op is considerably more efficient than performing the same calculation
    using standard TensorFlow ops. It supports first and second order gradients,
    but not third order gradients.

    Args:
        x:      Input activation tensor. Can have any shape, but if `b` is defined, the
                dimension corresponding to `axis`, as well as the rank, must be known.
        b:      Bias vector, or `None` to disable. Must be a 1D tensor of the same type
                as `x`. The shape must be known, and it must match the dimension of `x`
                corresponding to `axis`.
        axis:   The dimension in `x` corresponding to the elements of `b`.
                The value of `axis` is ignored if `b` is not specified.
        act:    Name of the activation function to evaluate, or `"linear"` to disable.
                Can be e.g. `"relu"`, `"lrelu"`, `"tanh"`, `"sigmoid"`, `"swish"`, etc.
                See `activation_funcs` for a full list. `None` is not allowed.
        alpha:  Shape parameter for the activation function, or `None` to use the default.
        gain:   Scaling factor for the output tensor, or `None` to use default.
                See `activation_funcs` for the default scaling of each activation function.
                If unsure, consider specifying `1.0`.
        impl:   Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

    Returns:
        Tensor of the same shape and datatype as `x`.
    """

    impl_dict = {
        'ref':  _fused_bias_act_ref,
        #'cuda': _fused_bias_act_ref if 'TPU_NAME' in os.environ or 'NO_NVCC' in os.environ else _fused_bias_act_cuda,
    }
    return impl_dict[impl](x=x, b=b, axis=axis, act=act, alpha=alpha, gain=gain)

#----------------------------------------------------------------------------

def _fused_bias_act_ref(x, b, axis, act, alpha, gain):
    """Slow reference implementation of `fused_bias_act()` using standard TensorFlow ops."""

    # Validate arguments.
    x = tf.convert_to_tensor(x)
    b = tf.convert_to_tensor(b) if b is not None else tf.constant([], dtype=x.dtype)
    act_spec = activation_funcs[act]
    assert b.shape.rank == 1 and (b.shape[0] == 0 or b.shape[0] == x.shape[axis])
    assert b.shape[0] == 0 or 0 <= axis < x.shape.rank
    if alpha is None:
        alpha = act_spec.def_alpha
    if gain is None:
        gain = act_spec.def_gain

    # Add bias.
    if b.shape[0] != 0:
        x += tf.reshape(b, [-1 if i == axis else 1 for i in range(x.shape.rank)])

    # Evaluate activation function.
    x = act_spec.func(x, alpha=alpha)

    # Scale by gain.
    if gain != 1:
        x *= gain
    return x


#----------------------------------------------------------------------------
# Get/create weight tensor for a convolution or fully-connected layer.

def get_weight(shape, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable.
    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable(weight_var, shape=shape, initializer=init, use_resource=True) * runtime_coef

#----------------------------------------------------------------------------
# Fully-connected layer.

from compare_gan.architectures import arch_ops as ops

def dense_layer(x, fmaps, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)
    w = tf.cast(w, x.dtype)
    w = ops.graph_spectral_norm(w)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Apply bias and activation func.

def apply_bias_act(x, act='linear', alpha=None, gain=None, lrmul=1, bias_var='bias'):
    b = tf.get_variable(bias_var, shape=[x.shape[1]], initializer=tf.initializers.zeros(), use_resource=True) * lrmul
    return fused_bias_act(x, b=tf.cast(b, x.dtype), act=act, alpha=alpha, gain=gain)

#----------------------------------------------------------------------------
# Mapping network.
# Transforms the input latent code (z) to the disentangled latent code (w).
# Used in configs B-F (Table 1).

@gin.configurable
def G_mapping(
    latents_in,                             # First input: Latent vectors (Z) [minibatch, latent_size].
    labels_in,                              # Second input: Conditioning labels [minibatch, label_size].
    latent_size             = 512,          # Latent vector (Z) dimensionality.
    label_size              = 0,            # Label dimensionality, 0 if no labels.
    dlatent_size            = 512,          # Disentangled latent (W) dimensionality.
    dlatent_broadcast       = None,         # Output disentangled latent (W) as [minibatch, dlatent_size] or [minibatch, dlatent_broadcast, dlatent_size].
    mapping_layers          = 8,            # Number of mapping layers.
    mapping_fmaps           = 512,          # Number of activations in the mapping layers.
    mapping_lrmul           = 0.01,         # Learning rate multiplier for the mapping layers.
    mapping_nonlinearity    = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    normalize_latents       = True,         # Normalize latent vectors (Z) before feeding them to the mapping layers?
    normalize_function      = 'mean',       # 'mean' or 'sum'
    dtype                   = 'float32',    # Data type to use for activations and outputs.
    **_kwargs):                             # Ignore unrecognized keyword args.

    if normalize_function not in ['mean', 'sum']:
        raise ValueError("Unknown normalize_function {!r}".format(normalize_function))
    normalize_function = tf.reduce_sum if normalize_function == 'sum' else tf.reduce_mean

    act = mapping_nonlinearity

    # Inputs.
    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    latents_in = tf.cast(latents_in, dtype)
    labels_in = tf.cast(labels_in, dtype)
    x = latents_in

    # Embed labels and concatenate them with latents.
    if label_size:
        with tf.variable_scope('LabelConcat'):
            w = tf.get_variable('weight', shape=[label_size, latent_size], initializer=tf.initializers.random_normal(), use_resource=True)
            y = tf.matmul(labels_in, tf.cast(w, dtype))
            x = tf.concat([x, y], axis=1)

    # Normalize latents.
    if normalize_latents:
        with tf.variable_scope('Normalize'):
            x *= tf.rsqrt(normalize_function(tf.square(x), axis=1, keepdims=True) + 1e-8)

    # Mapping layers.
    for layer_idx in range(mapping_layers):
        with tf.variable_scope('Dense%d' % layer_idx):
            fmaps = dlatent_size if layer_idx == mapping_layers - 1 else mapping_fmaps
            x = apply_bias_act(dense_layer(x, fmaps=fmaps, lrmul=mapping_lrmul), act=act, lrmul=mapping_lrmul)

    # Broadcast.
    if dlatent_broadcast is not None:
        with tf.variable_scope('Broadcast'):
            x = tf.tile(x[:, np.newaxis], [1, dlatent_broadcast, 1])

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return tf.identity(x, name='dlatents_out')


#----------------------------------------------------------------------------
# Main generator network.
# Composed of two sub-networks (mapping and synthesis) that are defined below.
# Used in configs B-F (Table 1).

@gin.configurable
def G_main(
    num_layers,
    latents_in,                                         # First input: Latent vectors (Z) [minibatch, latent_size].
    labels_in               = None,                     # Second input: Conditioning labels [minibatch, label_size].
    truncation_psi          = 0.5,                      # Style strength multiplier for the truncation trick. None = disable.
    truncation_cutoff       = None,                     # Number of layers for which to apply the truncation trick. None = disable.
    truncation_psi_val      = None,                     # Value for truncation_psi to use during validation.
    truncation_cutoff_val   = None,                     # Value for truncation_cutoff to use during validation.
    dlatent_avg_beta        = 0.995,                    # Decay for tracking the moving average of W during training. None = disable.
    style_mixing_prob       = 0.9,                      # Probability of mixing styles during training. None = disable.
    is_training             = False,                    # Network is under training? Enables and disables specific features.
    is_validation           = False,                    # Network is under validation? Chooses which value to use for truncation_psi.
    #return_dlatents         = False,                    # Return dlatents in addition to the images?
    #is_template_graph       = False,                    # True = template graph constructed by the Network class, False = actual evaluation.
    #components              = dnnlib.EasyDict(),        # Container for sub-networks. Retained between calls.
    mapping_func            = 'G_mapping',              # Build func name for the mapping network.
    #synthesis_func          = 'G_synthesis_stylegan2',  # Build func name for the synthesis network.
    dlatent_size            = 512,
    **kwargs):                                          # Arguments for sub-networks (mapping and synthesis).

    # Validate arguments.
    assert not is_training or not is_validation
    #assert isinstance(components, dnnlib.EasyDict)
    if is_validation:
        truncation_psi = truncation_psi_val
        truncation_cutoff = truncation_cutoff_val
    if is_training or (truncation_psi is not None and not is_tf_expression(truncation_psi) and truncation_psi == 1):
        truncation_psi = None
    if is_training:
        truncation_cutoff = None
    if not is_training or (dlatent_avg_beta is not None and not is_tf_expression(dlatent_avg_beta) and dlatent_avg_beta == 1):
        dlatent_avg_beta = None
    if not is_training or (style_mixing_prob is not None and not is_tf_expression(style_mixing_prob) and style_mixing_prob <= 0):
        style_mixing_prob = None

    if labels_in is None:
        labels_in = tf.zeros([tf.shape(latents_in)[0], 0], dtype=tf.float32)

    # Setup components.
    # if 'synthesis' not in components:
    #     components.synthesis = tflib.Network('G_synthesis', func_name=globals()[synthesis_func], **kwargs)
    # num_layers = components.synthesis.input_shape[1]
    # dlatent_size = components.synthesis.input_shape[2]
    # if 'mapping' not in components:
    #     components.mapping = tflib.Network('G_mapping', func_name=globals()[mapping_func], dlatent_broadcast=num_layers, **kwargs)
    assert mapping_func == 'G_mapping'
    mapping_func = G_mapping

    # Setup variables.
    #lod_in = tf.get_variable('lod', initializer=np.float32(0), trainable=False, use_resource=True)
    lod_in = tf.constant(np.float32(0))
    dlatent_avg = tf.get_variable('dlatent_avg', shape=[dlatent_size], initializer=tf.initializers.zeros(), trainable=False, use_resource=True)

    # Evaluate mapping network.
    #dlatents = components.mapping.get_output_for(latents_in, labels_in, is_training=is_training, **kwargs)
    dlatents = mapping_func(latents_in, labels_in, is_training=is_training, dlatent_broadcast=num_layers, **kwargs)
    dlatents = tf.cast(dlatents, tf.float32)

    # Update moving average of W.
    if dlatent_avg_beta is not None:
        with tf.variable_scope('DlatentAvg'):
            batch_avg = tf.reduce_mean(dlatents[:, 0], axis=0)
            update_op = tf.assign(dlatent_avg, lerp(batch_avg, dlatent_avg, dlatent_avg_beta))
            with tf.control_dependencies([update_op]):
                dlatents = tf.identity(dlatents)

    # Perform style mixing regularization.
    if style_mixing_prob is not None:
        with tf.variable_scope('StyleMix'):
            latents2 = tf.random_normal(tf.shape(latents_in))
            #dlatents2 = components.mapping.get_output_for(latents2, labels_in, is_training=is_training, **kwargs)
            dlatents2 = mapping_func(latents2, labels_in, is_training=is_training, dlatent_broadcast=num_layers, **kwargs)
            dlatents2 = tf.cast(dlatents2, tf.float32)
            layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
            cur_layers = num_layers - tf.cast(lod_in, tf.int32) * 2
            mixing_cutoff = tf.cond(
                tf.random_uniform([], 0.0, 1.0) < style_mixing_prob,
                lambda: tf.random_uniform([], 1, cur_layers, dtype=tf.int32),
                lambda: cur_layers)
            dlatents = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(dlatents)), dlatents, dlatents2)

    # Apply truncation trick.
    if truncation_psi is not None:
        with tf.variable_scope('Truncation'):
            layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
            layer_psi = np.ones(layer_idx.shape, dtype=np.float32)
            if truncation_cutoff is None:
                layer_psi *= truncation_psi
            else:
                layer_psi = tf.where(layer_idx < truncation_cutoff, layer_psi * truncation_psi, layer_psi)
            dlatents = lerp(dlatent_avg, dlatents, layer_psi)

    return dlatents

