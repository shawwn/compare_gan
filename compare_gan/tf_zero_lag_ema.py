import tensorflow as tf
import gin

def iszero(x):
  return tf.equal(tf.cast(0, x.dtype), tf.reduce_sum(tf.abs(x)))


def val(x, init=None):
  if hasattr(x, 'read_value'):
    result = x.read_value()
  else:
    result = tf.identity(x, name='read')
  if init is not None:
    return tf.cond(iszero(result),
        lambda: init,
        lambda: result)
  else:
    return result


def shapelist(x):
  if hasattr(x, 'shape'):
    x = x.shape
  return x.as_list()


def globalvar(name, **kws):
  shape = kws.pop('shape')
  initializer = kws.pop('initializer', None)
  if initializer is None:
    initializer = tf.initializers.zeros
  collections = kws.pop('collections', ['variables'])
  trainable = kws.pop('trainable', True)
  use_resource = kws.pop('use_resource', True)
  dtype = kws.pop('dtype', tf.float32)
  return tf.get_variable(name, dtype=dtype, initializer=initializer, shape=shape, collections=collections, use_resource=use_resource, trainable=trainable, **kws)


def localvar(name, **kws):
  collections = kws.pop('collections', ['local_variables'])
  trainable = kws.pop('trainable', False)
  use_resource = kws.pop('use_resource', True)
  return globalvar(name, **kws, collections=collections, trainable=trainable, use_resource=use_resource)

def getvar(name, **kws):
  with tf.variable_scope("", reuse=tf.compat.v1.AUTO_REUSE, use_resource=True):
    return localvar(name, **kws)


@gin.configurable(whitelist=['gain_limit', 'gain_precision', 'length_multiplier'])
def tf_zero_lag_ema(close, ec_var, ema_var, *, length=20.0, length_multiplier=1.0, gain_limit=500.0, gain_precision=100.0, dtype=tf.float32):
  initializer_ops = []
  alpha = 2.0 / ((length * length_multiplier) + 1.0)
  ec_1 = val(ec_var, close)
  ema_1 = val(ema_var, close)
  ema = alpha * close + (1.0 - alpha) * ema_1
  grid = tf.range(-gain_limit, gain_limit+1.0, delta=1.0, dtype=dtype)
  gain = grid / gain_precision
  def fn(gain):
    ec = alpha * (ema + gain*(close - ec_1)) + (1.0 - alpha) * ec_1
    error = tf.linalg.norm(close - ec)
    return error
  op = tf.vectorized_map(fn, gain)
  least_error_idx = tf.argmin(op)
  best_gain = gain[least_error_idx]
  least_error = op[least_error_idx]
  ec = alpha * (ema + best_gain*(close - ec_1)) + (1.0 - alpha) * ec_1
  read_ops = [ec, ema]
  update_ops = []
  with tf.control_dependencies([ec_var.assign(ec, read_value=False)]):
    with tf.control_dependencies([ema_var.assign(ema, read_value=False)]):
      update_ops = [val(ec), val(ema)]
  return update_ops, read_ops


