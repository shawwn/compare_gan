import tensorflow as tf
import gin

@gin.configurable(whitelist=['gain_limit', 'gain_precision', 'length_multiplier'])
def tf_zero_lag_ema(close, ec_var, ema_var, *, length=20.0, length_multiplier=1.0, gain_limit=500.0, gain_precision=100.0, dtype=tf.float32):
  alpha = 2.0 / ((length * length_multiplier) + 1.0)
  ec_1 = val(ec_var, close)
  ema_1 = val(ema_var, close)
  ema = alpha * close + (1.0 - alpha) * ema_1
  grid = tf.range(-gain_limit, gain_limit+1.0, delta=1.0, dtype=dtype)
  gains = grid / gain_precision
  def fn(gain):
    ec = alpha * (ema + gain*(close - ec_1)) + (1.0 - alpha) * ec_1
    error = tf.linalg.norm(close - ec)
    return error
  errors = tf.vectorized_map(fn, gains)
  least_error_idx = tf.argmin(errors)
  best_gain = gains[least_error_idx]
  least_error = errors[least_error_idx]
  ec = alpha * (ema + best_gain*(close - ec_1)) + (1.0 - alpha) * ec_1
  read_ops = [ec, ema]
  update_ops = []
  with tf.control_dependencies([ec_var.assign(ec, read_value=False)]):
    with tf.control_dependencies([ema_var.assign(ema, read_value=False)]):
      update_ops = [val(ec), val(ema)]
  return update_ops, read_ops


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


