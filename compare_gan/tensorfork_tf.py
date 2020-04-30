from . import tensorfork
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import importlib
import threading
import gin
import gin.tf.external_configurables

class State(threading.local):
  def __init__(self):
    pass

if 'state' in globals():
  state = globals()['state']
else:
  state = State()
  state.pinned_sessions = []
  state.session_timeout_in_ms = 1200000
  state.eval_lightweight_timeout = 20000
  state.load_lightweight_timeout = 20000
  #state.initialize_timeout = 60000
  state.initialize_timeout = 20*60000
  state.context_load_timeout = 20000
  state.ensure_on_init = True
  state.release_trainer_sema = True
  state.tpu_init_timeout = 10000
  state.summary_log_timeout = 20000
  state.use_global_data_sampler = True
  state.shuffle_cycles = True
  state.vars = {}

def eval_lightweight(variable, session=None, timeout_in_ms=None):
  session = session or tf.get_default_session()
  if session is None:
    return None
  if timeout_in_ms is None:
    timeout_in_ms = state.eval_lightweight_timeout
  return eval(variable, session=session, timeout_in_ms=state.eval_lightweight_timeout)

def load_lightweight(variable, value, session=None, timeout_in_ms=None):
  session = session or tf.get_default_session()
  if session is None:
    return
  if timeout_in_ms is None:
    timeout_in_ms = state.load_lightweight_timeout
  return load(variable, value, session=session, timeout_in_ms=timeout_in_ms)

from tensorflow.core.protobuf import config_pb2

def load(variable, value, session=None, timeout_in_ms=None):
  session = session or tf.get_default_session()
  if session is None:
    return
  ops = variable.initializer
  vals = dict([(variable.initializer.inputs[1], value)])
  #for x, (k, v) in zip(variables, vals.items()):
  #  print(x.name, x.shape.as_list(), k, v.shape)
  options = None
  if timeout_in_ms:
    options=config_pb2.RunOptions(timeout_in_ms=timeout_in_ms)
  return session.run(ops, vals, options=options)

def eval(variable, session=None, timeout_in_ms=None):
  session = session or tf.get_default_session()
  if session is None:
    return None
  options = None
  if timeout_in_ms:
    options=config_pb2.RunOptions(timeout_in_ms=timeout_in_ms)
  return session.run(variable, options=options)

def grab_values(variables, reader, reshape=False):
  for variable in variables:
    name = variable_name(variable).split(':')[0]
    value = reader.get_tensor(name)
    value = truncate_value(variable, value, reshape=reshape)
    yield variable, value

def assign_values(variables, values, session=None, timeout_in_ms=60000):
  session = session or tf.get_default_session()
  ops = [x.initializer for x in variables]
  vals = dict([(x.initializer.inputs[1], value) for x, value in zip(variables, values)]) # TODO: bfloat16 support
  #for x, (k, v) in zip(variables, vals.items()):
  #  print(x.name, x.shape.as_list(), k, v.shape)
  options = None
  if timeout_in_ms:
    options=config_pb2.RunOptions(timeout_in_ms=timeout_in_ms)
  session.run(ops, vals, options=options)

import os
import re

def fqn(name, scope=None):
  if scope is None or len(scope) <= 0:
    scope = tf.get_variable_scope().name
  name = '_'.join(re.split('[^a-z0-9]', os.path.join(scope, name)))
  return name

def absolute_name_scope(scope):
  return tf.name_scope(scope + "/")

def absolute_variable_scope(scope=None, **kwargs):
  if scope is None:
    scope = tf.get_variable_scope().name
  return tf.variable_scope(tf.VariableScope(name=scope, **kwargs), auxiliary_name_scope=False)

def scoped_name(name=None, scope=None):
  if scope is None:
    scope = tf.get_variable_scope().name
  if name is None:
    name = ''
  name = name.split(':')[0]
  return os.path.join(scope, name)

def get_var(name, default_value=None, update=False, scope=None, dtype=None, shape=(), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES], use_resource=True, **kws):
  if scope is not None:
    try:
      return get_var(os.path.join(scope, name), default_value=None, update=update, scope=None, dtype=dtype, shape=shape, trainable=trainable, collections=collections, use_resource=use_resource, **kws)
    except ValueError:
      scope = None
  knob = name if '.' in name else fqn(name)
  #knob = knob.split('/')[-1]
  if not tensorfork.has_knob(knob):
    update = True
  if update and default_value is None:
    value = tensorfork.knobs(knob)
    if value is None:
      raise ValueError("No such knob: {}".format(knob))
  else:
    value = tensorfork.knobs(knob, default_value, update)
  if dtype is None:
    if hasattr(value, 'dtype'):
      dtype = value.dtype
    elif isinstance(value, float):
      dtype = tf.float32
    elif isinstance(value, int):
      dtype = tf.int32
    else:
      raise ValueError("Unknown type")
  with absolute_variable_scope(reuse=tf.AUTO_REUSE):
    initializer = kws.pop('initializer', None)
    if initializer is None:
      initializer = tf.constant_initializer(value) if value is not None else None
    logging.info("tf.get_variable(name=%s, shape=%s, initializer=%s, default_value=%s, value=%s)", name, shape, initializer, default_value, value)
    var = tf.get_variable(name=name, dtype=dtype, shape=shape, trainable=trainable, collections=collections, use_resource=use_resource, initializer=initializer, **kws)
  state.vars[var.name.split(':')[0]] = {'knob': knob, 'variable': var}
  if update or default_value is None:
    load_lightweight(var, value)
  v = eval_lightweight(var)
  if v is not None and v != value:
    tensorfork.knobs(name, v, update=True)
  return var

from absl import logging

def compare_values(a, b, eps=1e-5):
  if abs(a - b) < eps:
    return 0
  if a > b:
    return -1
  return 1

def update_vars(name=None, skip_unknown=False):
  tensorfork.reload(name=name, skip_unknown=skip_unknown)
  for entry in state.vars.values():
    knob = entry['knob']
    variable = entry['variable']
    vm_value = tensorfork.knobs(knob)
    logging.info('Knob CPU value: %s %s', vm_value, knob)
    tf_value = eval_lightweight(variable)
    logging.info('Knob TPU value: %s %s', tf_value, variable)
    if compare_values(vm_value, tf_value) != 0:
      logging.info("Setting knob %s to %s (was %s)", knob, vm_value, tf_value)
      load_lightweight(variable, vm_value)

if __name__ == "__main__":
  logging.set_verbosity(0)
  tensorfork.tensorfork.tf = importlib.__import__('tensorfork_tf')
  #sess = tf1.InteractiveSession()
  tensorfork.main()

