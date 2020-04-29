from tensorfork import EasyDict
import tensorfork
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import importlib
import threading
import gin

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

def eval_lightweight(variable, session, timeout_in_ms=None):
  if timeout_in_ms is None:
    timeout_in_ms = state.eval_lightweight_timeout
  return state.eval(variable, session=session, timeout_in_ms=state.eval_lightweight_timeout)

def load_lightweight(variable, value, session, timeout_in_ms=None):
  if timeout_in_ms is None:
    timeout_in_ms = state.load_lightweight_timeout
  return state.load(variable, value, session=session, timeout_in_ms=timeout_in_ms)

from tensorflow.core.protobuf import config_pb2

def load(variable, value, session=None, timeout_in_ms=None):
  session = session or tf.get_default_session()
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
  options = None
  if timeout_in_ms:
    options=config_pb2.RunOptions(timeout_in_ms=timeout_in_ms)
  return session.run(variable, options=options)

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

def get_var(name, default_value=None, update=False, dtype=None, shape=(), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES], use_resource=True, **kws):
  knob = name if '.' in name else fqn(name)
  knob = knob.split('/')[-1]
  if not tensorfork.has_knob(knob):
    update = True
  if update and default_value is None:
    value = tensorfork.knobs(knob)
    if value is None:
      raise ValueError("No such knob: {}".format(knob))
  else:
    value = tensorfork.knobs(knob, default_value, update)
  if dtype is None:
    if isinstance(value, float):
      dtype = tf.float32
    elif isinstance(value, int):
      dtype = tf.int32
    else:
      raise ValueError("Unknown type")
  with absolute_variable_scope(reuse=tf.AUTO_REUSE):
    var = tf.get_variable(name=name, dtype=dtype, shape=shape, trainable=trainable, collections=collections, use_resource=use_resource, **kws)
  state.vars[var.name.split(':')[0]] = var
  if update or default_value is None:
    load(var, value)
  v = eval(var)
  if v != value:
    tensorfork.knobs(name, v, update=True)
  return var

if __name__ == "__main__":
  tensorfork.tensorfork.tf = importlib.__import__('tensorfork_tf')
  sess = tf1.InteractiveSession()
  tensorfork.main()

