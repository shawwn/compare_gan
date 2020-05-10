
import traceback
import importlib

def api_op(f):
  if hasattr(api, f.__name__):
    logging.warning("Redefining %s", f.__name__)
  setattr(api, f.__name__, f)
  module = importlib.__import__('compare_gan.tensorfork_tf')
  setattr(module, f.__name__, f)
  #module.__globals__[f.__name__] = f
  globals()[f.__name__] = f
  return f

api_op = api_op(api_op)


@api_op
def restore(step, session=None, override_step=None):
  session = session or tf.get_default_session()
  ckpt = os.path.join(os.environ['MODEL_DIR'], 'model.ckpt-{}'.format(step))
  with api.with_session(session):
    with api.with_graph(session.graph):
      api.rollback(ckpt, var_list=api.unique_variables('trains') + vals.gan._disc_optimizer_opt.variables() + vals.gan._gen_optimizer_opt.variables())
      # TODO: figure out how to roll back the hook timers.
      # all_vars = unique_variables()
      # step_vars = [x for x in all_vars if api.variable_name(x).startswith('global_step')]
      # set_step = override_step if override_step is not None else step
      # for v in step_vars:
      #   logging.info('Setting %s to %s', v, set_step)
      #   load_lightweight(v, set_step)


@api_op
def heartbeat(session=None):
  logging.info('Heartbeat')
  session = session or tf.get_default_session()
  if session is not None and session not in state.pinned_sessions:
    state.pinned_sessions.append(session)
  if os.path.exists('debug_break.txt'):
    api.rm('debug_break.txt')
    api.break_session(session=session)
  try:
    if os.path.exists('restore_step.txt'):
      with open('restore_step.txt') as f:
        chars = f.read()
      api.rm('restore_step.txt')
      parts = chars.strip().split()
      if len(parts) == 1:
        api.restore(int(parts[0]), session=session)
      elif len(parts) == 2:
        api.restore(int(parts[0]), session=session, override_step=int(parts[1]))
      else:
        logging.warn("Invalid restore_step %r", chars)
  except:
    traceback.print_exc()

