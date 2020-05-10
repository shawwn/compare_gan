from pprint import pprint as pp
from github import Github, Issue
from getpass import getpass
import os
import re
import gin
import time

class Namespace():
  pass

if 'tensorfork' not in globals():
  tensorfork = Namespace()

env = {}

def opts(env=env, name='.env'):
  if time.time() - env.get('parsed', 0) < 5.0:
    return env
  if os.path.exists(name):
    with open(name) as f:
      s = f.read()
    for k, v in parse_string(s):
      env[k] = v
  env['parsed'] = time.time()
  return env

def parse_string(s, included=[]):
  if isinstance(s, list):
    s = '\n'.join(s)
  parser = gin.config_parser.ConfigParser(s, gin.config.ParserDelegate(skip_unknown=True))
  for statement in parser:
    if isinstance(statement, gin.config_parser.IncludeStatement):
      if statement.filename in included:
        print('Skipping circular dependency: {}'.format(statement.filename))
      else:
        body = include(statement.filename)
        for k, v in parse_string(body, included.union([statement.filename])):
          yield k, v
    elif isinstance(statement, gin.config_parser.ImportStatement):
      yield statement.module, '@import'
    elif hasattr(statement, 'selector'):
      v = statement.value
      k = statement.arg_name
      if isinstance(k, str) and len(k.strip()) > 0:
        k = '{}.{}'.format(statement.selector, statement.arg_name)
      else:
        k = statement.selector
      k = os.path.join(statement.scope or '', k)
      v = statement.value
      yield k, v
    else:
      raise Exception("Bad statement {}".format(statement))

def repo(name="tensorfork/tensorfork", username=None, password=None):
  if hasattr(tensorfork, 'repo'):
    return tensorfork.repo
  if not hasattr(tensorfork, 'github'):
    if username is None:
      username = opts().get("github.username")
    if password is None:
      password = opts().get("github.password")
    if username is None:
      username = getpass("GitHub username: ")
    if password is None:
      password = getpass("GitHub password (never saved): ")
    tensorfork.github = Github(username, password)
  if not hasattr(tensorfork, 'repos'):
    tensorfork.repos = {}
  name = name.split('#')[0].strip().lower()
  if name not in tensorfork.repos:
    tensorfork.repos[name] = tensorfork.github.get_repo(name)
  return tensorfork.repos[name]

def parse_body(body):
  return re.findall("```gin[\n](.*?\n)```", body.replace('\r', ''), flags=re.DOTALL|re.MULTILINE)

def parse_thing(thing):
  author = thing.user.login
  url = thing.html_url
  json = thing.url
  cfg = parse_body(thing.body)
  
  return ["""# @tensorfork.config
# author: {}
# url: {}
# json: {}
# created: {}
# updated: {}
""".format(author, url, json, thing.created_at, thing.updated_at)] + cfg

def fetch_all(x):
  n = x.totalCount
  i = 0
  page = 0
  while i < n:
    for value in x.get_page(page):
      yield value
      i += 1
    page += 1

def parse_issue(issue):
  cfg = parse_thing(issue)
  for comment in fetch_all(issue.get_comments()):
    cfg.extend(parse_thing(comment))
  return cfg

def repr_value(v):
  if hasattr(v, 'selector'):
    if v.selector == 'gin.macro':
      return v.__repr__()
    else:
      return '@{}'.format(v.selector)
  else:
    return repr(v)

def parse_bindings(s, included=[]):
  if isinstance(s, str):
    s = parse_string(s, included=included)
  for k, v in s:
    if v == '@import':
      setattr(tensorfork, 'g_'+k, __import__(k))
    else:
      yield "{} = {}".format(k, repr_value(v))

def parse_title(title):
  r = {}
  for part in title.split(';'):
    k, v = part.strip().split(':', 1)
    k = k.lower().strip()
    v = v.strip()
    r[k] = v
  return r

def fetch_issue(x):
  if isinstance(x, Issue.Issue):
    return x
  r = repo()
  if isinstance(x, str):
    if '#' in x:
      path, name = x.split('#')
      r = repo(path)
      x = name
  try:
    x = int(x)
  except ValueError:
    pass
  if isinstance(x, int):
    return r.get_issue(number=x)
  if isinstance(x, str):
    for issue in fetch_all(r.get_issues()):
      title = x.lower().strip()
      if parse_title(issue.title).get("run") == title:
        return issue
  raise Exception("Could not fetch run: {}".format(x))

def include(issue):
  cfg = parse_issue(fetch_issue(issue))
  return '\n'.join(cfg)

# https://stackoverflow.com/questions/20656135/python-deep-merge-dictionary-data
def assign(destination, *sources, stack=[]):
    """
    run me with nosetests --with-doctest file.py

    >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
    >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
    >>> assign(a, b) == { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
    True
    """
    for source in sources:
      if source not in stack:
        stack += [source]
        try:
          for key, value in source.items():
            if isinstance(value, dict):
              # get node or create one
              node = destination.setdefault(key, {})
              assign(node, value, stack=stack)
            else:
              #assert isinstance(destination, dict)
              destination[key] = value
        finally:
          stack.pop()
    return destination

def fetch(name, dst=None, included=[]):
  name = name.split(',')
  if len(name) == 1:
    name = name[0]
  if isinstance(name, list):
    r = None
    for x in name.split(','):
      r = fetch(x, dst=r, included=included)
    return r
  dst = dst if dst is not None else {
      'meta': {
        'name': None,
        'tpu': os.environ.get("TPU_NAME", ""),
        'logdir': os.environ.get("MODEL_DIR", ""),
        },
      'cfg': "",
      'gin': {},
      'gin_bindings': [],
      'fetched_at': time.time(),
  }
  if name is not None:
    issue = fetch_issue(name)
    meta = parse_title(issue.title)
    cfg = include(issue)
    included = __builtins__['set'](included).union([name])
    parsed = list(parse_string(cfg, included=included))
    bindings = list(parse_bindings(parsed, included=included))
    dst['name'] = name
    dst['issue'] = issue
    dst['cfg'] = cfg
    dst['meta'] = assign(dst['meta'], meta)
    dst['gin'] = assign(dst['gin'], dict(parsed))
    dst['gin_bindings'] = bindings
  return dst

def get_name(name=None):
  if name is None:
    if hasattr(tensorfork, 'run'):
      return tensorfork.run['name']
  return name

def get(name=None):
  if not hasattr(tensorfork, 'run') and name is None:
    name = os.environ.get("TENSORFORK_RUN")
  if hasattr(tensorfork, 'run'):
    if tensorfork.run['name'] == name or name is None:
      return tensorfork.run
    raise Exception("Already fetched tensorfork run")
  run = fetch(name=name)
  os.environ["TPU_NAME"] = run['meta']['tpu']
  os.environ["MODEL_DIR"] = run['meta']['logdir']
  tensorfork.run = run
  return run

def parse(x, skip_unknown=False):
  try:
    return gin.config.ParsedBindingKey(x)
  except ValueError:
    if skip_unknown:
      pass
    else:
      raise

def fqn(name):
  if '.' not in name:
    name = "knobs." + name
  return name

def bind(name, value, skip_unknown=False):
  name = fqn(name)
  try:
    gin.config.bind_parameter(name, value)
    return value
  except ValueError:
    if skip_unknown:
      pass
    else:
      raise

@gin.config.register_file_reader(IOError)
def tensorfork_reader(filename):
  try:
    run = fetch(filename)
    return run['cfg']
  except:
    import traceback
    traceback.print_exc()
    raise IOError()

class EasyDict(dict):
  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError:
      raise AttributeError(name)
  def __setattr__(self, name, value):
    self[name] = value
  def __delattr__(self, name):
    del self[name]

def gin_query(name, unset=None):
  try:
    return gin.query_parameter(name)
  except ValueError:
    if unset == gin.REQUIRED:
      raise
    return unset

def is_expr(s):
  if isinstance(s, str) and hasattr(tensorfork, 'g_lumen'):
    try:
      result = tensorfork.g_lumen.reader.read_string(s)
      if isinstance(result, list):
        return result
    except:
      pass

def L():
  if not hasattr(tensorfork, "g_lumen"):
    tensorfork.g_lumen = __import__('lumen')
  return tensorfork.g_lumen

def read_all(s):
  r = []
  stream = L().reader.stream(s)
  pos = stream['pos'] - 1
  while pos < stream['pos']:
    pos = stream['pos']
    expr = L().reader.read(stream)
    if expr is not L().reader.eof:
      r.append(expr)
  return r

def read(s):
  return L().reader.read_string(s)

def read_safe(s, fail=None, reader=read):
  try:
    return reader(s)
  except Exception as e:
    msg = str(e)
    if msg.startswith('Expected ') or msg.startswith("Unexpected "):
      if callable(fail):
        fail = fail()
      return fail
    else:
      raise

def read_all_safe(s, fail=None):
  return read_safe(s, fail=fail, reader=read_all)

def read_body(s):
  if isinstance(s, str):
    s = s.strip()
    if len(s) >= 2 and s[0] == "!":
      return read_all_safe(s[1:])

def evaluate(form, context=globals()):
  context = assign({}, context, L().__dict__)
  pp(form)
  return L().compiler.eval(form, context)

def read_eval(value, context=globals()):
  body = read_body(value)
  if body is not None:
    for expr in body:
      value = evaluate(expr, context=context)
  return value

@gin.configurable(blacklist=["unset", "context"])
def knobs_query(name, unset=None, context=globals()):
  name = fqn(name)
  #print('knobs_query', name, unset)
  value = gin_query(name, unset=unset)
  return value
  #print('knobs_query value', value)
  forms = read_body(value)
  #print('knobs_query forms', forms)
  if forms is not None:
    for form in forms:
      #print('knobs_query evaluate', form)
      value = evaluate(form, context=context)
      #print('knobs_query evaluated', value)
  return value

def has_knob(name):
  unbound = {}
  result = knobs_query(name, unset=unbound)
  return result is not unbound

# @gin.configurable("options")
# def options(*args, **keys):
#   return EasyDict(keys)

@gin.configurable("knobs")
def knobs(*args, **keys):
  if len(args) <= 0:
    return EasyDict(keys)
  k = args[0]
  default_value = None if len(args) <= 1 else args[1]
  unbound = {}
  result = knobs_query(k, unset=unbound)
  if result is unbound or len(args) >= 3 and args[2]:
    if len(args) >= 2:
      bind(k, default_value)
    return default_value
  return result

def reload(name=None, finalize_config=False, skip_unknown=False):
  name = get_name(name)
  res = fetch(name)
  with gin.config.unlock_config():
    bindings = res['gin_bindings']
    gin.parse_config_files_and_bindings([], bindings, finalize_config=finalize_config, skip_unknown=skip_unknown)
  return res

# @gin.configurable("foo")
# def f(bork=99, baz=100, quux=101, bar=21, wow=0, *args, **keys):
#   return [keys, bork, baz, quux, bar, wow, *args]

def main():
  import sys
  sys.path += [os.path.abspath('../compare_gan')]
  import compare_gan
  from compare_gan.gans import modular_gan
  reload("tensorfork/test#4", skip_unknown=True)
  cfg = get()
  pp(cfg)
  breakpoint()

if __name__ == "__main__":
  main()
