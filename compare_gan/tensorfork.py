from github import Github, Issue
from getpass import getpass
import os
import re
import gin
import time

class Namespace():
  pass

tensorfork = globals().get("tensorfork", Namespace())

env = {}

def opts(env=env, name=".env"):
  if time.time() - env.get("parsed", 0) < 5.0:
    return env
  if os.path.exists(name):
    with open(name) as f:
      s = f.read()
    for k, v in parse_string(s):
      env[k] = v
  env["parsed"] = time.time()
  return env

def parse_string(s):
  if isinstance(s, list):
    s = '\n'.join(s)
  parser = gin.config_parser.ConfigParser(s, gin.config.ParserDelegate(skip_unknown=True))
  for statement in parser:
    k = '{}.{}'.format(statement.selector, statement.arg_name)
    v = statement.value
    yield k, v

def login(username=None, password=None):
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
  if not hasattr(tensorfork, 'repo'):
    tensorfork.repo = tensorfork.github.get_repo("tensorfork/tensorfork")
  return tensorfork.repo

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
    return '@{}'.format(v.selector)
  else:
    return repr(v)

def parse_bindings(s):
  if isinstance(s, str):
    s = parse_string(s)
  for k, v in s:
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
  if isinstance(x, int):
    return login().get_issue(number=x)
  if isinstance(x, str):
    for issue in fetch_all(login().get_issues()):
      title = x.lower().strip()
      if parse_title(issue.title).get("run") == title:
        return issue
  raise Exception("Could not fetch run: {}".format(x))

def parse_issue_config(issue):
  cfg = parse_issue(issue)
  return '\n'.join(cfg)

def get(name=None):
  if not hasattr(tensorfork, 'run') and name is None:
    name = os.environ.get("TENSORFORK_RUN")
  if hasattr(tensorfork, 'run'):
    if tensorfork.run['name'] == name or name is None:
      return tensorfork.run
    raise Exception("Already fetched tensorfork run")
  tensorfork.run = {
      'meta': {
        'name': None,
        'tpu': os.environ.get("TPU_NAME", ""),
        'logdir': os.environ.get("MODEL_DIR", ""),
        },
      'gin': [],
  }
  if name is not None:
    tensorfork.run['name'] = name
    tensorfork.run['issue'] = fetch_issue(name)
    tensorfork.run['meta'] = parse_title(tensorfork.run['issue'].title)
    tensorfork.run['cfg'] = parse_issue_config(tensorfork.run['issue'])
    parsed = list(parse_string(tensorfork.run['cfg']))
    tensorfork.run['gin'] = dict(parsed)
    tensorfork.run['gin_bindings'] = list(parse_bindings(parsed))
  os.environ["TPU_NAME"] = tensorfork.run['meta']['tpu']
  os.environ["MODEL_DIR"] = tensorfork.run['meta']['logdir']
  return tensorfork.run

def main():
  cfg = get()
  from pprint import pprint as pp
  pp(cfg)

if __name__ == "__main__":
  main()
