import numpy as np

# http://www.mesasoftware.com/papers/ZeroLag.pdf

class ZeroLagEma:
  def __init__(self, *, length=20, gain_limit=50, thresh=0.75):
    self.length = length
    self.gain_limit = gain_limit
    self.alpha = 0.0
    self.best_gain = 0.0
    self.thresh = thresh
    self.ec = None
    self.ec_1 = None
    self.ec_2 = None
    self.error = 0.0
    self.ema = None
    self.ema_1 = None
    self.ema_2 = None
  def dist(self, a, b):
    #return ((a - b) ** 2).sum() ** 0.5
    return np.linalg.norm(a - b)
  def add(self, close):
    if self.ec is None:
      self.ec = self.ec_1 = self.ec_2 = close
    if self.ema is None:
      self.ema = self.ema_1 = self.ema_2 = close
    self.alpha = 2 / (self.length + 1)
    self.ema_2 = self.ema_1
    self.ema_1 = self.ema
    self.ec_2 = self.ec_1
    self.ec_1 = self.ec
    self.ema = self.alpha * close + (1 - self.alpha) * self.ema_1
    self.least_error = 10000000.0
    for value1 in np.arange(-self.gain_limit, self.gain_limit + 1):
      gain = value1 / 10
      self.ec = self.alpha * (self.ema + gain*(close - self.ec_1)) + (1 - self.alpha) * self.ec_1
      #self.error = close - self.ec
      self.error = self.dist(close, self.ec)
      #if np.abs(self.error) < self.least_error:
      if self.error < self.least_error:
        #self.least_error = np.abs(self.error)
        self.least_error = self.error
        self.best_gain = gain
    self.ec = self.alpha * (self.ema + self.best_gain*(close - self.ec_1)) + (1 - self.alpha) * self.ec_1
    self.close = close
    # self.close_thresh = 100*self.least_error / close
    # if self.ec > self.ema and self.ec_1 < self.ema:
    #   if 100*self.least_error / close > self.thresh:
    #     print('buy next bar', close, vars(self))
    # if self.ec < self.ema and self.ec_1 > self.ema:
    #   if 100*self.least_error / close > self.thresh:
    #     print('sell short next bar', close, vars(self))
    return self.ec, self.ema


if __name__ == '__main__':
  from pprint import pprint as pp
  from importlib import reload; import zero_lag_ema as zle; reload(zle); e = zle.ZeroLagEma(length=23, gain_limit=50)
  plot = [e.add(0) for i in [1, -1]*5]; plot += [pp(vars(e)) or e.add(500 + i) for i in [1, -1, 0]*20]; plot += [pp(vars(e)) or e.add(0) for i in [1, -1]*5]; pp(plot)


