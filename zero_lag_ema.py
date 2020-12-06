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
  # the following code implements "Figure 2" of the paper. At least,
  # I thought it did. Figure 2 seems to show a signal starting at
  # zero, jumping upwards, rapidly oscillating up and down, then
  # finally dropping back to zero. Turns out, it's just a simple
  # step function: zero, then 20, then zero. But it was a happy
  # accident, because this has much more revealing behavior.
  #
  # Notice that the final ZLEMA value ends up oscillating up and down
  # *even after the input drops back to zero*. This seems like
  # unexpected, emergent behavior that the authors didn't point out.
  # The usual behavior of an EMA is that if there's a rapid
  # oscillation, the EMA value remains stable. But with ZLEMA, the
  # error-corrected value and the EMA value end up interacting in some
  # odd way that results in a rapid fluctuation even after the input
  # signal has stabilized.
  #
  e = ZeroLagEma(length=23, gain_limit=50)
  # first part of the signal: a constant zero value.
  plot = [e.add(0) for i in [1, -1]*5]
  # second part of the signal: an immediate step up to 500, followed
  # by a rapid oscillation like: 501, 499, 500, 501, 499, 500, ....
  plot += [pp(vars(e)) or e.add(500 + i) for i in [1, -1, 0]*20]
  # final part of the signal: drop back to zero.
  plot += [pp(vars(e)) or e.add(0) for i in [1, -1]*5]
  # print out the end result.
  pp(plot)


