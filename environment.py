from functions import getStockDataVec, getState, formatPrice

class SimpleTradeEnv(object):
  def __init__(self, stock_name, window_size, agent):
    self.data = getStockDataVec(stock_name)
    self.window_size = window_size
    self.agent = agent

  def step(self, action):
    # 0: Sit
    # 1: But
    # 2: Sell
    assert(action in (0, 1, 2))

    # State transition
    next_state = getState(self.data, self.t + 1, self.window_size + 1)

    # Reward
    if action == 0:
      reward = 0

    elif action == 1:
      reward = 0
      self.agent.inventory.append(self.data[self.t])
      print("Buy: " + formatPrice(self.data[self.t]))

    else:
      if len(self.agent.inventory) > 0:
        bought_price = self.agent.inventory.pop(0)
        profit = self.data[self.t] - bought_price
        reward = max(profit, 0)
        self.total_profit += profit
        print("Sell: " + formatPrice(self.data[self.t]) +
              " | Profit: " + formatPrice(profit))
      else:
        reward = 0 # try to sell, but con't do

    done = True if self.t == len(self.data) - 2 else False
    self.t += 1

    return next_state, reward, done, {}

  def reset(self):
    self.t = 0
    self.total_profit = 0
    return getState(self.data, self.t, self.window_size + 1)
