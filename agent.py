import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque

 
class Agent:
  def __init__(self, state_size, is_eval=False, eps_d=0.98, model_name=""):
    self.state_size = state_size  # normalized previous days
    self.action_size = 2  # change and not change
    self.memory = deque(maxlen=10000)
    self.inventory = []
    self.model_name = model_name
    self.is_eval = is_eval

    self.gamma = 0.95 # Future rewards normally 0.95
    self.epsilon = 1.0
    self.epsilon_min = 0.02
    self.epsilon_decay = eps_d # original value 0.998
    self.learning_rate = 1

    self.model = load_model("models/" + model_name) if is_eval else self._model()

  def _model(self):
    model = Sequential()
    model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=8, activation="relu"))
    model.add(Dense(self.action_size, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=0.001))
    return model

  def act(self, state, e):
    if not self.is_eval and np.random.rand() <= self.epsilon and e % 10 != 0: # in multiples of 10 we test the model
    #if not self.is_eval and np.random.rand() <= self.epsilon: # in multiples of 10 we test the model
      return random.randrange(self.action_size)
    options = self.model.predict(state)
    return np.argmax(options[0])

  def expReplay(self, batch_size):
    mini_batch = []
    l = len(self.memory)
    for i in range(l - batch_size + 1, l):
      mini_batch.append(self.memory.popleft())

    states, targets = [], []
    for state, action, reward, next_state, done in mini_batch:
      target = reward
      if not done:
        target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

      target_f = self.model.predict(state)
      target_f[0][action] = target

      states.append(state)
      targets.append(target_f)

    self.model.fit(np.vstack(states), np.vstack(targets), epochs=1, verbose=0)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay
    
  def Replay(self, batch_size=100, episode=0): 
    if len(self.memory) > batch_size:   # If the memory is higher than the size of the batch
        " vectorized implementation; 30x speed up compared with for loop "
        minibatch = random.sample(self.memory, batch_size) # A number of randomly selected rows are used to retrain the 
                                                           # neural network every N steps
        states = np.array([tup[0][0] for tup in minibatch]) # A list of arrays of the size of the window 
        actions = np.array([tup[1] for tup in minibatch])   # An array with the actions of the size of "batch_size"
        rewards = np.array([tup[2] for tup in minibatch])   # An array with the rewards 
        next_states = np.array([tup[3][0] for tup in minibatch]) # A list of arrays of the size of the window
        done = np.array([tup[4] for tup in minibatch])           # A list of boolean values to see if it is the last state
        # Q(s', a)
        target = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1) # ???


        # end state target is reward itself (no lookahead)
        target[done] = rewards[done]

        # Q(s, a)
        target_f = self.model.predict(states)

        # make the agent to approximately map the current state to future discounted reward
        # target_f is updated only for those states & action of the batch_size
        # if state x is recorded for action 1, then only action 1 will be updated    
        # https://github.com/piyush2896/Q-Learning
        # target_f[range(batch_size), actions] = target
        target_f[range(batch_size), actions] = ((1 - self.learning_rate)*target) + (self.learning_rate*target)

        # The neural network is retrained with the sample of batches of the model
        self.model.fit(states, target_f, epochs=1, verbose=0)

  def decrease_epsilon(self):  # As the number of episodes increases, the exploration rate goes down
    if self.epsilon > self.epsilon_min: # If the epsilon value is higher 
                                                #than the minimum threshold, we can decrease the value
      self.epsilon *= self.epsilon_decay # The decrease is multiplied by a decay factor

