# -*- coding: utf-8 -*-

import gym
import time
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os
from keras.models import load_model
from keras import backend as K
EPISODES = 3000

class DQN(object):
    def __init__(self,state_size, action_size,dd=True):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=3000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.train_batch = 32
        self._model = self._createModel(dd)
        
    @property
    def model(self):# 定义为只读属性
        return self._model
    
    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)
    
    def _createModel(self,dd=True):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        if dd:
            model.compile(loss=self._huber_loss,optimizer=Adam(lr=self.learning_rate))
        else:
            model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return model
        
    def train(self):
        if len(self.memory)>=self.train_batch:
            minibatch = random.sample(self.memory,self.train_batch) 
            state_batch = np.zeros([self.train_batch,self.state_size])
            target_batch = np.zeros([self.train_batch,self.action_size]) 
            for i,(state, action, reward, next_state, done) in enumerate(minibatch):
                state_batch[i,:] = state
                target_batch[i,:] = self.predict_action(state)
                target_batch[i,action] = reward if done else reward+self.gamma*np.amax(self.predict_action(next_state)[0])
            self.model.fit(state_batch, target_batch, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
    def predict_action(self,state):# 预测动作
        return self.model.predict(state)
    def act(self,state):# 执行的动作，具有随机性
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            #print(self.predict_action(state))
            return np.argmax(self.predict_action(state)[0])
        
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
        #self._train()
    def save(self,name = 'models/test'):
        self.model.save(name)
        self.saveWeight(name)
    def load(self,name = 'models/test'):
        self._model= load_model(name)
    def saveWeight(self,name = 'models/test'):
        self.model.save_weights(name+'.weight')
    def loadWeight(self,name = 'models/test'):
        self.model.load_weights(name+'.weight')
        
        
if __name__=='__main__':
    saveModelName = 'models/test'
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQN(state_size,action_size,False)   
    for times in range(EPISODES):
        state = env.reset().reshape(1,state_size) 
        for i in range(199):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = 0.1 if not done else -1
            next_state = next_state.reshape(1,state_size)
            agent.remember(state,action,reward,next_state,done)
            state = next_state
            if done:
                print('[times]:{}/{}\t\t[i]:{}\t\t[epsilon]:{}'.format(times,EPISODES,i,agent.epsilon))
                break
            if i == 198:
                print('[times]:{}/{}\t\t[i]:{}\t\t[epsilon]:{}\t#success#'.format(times,EPISODES,i,agent.epsilon))
        agent.train()
        if (times+1)%100==0:
            agent.save(saveModelName+str(times+1))
            print('[Saved] savename: `%s`'%(saveModelName+str(times+1)))
