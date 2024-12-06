import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
from collections import deque
import random
import imageio
import os
from IPython.display import Image

class GaussianActionNoise:
    ''' introduces noise sampled from a Gaussian '''
    ''' distribution to the actions taken by an agent'''

    def __init__(self, mean=0, std_deviation=1.0):
        self.mean = mean
        self.std_dev = std_deviation

    def __call__(self):
        x = np.random.normal(loc=self.mean,
                             scale=self.std_dev)
        return x

def policy(state, noise_object,actor_model,lower_bound = -2, upper_bound = 2):

    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object() #FixME, double check
    sampled_actions = sampled_actions.numpy() + noise

    #make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]

def build_actor(state_dim, action_dim, action_bound):
    state_input = layers.Input(shape=(state_dim,))
    x = layers.Dense(64, activation='relu')(state_input)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(action_dim, activation='tanh')(x)
    output = output * action_bound  # Scale output to [-action_bound, action_bound]
    model = models.Model(inputs=state_input, outputs=output)
    return model

def build_critic(state_dim, action_dim):
    state_input = layers.Input(shape=(state_dim,))
    state_out = layers.Dense(16, activation='relu')(state_input)
    action_input = layers.Input(shape=(action_dim,))
    action_out = layers.Dense(16, activation='relu')(action_input)
    concat = layers.Concatenate()([state_out, action_out])
    x = layers.Dense(32, activation='relu')(concat)
    output = layers.Dense(1)(x)
    model = models.Model(inputs=[state_input, action_input], outputs=output)
    return model

def update_target(target, original, tau):
    target_weights = target.get_weights()
    original_weights = original.get_weights()

    for i in range(len(target_weights)):
        target_weights[i] = original_weights[i] * tau + target_weights[i] * (1 - tau)

    target.set_weights(target_weights)
