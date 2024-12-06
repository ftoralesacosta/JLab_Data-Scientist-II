import gym
from functions import *
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

import numpy as np
import tensorflow as tf
from tensorflow import keras

class DDPG:
    """
    DDPG (Deep Deterministic Policy Gradient) Agent Class.

    This class implements an off-policy actor-critic approach 
    that uses a target actor and a target critic network for stable learning. 
    It stores experiences in a replay buffer and uses them to train the agent's networks.
    """

    def __init__(self,
                 actor_model: keras.Model,
                 critic_model: keras.Model,
                 target_actor: keras.Model,
                 target_critic: keras.Model,
                 num_states: int,
                 num_actions: int,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 critic_lr: float = 1e-3,
                 actor_lr: float = 1e-3,
                 buffer_capacity: int = 100000,
                 batch_size: int = 64):
        """
        Initialize the DDPG agent.

        Args:
            actor_model: The actor network (deterministic policy).
            critic_model: The critic network (action-value function approximator).
            target_actor: The target actor network (soft updates from the actor model).
            target_critic: The target critic network (soft updates from the critic model).
            num_states: Dimensionality of the state space.
            num_actions: Dimensionality of the action space.
            gamma: Discount factor for future rewards.
            tau: Soft update rate for the target networks.
            critic_lr: Learning rate for the critic optimizer.
            actor_lr: Learning rate for the actor optimizer.
            buffer_capacity: Maximum capacity of the replay buffer.
            batch_size: Batch size for training.
        """

        self.actor_model = actor_model
        self.critic_model = critic_model
        self.target_actor = target_actor
        self.target_critic = target_critic

        self.gamma = gamma
        self.tau = tau

        self.critic_optimizer = keras.optimizers.legacy.Adam(learning_rate=critic_lr)
        self.actor_optimizer = keras.optimizers.legacy.Adam(learning_rate=actor_lr)
        # .legacy supposedly faster on M1 macs (where I'm running this)

        # Replay Buffer parameters
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        # Initialize replay buffers
        # The typical loop for RL learning: state + action -> reward + new state
        self.state_buffer = np.zeros((self.buffer_capacity, num_states), dtype=np.float32)
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions), dtype=np.float32)
        self.reward_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states), dtype=np.float32)

    def record(self, obs_tuple: tuple):
        """
        Store a single transition (state, action, reward, next_state) into the replay buffer.
        """
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    @tf.function
    def train_step(self, state_batch, action_batch, reward_batch, next_state_batch):
        """
        Perform a single training step on both the critic and actor using one batch of data.

        Args:
            state_batch: A batch of states.
            action_batch: A batch of actions taken in those states.
            reward_batch: A batch of rewards observed after taking the actions.
            next_state_batch: A batch of subsequent states observed.
        """
        # Update critic
        with tf.GradientTape() as tape:
            # Compute target actions using the target actor
            target_actions = self.target_actor(next_state_batch, training=True)

            # Compute target Q-values using target critic
            y = reward_batch + self.gamma * self.target_critic([next_state_batch, target_actions], training=True)

            # Predicted Q-values by the critic
            critic_value = self.critic_model([state_batch, action_batch], training=True)

            # Mean squared error loss for critic
            critic_loss = tf.reduce_mean(tf.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        # Update actor
        with tf.GradientTape() as tape:
            # The actor tries to maximize the critic's predicted Q-value
            actions = self.actor_model(state_batch, training=True)
            critic_value_for_actions = self.critic_model([state_batch, actions], training=True)
            actor_loss = -tf.reduce_mean(critic_value_for_actions)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    def learn(self):
        """
        Sample a batch from the replay buffer and perform a training step.
        """
        record_range = min(self.buffer_counter, self.buffer_capacity)
        if record_range < self.batch_size:
            # Not enough samples to start training
            return

        # Randomly sample a batch of experiences from the replay buffer
        batch_indices = np.random.choice(record_range, self.batch_size, replace=False)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices], dtype=tf.float32)
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices], dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices], dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices], dtype=tf.float32)

        # Perform a training step
        self.train_step(state_batch, action_batch, reward_batch, next_state_batch)
