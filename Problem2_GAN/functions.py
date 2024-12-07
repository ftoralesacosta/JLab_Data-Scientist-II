import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
from tensorflow.keras import optimizers, Model
import tensorflow as tf

# Load data
ages_data = np.load('./eICU_age.npy')
ages = ages_data['age']


# Define the generator model
def build_generator(latent_dim):
    model = Sequential([
        Dense(64, input_dim=latent_dim),
        LeakyReLU(alpha=0.01),
        BatchNormalization(momentum=0.8),
        Dense(128),
        LeakyReLU(alpha=0.01),
        BatchNormalization(momentum=0.8),
        Dense(1, activation='linear')
    ])
    return model


# Define the discriminator model
def build_discriminator():
    model = Sequential([
        Dense(128, input_dim=1),
        LeakyReLU(alpha=0.01),
        Dense(64),
        LeakyReLU(alpha=0.01),
        Dense(1, activation='sigmoid')
    ])
    return model


# Training the GAN
def train_gan(generator,
              discriminator,
              gan, ages, epochs,
              batch_size, latent_dim):

    d_losses = []
    g_losses = []

    for epoch in range(epochs):

        # Random noise to latent space
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        # Generate fake ages
        generated_ages = generator.predict(noise, verbose=0)

        # Get a random batch of real ages
        idx = np.random.randint(0, ages.shape[0], batch_size)
        real_ages = ages[idx]
        real_ages = real_ages.reshape(-1, 1)

        # Labels for generated and real data
        fake_labels = np.zeros((batch_size, 1))
        real_labels = np.ones((batch_size, 1))

        # Combine real and fake data
        combined_ages = np.concatenate([real_ages, generated_ages])
        combined_labels = np.concatenate([real_labels, fake_labels])

        # Shuffle
        indices = np.arange(combined_ages.shape[0])
        np.random.shuffle(indices)
        combined_ages = combined_ages[indices]
        combined_labels = combined_labels[indices]
        combined_labels += 0.05 * tf.random.uniform(tf.shape(combined_labels))  #noise for stability

        # descriminate
        d_loss = discriminator.train_on_batch(combined_ages, combined_labels)

        # separate training, I prefer combinet training, since equal portions
        # d_loss_real = discriminator.train_on_batch(real_ages, real_labels)
        # d_loss_fake = discriminator.train_on_batch(generated_ages, fake_labels)
        # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_y)

        d_losses.append(d_loss[0])  # BCE, accuracy
        g_losses.append(g_loss)

        if epoch % 100 == 0:
            print(f'Epoch: {epoch} [D loss: {d_loss[0]}] [G loss: {g_loss}]')

    return d_losses, g_losses
