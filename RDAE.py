from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model

def plot_ten_images(X_corr, X):
    n=10
    plt.figure(figsize=(20, 4))
    for i in range(n):

        # display original + noise
        ax = plt.subplot(2, n, i + 1)
        plt.title("original + noise")
        plt.imshow(tf.squeeze(X_corr[i]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        bx = plt.subplot(2, n, i + n + 1)
        plt.title("reconstructed")
        plt.imshow(tf.squeeze(X[i]))
        plt.gray()
        bx.get_xaxis().set_visible(False)
        bx.get_yaxis().set_visible(False)
    plt.show()

def corrupt(X, eps, p_corr):
    X_all_corr = X + eps * np.random.randn(X.shape[0], X.shape[1])
    mask = np.random.choice([0, 1], size=X.shape, p=[1-p_corr, p_corr]) != 0
    X_corr = X.copy()
    X_corr[mask] = X_all_corr[mask]
    return X_corr


def prox_l1(lam, x):
    return (x > lam) * (x - lam) + (x < -lam) * (x + lam)

def prox_l21(lam, x):
    e = np.linalg.norm(x, axis=0, keepdims=True)
    return (e > lam) * (x - lam*x/e)
    

def get_Dense_encoder():
    encoder = tf.keras.Sequential([
        layers.Input(shape=(784)),
        layers.Dense(units=200, activation='relu'),
        layers.Dense(units=10, activation='relu'),    
    ], name='Encoder')
    return encoder

def get_Dense_decoder():
    decoder = tf.keras.Sequential([
        layers.Dense(units=200, activation='relu'),
        layers.Dense(units=784, activation='sigmoid'),
    ], name='Decoder')
    return decoder

### Deep Dense Autoencoder Model
class DAE_Dense(Model):
    def __init__(self):
        super(DAE_Dense, self).__init__()
        self.encoder = get_Dense_encoder()
        self.decoder = get_Dense_decoder()
        
    def call(self, x, training=False):
        encoded = self.encoder(x, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded

    def encode(self, x, training=False):
        encoded = self.encoder(x, training=training)
        return encoded
    
    def decode(self, x, training=False):
        decoded = self.decoder(x, training=training)
        return decoded
    
    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Reconstruct input
            x_encoded = self.encode(x, training=True)
            x_recon = self.decode(x_encoded, training=True)
            # Calculate loss
            loss = self.compiled_loss(x, x_recon)
            
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(x, x_recon)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
            
    

### Robust Autoencoder Model
class RobustAutoencoder:
    def __init__(self, AE_type: str, prox_type: str):
        super(RobustAutoencoder, self).__init__()
        assert AE_type=='Dense' or AE_type=='LSTM', 'AE_type has to be either Dense or LSTM'
        self.AE_type = AE_type
        
        assert prox_type=='l1' or prox_type=='l21', 'prox_type has to be either l1 or l21'
        self.prox_type = prox_type
        
        if self.AE_type=='Dense':
            self.AE = DAE_Dense()
            self.AE.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                loss='mse',
                metrics=['mse']
            )
        
        if self.prox_type=='l1':
            self.prox_fn = prox_l1
        elif self.prox_type=='l21':
            self.prox_fn = prox_l21
            
    def train_and_fit(self, X, train_iter: int, AE_train_iter: int, batch_size: int, eps: float):
        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)
        self.LD = np.zeros(X.shape)
        self.LS = X
        for i in range(train_iter):
            print(f'RAE training iteration: {i+1}')
            self.LD = X - self.S
            # Now fit the autoencoder for some iters
            self.AE.fit(x=self.LD, batch_size=batch_size, epochs=AE_train_iter)
            self.LD = self.AE(self.LD).numpy()
            self.S = X - self.LD
            self.S = self.prox_fn(1.0, self.S)
            c1 = tf.linalg.norm(X - self.LD - self.S) / tf.linalg.norm(X)
            c2 = tf.linalg.norm(self.LS - self.LD - self.S) / tf.linalg.norm(X)
            if c1 < eps or c2 < eps:
                print(f'Early Denseergence at iter {i+1}')
            self.LS = self.LD + self.S
        return self.LD, self.S
                
        
        


if __name__=='__main__':
    (x_train, _), (x_test, _) = mnist.load_data()
    DAE = DAE_Dense()
    
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = x_train.reshape((x_train.shape[0], 784))
    x_test = x_test.reshape((x_test.shape[0], 784))
    
    x_train_corr = corrupt(x_train, eps=0.5, p_corr=0.1)

    print(f'Train data shape: {x_train.shape}')
    print(f'Test data shape: {x_test.shape}')
    RAEl1Dense = RobustAutoencoder(AE_type='Dense', prox_type='l1')
    
    LD, S = RAEl1Dense.train_and_fit(X=x_train_corr, train_iter=100, AE_train_iter=10, batch_size=1024, eps=1e-4)
    
    x_train_corr = x_train_corr.reshape((x_train_corr.shape[0], 28, 28, 1))
    LD = LD.reshape((LD.shape[0], 28, 28, 1))
    
    plot_ten_images(x_train_corr, LD)

