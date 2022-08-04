from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model


def prox_l1(lam, x):
    return (x > lam) * (x - lam) + (x < -lam) * (x + lam)

def prox_l21(lam, x):
    e = np.linalg.norm(x, axis=0, keepdims=True)
    return (e > lam) * (x - lam*x/e)
    

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print(f'Train data shape: {x_train.shape}')
print(f'Test data shape: {x_test.shape}')

class RobustAutoencoderConv(Model):
    def __init__(self, latent_dim):
        super(RobustAutoencoderConv, self).__init__()
        self.latent_dim = latent_dim
        
        
prova1 = np.array([1, 2, 3, 4, 5, 6])
prova2 = np.array([[1, 2, 3], [4, 5, 6]])


