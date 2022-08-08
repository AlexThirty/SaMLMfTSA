from matplotlib import pyplot as plt
import numpy as np

def plot_ts(ts):
    plt.grid()
    plt.plot(np.arange(len(ts)), ts)
    plt.title("Time series")
    plt.show()
    
def plot_ts_recon(ts, ts_recon):
    plt.grid()
    plt.plot(np.arange(len(ts)), ts)
    plt.plot(np.arange(len(ts)), ts_recon)
    plt.show()