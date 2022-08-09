from matplotlib import pyplot as plt
import numpy as np

def plot_ts(ts):
    plt.grid()
    plt.plot(np.arange(len(ts)), ts)
    plt.title("Time series")
    plt.ylim(0, 1)
    plt.show()
    
def plot_ts_recon(ts, ts_recon):
    plt.grid()
    plt.plot(np.arange(len(ts)), ts)
    plt.plot(np.arange(len(ts)), ts_recon)
    plt.ylim(0, 1)
    plt.show()
    
    
def save_ts(ts, filename):
    plt.grid()
    plt.plot(np.arange(len(ts)), ts)
    plt.title("Time series")
    plt.ylim(0, 1)
    plt.savefig(filename, bbox_inches='tight')
    
def save_ts_recon(ts, ts_recon, filename):
    plt.grid()
    plt.plot(np.arange(len(ts)), ts)
    plt.plot(np.arange(len(ts)), ts_recon)
    plt.ylim(0, 1)
    plt.savefig(filename, bbox_inches='tight')