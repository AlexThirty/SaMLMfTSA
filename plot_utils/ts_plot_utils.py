from tracemalloc import start
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
    plt.savefig(filename)
    
def save_ts_recon(ts, ts_recon, ts_L, ts_S, startpoint, zoom, filename):
    indeces = np.arange(len(ts))+startpoint
    max_ts = np.max(ts)
    min_ts = np.min(ts)
    max_recon = np.max(ts_recon)
    min_recon = np.min(ts_recon)
    max_L = np.max(ts_L)
    min_L = np.min(ts_L)
    max_S = np.max(ts_S)
    min_S = np.min(ts_S)
    lower = np.min(np.array([-0.1, min_ts, min_recon, min_L, min_S]))
    upper = np.max(np.array([1.1, max_ts, max_recon, max_L, max_S]))
    plt.grid()
    plt.plot(indeces, ts, label='Original')
    plt.plot(indeces, ts_recon, label='Reconstruction')
    plt.plot(indeces, ts_L, label='Cleaned')
    plt.plot(indeces, ts_S, label='Sparse')
    plt.title('RDAE on ts from '+str(startpoint)+' to '+str(startpoint+len(ts)))
    plt.legend()
    if zoom:
        plt.ylim(-0.1, 1.1)
    else:
        plt.ylim(lower, upper)
    plt.ylabel('Normalized value')
    plt.xlabel('Timestep')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()