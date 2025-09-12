import numpy as np
import matplotlib.pyplot as plt

# computes the value of epsilon which decreases during training

def epsilon(iter,n_games,k,eps_0):
    return (np.exp(k)-np.exp(k*iter/n_games))*eps_0/(np.exp(k)-1)
def plot_epsilon(n_games,k_list,eps_0): # displays the evolution of epsilon when k changes
    X = np.arange(0,n_games,0.1)
    Y = []
    for k in k_list:
        Y.append((np.exp(k)-np.exp(k*X/n_games))*eps_0/(np.exp(k)-1))
    for i, Yk in enumerate(Y):
        plt.plot(X,Yk, label=f'k={k_list[i]}')
    plt.xlabel('Training Episodes')
    plt.ylabel('Epsilon Value')
    plt.title('Epsilon Decay Strategies')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/epsilon_decay_plot.png', dpi=300, bbox_inches='tight')
    print("📊 Plot saved as 'epsilon_decay_plot.png'")
    plt.show()  # Now this will work!

if __name__=="__main__":
    plot_epsilon(1000,[1,2,3],0.2)