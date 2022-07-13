# SUPPORT FUNCTIONS FOR THE PROJECT
import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from tqdm import tqdm
from scipy import interpolate

def generate_populations_dataset():
    SH = np.array([20,20,52,83,64,68,83,12,36,150,110,60,7,10,70,100,92,70,10,11,137,137,18,22,52,83,18,10,9,65])
    CL = np.array([32,50,12,10,13,36,15,12,6,6,65,70,40,9,20,34,45,40,15,15,60,80,26,18,37,50,35,12,12,25])

    X = np.zeros((len(SH),2))
    X[:,0] = SH
    X[:,1] = CL

    dt = 2
    t = np.arange(1845,1903+dt,dt) 

    # plot dataset
    plt.figure()
    plt.plot(t,SH, '-b')
    plt.plot(t,CL, '-r')

    plt.legend(("Snowshoe Hare","Canadian Lynx"), loc = 2)

    plt.ylabel('number of animals')
    plt.xlabel('years')
    plt.ylim((0,200))
    plt.grid()
    

    return X,t,dt



def interpolation(t ,X, t_new):
    #f_0 = interpolate.interp1d(t, X[:,0])
    #f_1 = interpolate.interp1d(t, X[:,1])

    f_0 = interpolate.splrep(t, X[:,0])
    f_1 = interpolate.splrep(t, X[:,1])

    X_new = np.zeros((len(t_new),2))
    #X_new[:,0] = f_0(t_new)
    #X_new[:,1] = f_1(t_new)

    X_new[:,0] = interpolate.splev(t_new, f_0)
    X_new[:,1] = interpolate.splev(t_new, f_1)

    return X_new


def simulate_model( x_0, t_test, model, ensemble_coefs, ensemble_optimizer , integrator_kws , integrator = 'solve_ivp'):
    # Predict the testing trajectory with all the models

    x_test_sims = np.zeros((ensemble_coefs.shape[0],len(t_test), len(x_0)))
    for i in tqdm(range(ensemble_coefs.shape[0])):
        ensemble_optimizer.coef_ = ensemble_coefs[i, :, :]
        x_test_sims[i,:,:] = model.simulate(x_0, t_test, integrator=integrator, integrator_kws = integrator_kws)

    # Compute the 2.5 and 97.5 percentile trajectories
    bottom_line = np.percentile(x_test_sims, 0.5, axis=0)
    top_line = np.percentile(x_test_sims, 99.5, axis=0)
    x_test_sim_mean = np.mean(x_test_sims, axis=0)

    return x_test_sims, x_test_sim_mean, top_line, bottom_line



def plot_functions(t_test, x_test_sim_mean, t, X, top_line, bottom_line , feature_names):
    # Plot trajectory results
    plt.figure(figsize=(7, 10))
    for i in range(2):
        plt.subplot(2, 1, i + 1)
        plt.plot(t_test, x_test_sim_mean[:, i], 'b', label='Mean')
        plt.plot(t , X[:,i], '--k')
        plt.plot(t_test, bottom_line[:, i], 'g', alpha=0.6, label='95th percentile')
        plt.plot(t_test, top_line[:, i], 'g', alpha=0.6)
        ax = plt.gca()
        ax.fill_between(t_test, bottom_line[:, i], top_line[:, i], color='g', alpha=0.3)
        plt.grid(True)
        if i != 2:
            ax.set_xticklabels([])
        else:
            plt.xlabel('t', fontsize=20)
            plt.ylabel(feature_names[i], fontsize=20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            ax.yaxis.set_label_coords(-0.05, 0.75 - 0.1 * (i + 1))
            if i == 0:
                ax.legend(bbox_to_anchor=(1.01, 1.05), fontsize=18)