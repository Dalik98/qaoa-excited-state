import matplotlib.pyplot as plt
import csv
import numpy as np
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from sklearn.metrics import r2_score
from numpy import savetxt
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import qutip as qt
import ast

from data_analysis.correlations import main_folder
from data_analysis.fidelity import get_fidelity
import util

def func(x, a, b, c):

    return a * (1 - np.exp(-x / b)) + c

def r2_corr(p_values = [0, 1, 2, 3, 4, 5, 6, 7], data_type = "eigenenergies"):

    data_used = data_type if data_type == "eigenenergies" else "fidelity"
    folder = main_folder + "/corr/{}_corr".format(data_used)
    Path(folder).mkdir(parents=True, exist_ok=True)
    folder_corr = folder + "/linear_corr/"
    Path(folder_corr).mkdir(parents=True, exist_ok=True)

    y_qaoa = None
    x_diag = None

    if(data_type == "eigenenergies"):
        y_ground_array, _= load_data(p_values)
        x_diag = y_ground_array[0]
        y_ground_array = y_ground_array[1:]

    elif(data_type == "eigenstates"):
        y_ground_array, _ = get_fidelity(p_values[1:])

        for y in range(len(p_values[1:])):
            for x in range(len(y_ground_array[0])):
                y_ground_array[y][x] *= x + 1

        x_diag = [n + 1 for n in range(len(y_ground_array[0]))]

    else:
        raise Exception("data_type must be equal \"eigenenergies\" or \"eigenstates\"") 

    p_values = p_values[1:]

    b_array = []    
    m_array = []
    r2_array = []

    for i, p in enumerate(p_values):

        y_qaoa = y_ground_array[i]

        b, m = np.polyfit(x_diag, y_qaoa, 1)
        r2 = r2_score(y_qaoa, x_diag)

        x_line = np.arange(min(x_diag), max(x_diag) + 0.1, 0.1)
        y_line = b*x_line + m

        print("P{}; b:{}, m:{}, r2:{}".format(p, b, m, r2))
        b_array.append(b)
        m_array.append(m)
        r2_array.append(r2)

        if main_folder == "data_ground":
            plt.ylim([-1.15, 0])
            plt.xlim([-1.15, -0.91])

        plt.scatter(x_diag, y_qaoa)
        plt.plot(x_line, y_line, 'r')

        label_prefix = "Energy (a.u.)" if data_type == "eigenenergies" else "Fidelity"

        plt.xlabel("{} - Diagonalization method".format(label_prefix))
        plt.ylabel("{} - QAOA method".format(label_prefix))
        plt.title("Linear correlation")
        
        plt.savefig(folder_corr + '/linear_correlation_p{}.png'.format(p))
        plt.clf()

    # save the calculation from the linear correlations in a csv file
    data = np.array([p_values, b_array, m_array, r2_array]).transpose()
    savetxt(folder + '/liner_corr_data.csv', data, delimiter=' ', fmt="%.9f")

    popt, pcov = curve_fit(func, p_values, r2_array)
    x_values = np.arange(p_values[0], p_values[-1] + 0.25, 0.25)
    fit_function = func(x_values, popt[0], popt[1], popt[2])

    # plot the r2 in function of the values of p and save it on a file
    plt.scatter(p_values, r2_array)
    plt.plot(x_values, fit_function, 'r', label='Fit: a={:.3f}, b={:.3f}, c={:.3f}'.format(*popt))
    plt.xlabel("p")
    plt.ylabel("$R^2$")
    plt.title("$R^2$ vs p (using {} data)".format(data_used))
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.25))
    plt.legend(loc = 'lower right')

    plt.savefig(folder + "/rsquare_vs_p.png")
    plt.clf()

    # plot the linear coefficient b in function of the values of p and save it on a file
    plt.scatter(p_values, b_array)
    plt.xlabel("p")
    plt.ylabel("Linear coefficient")
    plt.title("Linear coefficient vs p (using {} data)")
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    # plt.gca().yaxis.set_major_locator(MultipleLocator(0.25))

    plt.savefig(folder + '/linear_coefficent_vs_p.png')

