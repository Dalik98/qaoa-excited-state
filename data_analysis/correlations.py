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
import util

def create_all_state_plot(atom1 = "Li", atom2 = "H", p= None):

    
    input_folders = [util.getFolderName(state, atom1, atom2) for state in range(4)]

    if(p == None):
        input_folders = [folder + "_sigmax/diag" for folder in input_folders]
    else:
        input_folders = [folder + "_sigmax/qaoa_p{}/average".format(p) for folder in input_folders]
    
    extra_folder = "data/" + util.getMoleculeName(atom1, atom2) + "/extra"
    Path(extra_folder).mkdir(parents=True, exist_ok=True)

    y_array, x_distances = util.read_eigenvalues_data(input_folders)

    # y_array_div = [y_array[:len(y_array)//2], y_array[len(y_array)//2:]]
    y_array_div = [y_array]
    legend_array = [["ground state", "1st excited state", "2nd excited state", "3rd excited state"]]

    for i, y_array in enumerate(y_array_div):

        for y in y_array:
            # plt.plot(x_distances, y)
            plt.scatter(x_distances, y, s = 9)

        molecule_name = util.getMoleculeName(atom1, atom2)
        formatted_name = molecule_name
        if(molecule_name == "H2"):
            formatted_name = "$H_{2}$"

        plt.xlabel("Inter-nuclear distance (a.u.)")
        plt.ylabel("Energy (a.u.)")
        plt.title("Bond energy vs Inter-nuclear distance for {} {}".format(formatted_name, "" if p == None else "and p={}".format(p)))
        plt.legend(legend_array[i], loc = "upper right")

        plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))
        plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
        
        p_text = "" if p == None else "_p{}".format(p)

        name_file = extra_folder + '/all_states{}.png'.format(p_text)
        plt.savefig(name_file)
        print("File created:", name_file)

        plt.figure().clear()

def create_all_state_plot_qaoa(atom1 = "Li", atom2 = "H"):

    file_paths = [util.getFolderName(state, atom1, atom2) + "qaoa_p{}".format(p) for state in range(4)]

    extra_folder = "data/" + util.getMoleculeName(atom1, atom2) + "/extra"
    Path(extra_folder).mkdir(parents=True, exist_ok=True)

    y_array, x_distances = util.read_eigenvalues_data(file_paths)

    # y_array_div = [y_array[:len(y_array)//2], y_array[len(y_array)//2:]]
    y_array_div = [y_array]
    legend_array = [["ground state", "1st excited state", "2nd excited state", "3rd excited state"]]

    for i, y_array in enumerate(y_array_div):

        for y in y_array:
            # plt.plot(x_distances, y)
            plt.scatter(x_distances, y, s = 9)

        molecule_name = util.getMoleculeName(atom1, atom2)
        formatted_name = molecule_name
        if(molecule_name == "H2"):
            formatted_name = "$H_{2}$"

        plt.xlabel("Inter-nuclear distance (a.u.)")
        plt.ylabel("Energy (a.u.)")
        plt.title("Bond energy vs Inter-nuclear distance for {}".format(formatted_name))
        plt.legend(legend_array[i], loc = "upper right")

        plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))
        plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))
        
        name_file = extra_folder + '/all_states_{}_part{}.png'.format(util.getMoleculeName(atom1, atom2), i)
        plt.savefig(name_file)
        print("File created:", name_file)

        plt.figure().clear()
