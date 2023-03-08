import numpy as np
import csv
import ast
import qutip as qt
import math

main_folder = "data/ground_state_sigmax"

NUCLEAR_DISTS = {"H2": np.arange(0.25, 2.5, 0.05, dtype = float).round(2), "LiH": np.arange(0.6, 3.3, 0.05, dtype = float).round(2)}
BOND_LENGTH = {"H2": 0.75, "LiH": 1.5}

ANG_TO_BOHR = 1.8897259886
FOLDER_NAMES = ["ground_state", "excited_state", "excited_state_2", "excited_state_3"]
LAMBDA = 10

def getMoleculeName(atom1, atom2):
    atom_names = [atom1, atom2]
    atom_names.sort()
    atom_names.reverse()

    if(atom1 == "H" and atom1 == atom2):
        return "H2"
    return atom_names[0] + atom_names[1]

def getFolderName(state = None, atom1 = "Li", atom2 = "H"):

    folder = "data/{}".format(getMoleculeName(atom1, atom2))
    if(state != None): folder += "/{}".format(FOLDER_NAMES[state])

    return folder

def round_num(number, digits = 2): return math.trunc(np.round(number * 10**digits))/(10**digits)

def read_eigenvalues_data(file_paths):

    return read_float_data(file_paths, "eigenenergies")

def read_variance_data(file_paths, dim = 54, energy = True):

    return read_float_data(file_paths, "variance" if energy else "variance_fidelity", dim = dim)

def read_float_data(file_paths, file_name, dim = 54):

    eigenvalues_list = []
    distances = []

    #it reads from file eigenvalues calculated through qaoa
    for n, path in enumerate(file_paths):

        if("diag" in path and file_name == "variance"):
            eigenvalues_list.append([0]*dim)
            continue

        path += "/{}.csv".format(file_name)

        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ')

            eigenvalues = []

            for row in spamreader:
                eigenvalue = ast.literal_eval(row[1])
                eigenvalues.append(float(eigenvalue))

                if(n == 0):
                    distance = ast.literal_eval(row[0])
                    distances.append(float(distance))

        eigenvalues_list.append(eigenvalues)
            
    return eigenvalues_list, distances

def read_eigenstates_data(file_paths):

    final_states_list = []
    distances = []

    #it reads from file eigenstates calculated through qaoa
    for n, path in enumerate(file_paths):

        path += "/eigenstates.csv"

        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ')

            final_states = []

            for row in spamreader:
                final_state = ast.literal_eval(row[1])
                final_states.append(qt.Qobj([final_state]))

                if(n == 0):
                    distance = ast.literal_eval(row[0])
                    distances.append(float(distance))

        final_states_list.append(final_states)
            
    return final_states_list, distances

def getMixMatName(mix_mat):

    postfix_mixmat = ""

    if(mix_mat == qt.sigmay()):
        postfix_mixmat = "sigmay"

    if(mix_mat == qt.sigmaz()):
        postfix_mixmat = "sigmaz"

    if(mix_mat == qt.sigmax()):
        postfix_mixmat = "sigmax"

    return postfix_mixmat
