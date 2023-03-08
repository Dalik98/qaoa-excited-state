import matplotlib.pyplot as plt
import csv
import numpy as np
from pathlib import Path
import qutip as qt
import ast
import util
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MultipleLocator

def func(x, a, b, c):

    return a * (1 - np.exp(-x / b)) + c

class Fidelity():


    def __init__(self, state = 0, mix_mat = qt.sigmax(), atom1 = "H", atom2 = "H"):

        self.state = state
        self.mix_mat = mix_mat

        postfix_mixmat = util.getMixMatName(mix_mat)

        self.upper_folder = util.getFolderName(state, atom1, atom2) + "_" + postfix_mixmat
        self.molecule_name = util.getMoleculeName(atom1, atom2)

        self.fidelity = None

    #calculate the fidelity for all the p values selected, and creates the plot fidelity vs distance
    def get_fidelity(self, p_values = [], create_plots = False, custom_diag_folder = None, qaoa_folders = None, \
        custom_legend = None):

        main_folder = self.upper_folder if not custom_diag_folder else custom_diag_folder
        diag_file_path = main_folder + '/diag'
        qaoa_file_paths = []

        diag_final_states = []

        if(qaoa_folders == None):
            qaoa_file_paths = [self.upper_folder + '/qaoa_p{}'.format(p) for p in p_values]
        else:
            qaoa_file_paths = [qaoa_folder for qaoa_folder in qaoa_folders]

        diag_final_states_array, x_distances = util.read_eigenstates_data([diag_file_path])
        diag_final_states = diag_final_states_array[0]

        fidelities_array = []
        variance_array = []

        #The fidelity of the eigenstates are mediated through the number of instances under a value of p
        for qaoa_file_path, p in zip(qaoa_file_paths, p_values):
            with open(qaoa_file_path + "/average/info.txt", "r") as f:

                instances = f.readlines()[0].split(',')[1:]
                instances = [int(instance) for instance in instances]
                instances.sort()

                qaoa_instance_folders = [qaoa_file_path + "/{}".format(instance) for instance in instances]
                array_qaoa_instance_final_states, _ = util.read_eigenstates_data(qaoa_instance_folders)

                fidelity_instance_array = []

                for qaoa_instance_final_states in array_qaoa_instance_final_states:

                    fidelities = []

                    for i, final_state in enumerate(qaoa_instance_final_states):

                        fidelity = np.abs(final_state.overlap(diag_final_states[i])) ** 2
                        fidelities.append(fidelity)
                    
                    fidelity_instance_array.append(fidelities)

                fidelity_average_array = sum(np.array(fidelity_instance_array))/len(fidelity_instance_array)

                fidelities_array.append( fidelity_average_array )

            #calculation of the variance for a fixed p
            variance_fixed_p = []
            for i, fidelity_fixed_dist in enumerate(np.array(fidelity_instance_array).T):
                tot = []
                for fidelity in fidelity_fixed_dist:
                    tot.append( (fidelity - fidelity_average_array[i])**2 )

                variance_fixed_p.append( np.sqrt(sum(np.array(tot))/len(fidelity_fixed_dist)) ) 

            #It saves in file the variance and the fidelity for a fixed p
            data = [x_distances, variance_fixed_p]
            transposed_data = list(zip(*data))

            file_name = main_folder + '/qaoa_p{}/average'.format(p) + '/{}.csv'.format("variance_fidelity")
            with open(file_name, 'w', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f, delimiter=' ')
                    writer.writerows(transposed_data)
            print("Data saved: {}".format(file_name))

            variance_array.append(variance_fixed_p)

        self.fidelity = dict(zip(p_values, fidelities_array))

        if(create_plots):
            for n, (fidelities, variances) in enumerate(zip(fidelities_array, variance_array)):
                plt.errorbar(x_distances, fidelities, variances, fmt='.', markersize=9., label = "p={}".format(p_values[n]))

            plt.ylim([0, 1.05])
            plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))
            plt.ylabel("Fidelity")
            plt.xlabel("Inter-nuclear distance (a.u.)")
            #plt.title("Fidelity $|<\psi_{diag}|\psi_{qaoa}>|^{2}$ vs Inter-nuclear distance")

            legend = ["p = {}".format(p) for p in p_values] if not custom_legend else custom_legend
            file_to_write = self.upper_folder + '/corr' if not qaoa_folders else "data_extra"
            file_to_write += "/fidelity_vs_distance.png"

            plt.legend(legend, loc = "lower left")
            plt.savefig(file_to_write)
            plt.figure().clear()

            print("File created: {}".format(file_to_write))

    def get_fidelity_min_value(self, p_values = [], mode = "fidelity"):

        y_data = []
        y_variance = []

        diag_file_path = self.upper_folder + '/diag'
        qaoa_file_paths = [self.upper_folder + '/qaoa_p{}/average'.format(p) for p in p_values]

        binding_dist = util.BOND_LENGTH[self.molecule_name]
        index = list(util.NUCLEAR_DISTS[self.molecule_name]).index(binding_dist)

        if(mode == "fidelity"):
            fidelities = []
            fidelity_variance = []

            array_fidelity_variance, _ = util.read_variance_data(qaoa_file_paths, energy = False)

            for p, y_var in zip(p_values, array_fidelity_variance):
                fidelities.append(self.fidelity[p][index])
                fidelity_variance.append(y_var[index])

            y_data = fidelities
            
            y_variance = fidelity_variance

        if(mode == "energies"):

            norm_eigenenergies = []
            norm_variance = []

            diag_eigenvalues_array, _ = util.read_eigenvalues_data([diag_file_path])
            diag_eigenvalues = diag_eigenvalues_array[0]
            array_qaoa_eigenvalues, _ = util.read_eigenvalues_data(qaoa_file_paths)
            array_qaoa_variance, _ = util.read_variance_data(qaoa_file_paths)

            for y_qaoa, y_var in zip(array_qaoa_eigenvalues, array_qaoa_variance):
                norm_eigenenergies.append((y_qaoa[index] - diag_eigenvalues[index])/diag_eigenvalues[index])
                norm_variance.append(y_var[index]/diag_eigenvalues[index])

            y_data = norm_eigenenergies
            y_variance = norm_variance

        fig, ax = plt.subplots(1,1)

        # ax.scatter(p_values, y_data)
        ax.errorbar(p_values, y_data, yerr=y_variance, fmt='.')
        
        # plt.plot(x_values, y_fit_values, 'r', label='Fit: a={:.3f}, b={:.3f}, c={:.3f}'.format(*popt))
        ax.set_ylabel("Fidelity" if mode == "fidelity" else "Normalized energy")
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.set_xlabel("p")
        pre_title = "Fidelity $|<v_{diag}|v_{qaoa}>|^{2}$" if mode == "fidelity" else "Normalized energy"
        
        #ax.set_title(pre_title + " vs P" + " for they_variance bond length ({} a.u.)".format(util.round_num(binding_dist*util.ANG_TO_BOHR), 2))
        
        file_to_write = self.upper_folder + "/corr/" + mode + "_vs_p.png"

        plt.savefig(file_to_write)
        plt.figure().clear()

        print("File created:", file_to_write)

    #creates a plot, energy vs internuclear distance, with more than a value of p with different colors
    def plot_energy_from_file(self, p_values = [0], atoms = None):
        
        #loads the data for the ground state saved in the data.csv files for different values of p
        file_paths = [self.upper_folder + '/qaoa_p{}/average'.format(p) if p > 0 else self.upper_folder + '/diag' for p in p_values]

        dim = 54
        if(not atoms == None):
            if(util.getMoleculeName(atoms[0], atoms[1]) == "H2"):
                dim = 45

        y_eigenvalues_list, _ = util.read_eigenvalues_data(file_paths)
        variance_all_list, _ = util.read_variance_data(file_paths, dim)
        
        self.plot_energy_vs_distance(y_eigenvalues_list, p_values, variance_all_list = variance_all_list)

    def plot_save_energy(self, eigenvalues, final_states, folder = "", p = 0, variance_list = None):

        distances = util.NUCLEAR_DISTS[self.molecule_name] * util.ANG_TO_BOHR

        data_to_save_array = [np.real(eigenvalues), final_states]
        file_name_array = ["eigenenergies", "eigenstates"]

        if(not variance_list == None):
            data_to_save_array += [variance_list]
            file_name_array += ["variance"]

        for data_to_save, file_name in zip(data_to_save_array, file_name_array):

            data = [distances, data_to_save]
            transposed_data = list(zip(*data))

            with open(folder + '/{}.csv'.format(file_name), 'w', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f, delimiter=' ')
                    writer.writerows(transposed_data)
            print("Data saved: {}".format(folder + '/{}.csv'.format(file_name)))

        self.plot_energy_vs_distance([eigenvalues], p_values = [p], folder = folder, variance_all_list = [variance_list])

    def plot_energy_vs_distance(self, y_eigenvalues_list, p_values = [0], folder = "", variance_all_list = [None]):

        x_distances = util.NUCLEAR_DISTS[self.molecule_name] * util.ANG_TO_BOHR

        _, ax = plt.subplots(1,1)

        for y_eigenvalues, variances, p in zip(y_eigenvalues_list, variance_all_list, p_values):
            
            if(variances == None):
                ax.plot(x_distances, y_eigenvalues, "o", linewidth = 0.8)
            else:
                if(p != 0):
                    ax.errorbar(x_distances, y_eigenvalues, yerr=variances, fmt='.', markersize=10.)
                if(p == 0):

                    ax.errorbar(x_distances, y_eigenvalues, yerr=variances, fmt='.', markersize=5, color = 'black')

        formatted_name = self.molecule_name
        if(self.molecule_name == "H2"):
            formatted_name = "$H_{2}$"

        ax.set_xlabel("Inter-nuclear distance (a.u.)")
        ax.set_ylabel("Energy (a.u.)")
        #ax.set_title("Bond energy vs Inter-nuclear distance for {}".format(formatted_name))

        ax.xaxis.set_major_locator(MultipleLocator(0.5))

        max_value = np.max(np.array(y_eigenvalues_list))
        min_value = np.min(np.array(y_eigenvalues_list))

        ax.yaxis.set_ticks(np.arange(min_value, max_value + 0.10, 0.10))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
        
        if(not p_values[0] == -1):
            plt.legend(["p = {}".format(p) if p > 0 else "diag" for p in p_values], loc = "upper right")

        folder = self.upper_folder + "/corr" if folder == "" else folder
        Path(folder).mkdir(parents=True, exist_ok=True)
        plt.savefig(folder + '/energy_vs_distance.png')
        plt.figure().clear()

        print("File created: {}".format(folder + '/energy_vs_distance.png'))
