import qutip as qt
import time
import numpy as np
from datetime import datetime
import multiprocessing
from queue import Queue
import threading
import math
from pathlib import Path
import ast
import csv
import util
import os

from data_analysis.fidelity import Fidelity
from qu_chemistry import qu_chemistry
from qaoa import Qaoa

class CalcHandler():

    def __init__(self, state = 0, atom1="H", atom2="H", mix_matrix = qt.sigmax()):

        # self.ANG_TO_BOHR = 1.8897259886 #angstrom to bohr conversion
        self.state = state #the number of the state to find, e.g. state = 0 is the ground state
        self.atom1 = atom1
        self.atom2 = atom2
        self.qaoa_final_states = []

        self.mix_matrix = mix_matrix
        self.mix_matrix_name = util.getMixMatName(self.mix_matrix)
        self.fid = Fidelity(state = state, atom1 = atom1, atom2 = atom2)

        self.general_main_folder = util.getFolderName(state, atom1, atom2)
        self.q_chem = qu_chemistry(n_qubits = 4, atom1 = self.atom1, atom2 = self.atom2, temp = False)

    # this function uses the QAOA algorithm to calculate eigenenergies and eigenvectors of the hamiltonians
    def find_optimal_distance_qaoa(self, p, qaoa_gs, sgd_method, grad_desc_opts):

        folder = self.general_main_folder + "_" + self.mix_matrix_name + "/qaoa_p{}".format(p)
        inst = self.getNewInstances(folder)
        folder += "/" + str(inst)

        #In case we want to find the excited states
        if(self.state > 0):
            file_path = util.getFolderName(0, self.atom1, self.atom2) + '_' + self.mix_matrix_name + '/qaoa_p5/average'

            self.qaoa_final_states, _ = util.read_eigenstates_data([file_path])
            self.qaoa_final_states = self.qaoa_final_states[0]
        
        start_calc_string = "Starting the calculation for the state {} of the molecule {} with QAOA: p = {}, mixing: {}"\
            .format(self.state, util.getMoleculeName(self.atom1, self.atom2), p, "X Pauli" if self.mix_matrix == qt.sigmax() else "Y Pauli")
        if(self.state > 0): start_calc_string += ", use QAOA g.s.: {}".format(qaoa_gs)
        sgd_param_string = "Using the {} algorithm with n. iterations: {} ".format(sgd_method, grad_desc_opts[0])
        print(start_calc_string)
        print(sgd_param_string)
        print("Independent run: {}".format(inst))

        start = time.perf_counter()

        qa = Qaoa(n_qubits = 2, mixing_matrix = self.mix_matrix)

        y = []
        final_states = []

        print("Inter-nuclear distances considered from {} to {} step 0.05 A".format(self.q_chem.distances[0], self.q_chem.distances[-1]))

        for n, distance in enumerate(self.q_chem.distances):

            ham = self.q_chem.calc_reduced_ham(distance)
            diag_ground_state = np.round(ham.eigenenergies()[self.state], 5)

            first_state = None
            #Only if we are calculating the excited state
            if(self.state > 0):

                if(qaoa_gs):
                    first_state = self.qaoa_final_states[n]
                else:
                    first_state = ham.eigenstates()[1][0]

            qa.set_ham(ham, eigenstate = first_state, lambda_ = util.LAMBDA)

            if(sgd_method == 'basin'):
                qa.optimizeBasinHopper(p, grad_desc_opts[0])
            elif(sgd_method == 'adam'):
                qa.optimizeAdam(p, grad_desc_opts[0])

            #get the least energy of the hamiltonian
            y.append(qa.fp.real)
            final_states.append(list(qa.final_state.dag().full()[0]))

            print("At distance: {} A, calculated eigenenergy: {}/{} a.u.".\
                format(util.round_num(distance), util.round_num(y[-1], 5), diag_ground_state, flush=True))

        duration = time.perf_counter() - start

        print("Algorithm executed in: {:.4f} s".format(duration))

        #This creates the folder where data will be stored
        Path(folder).mkdir(parents=True, exist_ok=True)

        self.fid.plot_save_energy(y, final_states, folder = folder, p = p)

        dateTimeObj = datetime.now()
        todaydate = "{}/{}/{}".format(dateTimeObj.day, dateTimeObj.month, dateTimeObj.year)
        todaytime = "{}:{}:{}".format(dateTimeObj.hour, dateTimeObj.minute, dateTimeObj.second)
        
        #it writes an info file with information about the duration and the parameters
        with open(folder + '/info.txt', 'w') as f:
            f.write("Computation finished in date {} {}, time elapsed: {}\n".format(todaydate, todaytime, duration))

            info_text = "Qaoa parameters: method: {}, n_steps: {}, p_levels: {}, mixing matrix: {}".\
                format(sgd_method, grad_desc_opts[0], p, "X Pauli" if self.mix_matrix == qt.sigmax() else "Y Pauli")
            if(self.state > 0): info_text += ", Use QAOA gs: {}".format(qaoa_gs)
            info_text += "\n"

            f.write(info_text)

        print("Data stored in {}".format(folder))

    # this function uses the diagonalization methods to calculate eigenenergies and eigenvectors of the hamiltonians
    def find_optimal_distance_diag(self):

        print("Starting the calculation for the molecule {} with diagonalization method".format(util.getMoleculeName(self.atom1, self.atom2)))

        folder = self.general_main_folder + "_" + self.mix_matrix_name + "/diag"
        Path(folder).mkdir(parents=True, exist_ok=True)

        # here are defined the array of internuclear distances, eigenenergies and the eigenstates
        
        y = []
        final_states = []

        start = time.perf_counter()

        for distance in self.q_chem.distances:

            ham = self.q_chem.calc_reduced_ham(distance)

            #the eigenenergy and the eigenstate are extracted from the hamiltonian
            energy = ham.eigenenergies()[self.state]
            final_state = list(ham.eigenstates()[1][self.state].dag().full()[0])
            y.append(energy)
            final_states.append(final_state)
                
            #data are rounded in order to make a prittier print
            rounded_distance = math.trunc(np.round(distance * 100))/100
            rounded_distance_bohr = math.trunc(rounded_distance * util.ANG_TO_BOHR * 100)/100
            print("energy: {} a.u., at distance {} A, {} Bohr".format(energy, rounded_distance, rounded_distance_bohr))
            
        duration = time.perf_counter() - start
        print("Algorithm executed in: {:.4f} s".format(duration))
        
        self.fid.plot_save_energy(y, final_states, folder = folder)

    def createAverageData(self, p, calc_variance = True):

        folder = self.general_main_folder + "_" + self.mix_matrix_name + "/qaoa_p{}".format(p)
        diag_folder = self.general_main_folder + "_" + self.mix_matrix_name + "/diag"
        
        average_folder = folder + "/average"
        Path(average_folder).mkdir(parents=True, exist_ok=True)

        instances = self.getNewInstances(folder, get_all_insts = True)
        while(0 in instances): instances.remove(0)
        qaoa_instance_folders = [folder + "/{}".format(instance) for instance in instances]

        diag_final_states_array, _ = util.read_eigenstates_data([diag_folder])
        diag_final_states = diag_final_states_array[0]
        qaoa_instance_final_states_list, _ = util.read_eigenstates_data(qaoa_instance_folders)

        qaoa_eigenvalues_instance_list, _ = util.read_eigenvalues_data(qaoa_instance_folders)

        best_qaoa_final_state = [None] * len(diag_final_states)
        max_fidelity = [0] * len(diag_final_states)

        # An average of the eigenvalues for different instances has calculated
        eigenvalues_average = sum(np.array(qaoa_eigenvalues_instance_list)) / len(instances)
        variance_list = None

        if(calc_variance):
            variance_list = []
            
            for i, eigenvalue_fixed_dist in enumerate(np.array(qaoa_eigenvalues_instance_list).T):
                tot = []
                for eigenvalue in eigenvalue_fixed_dist:
                    tot.append( (eigenvalue - eigenvalues_average[i])**2 )

                variance_list.append( np.sqrt(sum(np.array(tot))/len(eigenvalue_fixed_dist)) )

        #it find the eigenvector calculated with QAOA with the best fidelity for a fixed distance
        for qaoa_instance_final_states in qaoa_instance_final_states_list:

            for i, final_state in enumerate(qaoa_instance_final_states):

                fidelity = np.abs(final_state.overlap(diag_final_states[i])) ** 2
                if(fidelity > max_fidelity[i]):
                    
                    max_fidelity[i] = fidelity
                    best_qaoa_final_state[i] = list(final_state.full()[0])

        self.fid.plot_save_energy(eigenvalues_average, best_qaoa_final_state, variance_list = variance_list, folder = average_folder, p = p)

        #creation of the info.txt file with all the possible information about the instances that have created the average data
        with open(average_folder + "/info.txt", 'w') as file_average:
            file_average.write("instance:{}\n\n".format(''.join([',{}'.format(num) for num in instances])))

            for num, qaoa_file_path in zip(instances, qaoa_instance_folders):
                with open(qaoa_file_path + '/info.txt', 'r') as f:
                    lines = f.readlines()

                    file_average.write("instances n.{}\n".format(num))
                    for line in lines: file_average.write(line)
                    file_average.write("\n")

        print("The average has been calculated using {} instances with p={}, the data has been stored in {}".\
            format(len(instances), p, average_folder))

    def getNewInstances(self, folder, get_all_insts = False):
        
        Path(folder).mkdir(parents=True, exist_ok=True)
        nums = [ int(name) if(name.isnumeric()) else 0 for name in os.listdir(folder)]
        
        if(get_all_insts): return nums

        i = 0
        while(True):
            i += 1
            if not i in nums:
                return i
