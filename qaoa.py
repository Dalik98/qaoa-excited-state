import qutip as qt
import math
import numpy as np
import scipy
from scipy.optimize import basinhopping
import random as rm
import scipy.linalg as sl

import optimization as opt
from optimization import Optimizer



#Class designed in order to perform the QAOA algorithm
class Qaoa:

    def __init__(self, n_qubits, prob_ham = None, mixing_matrix = qt.sigmax()):

        #variable used for test purposes
        self.temp = []

        #Information needed for the QAOA algorithm
        self.n_qubits = n_qubits
        self.prob_ham = prob_ham
        self.mixing_matrix = mixing_matrix

        self.mixed_ham = self.create_mixed_ham()
        self.init_state = self.create_init_state()

        #Data calculated at the end of the algorithm
        self.final_state = []
        self.fp = None
        self.all_params = []
    
    #Tensor product among the matrix in input and unitary matrixes proportionally to the number of qubits
    def full_mat(self, matrix, qubit_pos):

        if qubit_pos < 0 or qubit_pos > self.n_qubits-1:
            raise ValueError('qubit position must be > 0 or < n_qubits-1, but is {}'.format(qubit_pos))

        return qt.tensor([qt.qeye(2)]*qubit_pos + [matrix] + [qt.qeye(2)]*(self.n_qubits-qubit_pos-1))

    #The init state for the QAOA algorithm
    def create_init_state(self):

        init_state = ( qt.basis(2,0) + qt.basis(2,1)).unit()
        init_state = qt.tensor([init_state]*self.n_qubits)
        
        return init_state
    
    def create_mixed_ham(self):

        list_n_sigmax = [self.full_mat(self.mixing_matrix, i) for i in range(self.n_qubits)]

        return sum(list_n_sigmax)

    def set_ham(self, ham, eigenstate = None, lambda_ = 0):

        if(eigenstate != None):
            proj_gs = qt.Qobj(eigenstate.proj(), dims = ham.dims)
            self.prob_ham = ham + lambda_ * proj_gs
        else:
            self.prob_ham = ham

        #reset of calculated data
        self.fp = 0
        self.final_state = []
        self.all_params = []

    def evolution_operator(self, gammas, betas):

        evol_oper = qt.tensor([qt.qeye(2)] * self.n_qubits)
        print(evol_oper)

        for i in range(0, len(gammas)):

            u_mix_hamilt_i = (-complex(0, betas[i])*self.mixed_ham).expm()
            print(u_mix_hamilt_i)
            u_prob_ham_i = (-complex(0, gammas[i])*self.prob_ham).expm()
            print(u_prob_ham_i)
            evol_oper = u_mix_hamilt_i * u_prob_ham_i * evol_oper

        return evol_oper

    def evolution_operator_scipy(self, gammas, betas):

        evol_oper = np.array ( qt.tensor([qt.qeye(2)] * self.n_qubits) )

        mixed_ham_np = np.array(self.mixed_ham)
        prob_ham_np = np.array(self.prob_ham)

        for i in range(0, len(gammas)):

            u_mix_hamilt_i = sl.expm( (-complex(0, betas[i])*mixed_ham_np) )
            u_prob_ham_i = sl.expm( (-complex(0, gammas[i])*prob_ham_np)) 
            evol_oper = np.matmul( u_mix_hamilt_i, ( np.matmul(u_prob_ham_i, evol_oper)))

        return qt.Qobj( evol_oper, dims = [[2]*self.n_qubits, [2]*self.n_qubits])

    def evaluate_final_state(self, params, zero_phase = False):

        gammas = params[:int(len(list(params))/2)]
        betas = params[int(len(list(params))/2):]

        final_state = self.evolution_operator_scipy(gammas, betas) * self.init_state

        if zero_phase:
            #search the greatest coefficent with the biggest real part
            biggest_num = final_state[0][0][0]

            for i in range(1, len(final_state.dag()[0][0])):

                if(abs(biggest_num) < abs(final_state[i][0][0])):
                    biggest_num = final_state[i][0][0]

            phase = np.arctan2(biggest_num.imag, biggest_num.real)
            exp = qt.Qobj(complex(0, -phase)).expm()
            final_state = exp * final_state

        return final_state

    def evaluate_F_p(self, params):
        
        #obtain final state
        fin_state = self.evaluate_final_state(params)

        fp = fin_state.dag() * self.prob_ham * fin_state
        
        return fp[0][0][0]

    def optimizeGrid(self, method_params = [50, True, True]):

        opt_params = opt.perform_grid(self.evaluate_F_p, *method_params)

        self.final_state = self.evaluate_final_state(opt_params, zero_phase = True)
        self.fp = self.evaluate_F_p(opt_params)

        return opt_params

    #Launch the optimization selected
    #Possible optimization are: Adam, SGD and Grid
    #Is possible to pass at the function the parameters for the optimizers
    #Return the optimal parameters founded
    def optimizeAdam(self, p, method_params, show_fp_plot = True):

        if(self.prob_ham == None):
            print("hamiltonian is not setted")
            return

        adamOpt = Optimizer()
        opt_params, self.all_params = adamOpt.perform_SGD_adam(self.evaluate_F_p, *method_params)

        self.final_state = self.evaluate_final_state(opt_params, zero_phase = True)
        self.fp = self.evaluate_F_p(opt_params)

        return opt_params

    def optimizeBasinHopper(self, p, niter = 10):

        if(self.prob_ham == None):
            print("hamiltonian is not setted")
            return

        parameters = np.random.rand(2*p) * np.pi - np.pi

        res = basinhopping(self.evaluate_F_p, parameters, niter, minimizer_kwargs={"method": "L-BFGS-B"})
        opt_params = res.x

        self.final_state = self.evaluate_final_state(opt_params, zero_phase = True)
        self.fp = self.evaluate_F_p(opt_params)

        return opt_params

    def createPlots(self, create_fp_plot = True, create_devstd_plot = True, create_prob_plot = True, fp_path = "", devstd_path = ""):
        
        #calculate all fp for all params obtained, and show it in function of the steps
        if(create_fp_plot):
            fp_array = []
            for param in self.all_params: fp_array.append(self.evaluate_F_p(param))
        
            pl.plot_fp(fp_array, fp_path)
        
        # calculate all the final states for all the params in order to find the standard deviation in function of steps
        if(create_devstd_plot):
            std_dev_array = []
            self.prob_ham_square = self.prob_ham ** 2

            for param in self.all_params:

                final_state = self.evaluate_final_state(param)
                std_dev_square = final_state.dag() * self.prob_ham_square * final_state - abs((final_state.dag() * self.prob_ham * final_state)) ** 2
                std_dev_array.append(np.sqrt( abs(std_dev_square[0][0][0])))
            pl.plot_std_dev(std_dev_array, devstd_path)

        #calculate probabilities of the possible states and show it
        if(create_prob_plot):
            num_fin_state = np.array(self.final_state)

            probs = []
            for x in range(len(num_fin_state)):
                prob = abs( num_fin_state[x] ) ** 2
                probs.append( prob[0] )

            pl.show_states_dist(probs)

    def calc_IRC(self):

        IRC = 0

        num_fin_state = np.array(self.final_state)

        for x in range(len(num_fin_state)):
            IRC += abs( num_fin_state[x] ) ** 4

        return IRC
