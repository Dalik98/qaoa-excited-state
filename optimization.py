import numpy as np
import multiprocessing
import threading
from queue import Queue
from scipy.optimize import basinhopping

class Optimizer:

    def __init__(self):
        self.func = None
        self.increment = 0

    #Perform the normal SGD on the function passed
    #Return the optimal parameters
    def perform_SGD_adam(self, func, n_steps = 150, p_levels = 10, eta = 0.005, increment = 0.01, decay = 0.05, minimum = True):
        
        self.func = func
        self.increment = increment

        sign = 1 if minimum else -1

        parameters = np.random.rand(2*p_levels) * np.pi - np.pi

        all_params = []

        #parameters for the exponential decay
        initial_eta = eta

        #parameters for the adam optimizer
        beta_1 = 0.8
        beta_2 = 0.999
        epsilon = 1e-8
        m_t = np.zeros(2*p_levels)
        v_t = np.zeros(2*p_levels)
        t = 0

        #with this try is possible to end the cycle by keyboard whenever one wants
        try:
            for _ in range(n_steps):
                
                g_t = self.fin_diff_grad(parameters)

                # if((t + 1) % 10 == 0):
                #     id_th = threading.get_ident()
                #     print("Thread {}, Fp after step {:5d}: {: .13f}".format(id_th, t + 1, func(parameters).real), flush=True)
                all_params.append(parameters)

                t += 1
                m_t = beta_1*m_t + sign * (1-beta_1)*g_t
                v_t = beta_2*v_t + (1-beta_2)*g_t**2
                m_t_hat = m_t/(1-beta_1**t)
                v_t_hat = v_t/(1-beta_2**t)
                delta_params = eta*m_t_hat/(np.sqrt(v_t_hat) + epsilon)

                parameters = parameters - delta_params
                
        except KeyboardInterrupt:
            print('Interrupted from keyboard\n')

        return parameters, all_params

    #Define a function that evaluate the gradient estimator g_t
    def fin_diff_grad(self, params):

        d = len(list(params))
        a_params = np.array(params)
        g_t = np.zeros(d).astype(complex)

        for i in range(d):
            e_i = np.zeros(d)
            e_i[i] = 1.0
            g_t[i] = (self.func(a_params+e_i * self.increment) - self.func(a_params-e_i * self.increment)) / (2*self.increment)

        return g_t

#Try to find the minimum/maximum of the function creting a grid, this method use only 2 parameters
#The parameter div describe the resolution for the grid, the point of the grid are equal to div^2
#Return the optimal parameters
def perform_grid(func, div = 20, find_min = True, show = True):
    
    sign = 1 if find_min else -1

    beta_list = np.linspace(-np.pi/2, np.pi/2, div)
    gamma_list = beta_list.copy()
    fp_list = []

    minimum = 200

    opt_params = [0, 0]

    for beta in beta_list:

        for gamma in gamma_list:
            
            parameters = [gamma, beta]
            fp = func(parameters)
            fp_list.append(fp)
            
            # print("gamma: {:.3f}, beta: {:.3f}, Fp: {:.6f}".format(gamma, beta, fp))

            if sign * minimum > fp: 
                print(fp)
                minimum = fp
                opt_params = [gamma, beta]
    
    pl.plot_surface(beta_list, gamma_list, fp_list)

    return opt_params

### NOT USED IN THE CODE
#Perform the normal SGD on the function passed
#Return the optimal parameters
def perform_SGD_const(func, n_steps = 20, p_levels = 1, eta = 0.01, increment = 0.01, minimum = True):
    # perform M-steps for expectation value of the cost-function

    sign = 1 if minimum else -1

    #parameters = np.array(0.5*np.random.random_sample(2*n_levels))
    parameters = np.random.rand(2*p_levels) * 2*np.pi - np.pi
    
    for i in range(n_steps):
        g_t = fin_diff_grad(func, parameters, increment)
        parameters = parameters - sign * eta * g_t
        # if (i + 1) % 1 == 0:
        #         print('in', parameters, 'with grad ', g_t,  "objective after step {:5d}: {: .7f}".format(i + 1, self.evaluate_F_p(parameters)))
    optimal_parameters = parameters
    
    return optimal_parameters



