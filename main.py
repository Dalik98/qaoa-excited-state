import qutip as qt
import argparse
import numpy as np
import warnings 

from  data_analysis import correlations
from  data_analysis import pauli_rapp
from  data_analysis.fidelity import Fidelity
from calculations import CalcHandler
from qu_chemistry import qu_chemistry
import data_analysis.correlations as corr

np.set_printoptions(precision=5)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

warnings.filterwarnings("ignore")

def start_algorithm(method = 'qaoa', state = 0, p_lst = list(range(1, 7 + 1)), qaoa_gs = None, mix_matrix = qt.sigmax(), \
    sgd_method = 'basin', grad_desc_opts = None, atom1 = "H", atom2 = "Li"):

    ch = CalcHandler(state, atom1 = atom1, atom2 = atom2, mix_matrix = mix_matrix)

    if(method == 'diag'):

        ch.find_optimal_distance_diag()
    elif(method == 'qaoa'):

        for p in p_lst:
            ch.find_optimal_distance_qaoa(p, qaoa_gs, sgd_method, grad_desc_opts)
    elif(method == 'average'):

        for p in p_lst:
            ch.createAverageData(p)

#Argparse settings
parser = argparse.ArgumentParser()
parser.add_argument("method", type=str)
parser.add_argument('-p', '--pvalues', action='store', default="0", help='what p values use in the computation, default: 1to5')
parser.add_argument('-m', '--mixing', action='store', default="x", help='specify the mixing matrix to be used in QAOA,' + 
" possible values: x, y, z, default: x")
parser.add_argument('-t', '--times', type=int, action='store', default=1, help='how many times perform the same algorithm, default: 1')
parser.add_argument('-s', '--state', type=int, action='store', default=0, help='the eigenstate to evaluate,' +
' 0 corresponds to the ground state, 1 to the first excited state and so on, default: 0')
parser.add_argument('-a1', '--atom1', type=str, action='store', default="Li", help='Specify an atom of the diatomic molecule')
parser.add_argument('-a2', '--atom2', type=str, action='store', default="H", help='Specify an atom of the diatomic molecule')
args = parser.parse_args()

state = args.state
qaoa_gs = True if state == 1 else None

mixing_dict = {'x': qt.sigmax(), 'y': qt.sigmay(), 'z': qt.sigmaz()}

p_lst = list(range(1, 5 + 1))
# p_lst = [5]

if(args.pvalues[1:3] == 'to'): 
    p_lst = [num for num in range(int(args.pvalues[0]), int(args.pvalues[3]) + 1)]
elif(not args.pvalues == "0"):
    p_lst = [int(ch) for ch in args.pvalues]

#iteration for the basin hopping method
basin_opts = [10]
#iteration for the adam method
adam_opts = [200]

atoms = [args.atom1, args.atom2]

diag_param = ['diag', state, None, None, mixing_dict[args.mixing], None, None] + atoms
qaoa_param_gs_basin = ['qaoa', state, p_lst, qaoa_gs, mixing_dict[args.mixing], 'basin', basin_opts] + atoms
qaoa_param_gs_adam = ['qaoa', state, p_lst, qaoa_gs, mixing_dict[args.mixing], 'adam', adam_opts] + atoms
qaoa_param_gs_average = ['average', state, p_lst, qaoa_gs, mixing_dict[args.mixing], None, None] + atoms

if(args.method == 'corr'):

    fid = Fidelity(state = state, mix_mat = mixing_dict[args.mixing], atom1 = atoms[0], atom2 = atoms[1])
    fid.plot_energy_from_file(p_values = p_lst + [0], atoms = atoms)
    fid.get_fidelity(p_lst, create_plots = True)
    fid.get_fidelity_min_value(p_lst, mode = "fidelity")
    fid.get_fidelity_min_value(p_lst, mode = "energies")

elif(args.method == 'basin'):

    for _ in range(args.times):
        start_algorithm(*qaoa_param_gs_basin)

elif(args.method == 'diag'):

    for _ in range(args.times):
        start_algorithm(*diag_param)

elif(args.method == 'average'):
    start_algorithm(*qaoa_param_gs_average)

elif(args.method == 'allstate'):
    corr.create_all_state_plot(*atoms)

elif(args.method == 'pauli'):
    pauli_rapp.coeff_vs_dist(*atoms)

