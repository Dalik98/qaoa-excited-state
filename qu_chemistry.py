import numpy as np
import qutip as qt
import openfermion as of
from openfermion.chem import MolecularData
from openfermionpsi4 import run_psi4
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.linalg import get_sparse_operator
from openfermion.ops import FermionOperator
from openfermion.ops import QubitOperator

import util

class qu_chemistry:

    def __init__(self, n_qubits, atom1, atom2, temp = False):

        self.n_qubits = n_qubits
        self.atom1 = atom1
        self.atom2 = atom2

        self.temp = temp

        dim = 4
        self.two_ferm_tensor = np.zeros((dim, dim)).tolist()
        self.four_ferm_tensor = np.zeros((dim, dim, dim, dim)).tolist()

        self.molecule_name = util.getMoleculeName(atom1, atom2)
        self.distances = util.NUCLEAR_DISTS[self.molecule_name]
        self.one_qubit_simplification = True

        self.operatorsProduct(dim)

    def operatorsProduct(self, dim):
        for x in range(dim):
            for y in range(dim):
                for z in range(dim):
                    for w in range(dim):
                        oper = self.destroy_oper(x).dag()*self.destroy_oper(y).dag()*self.destroy_oper(z)*self.destroy_oper(w)
                        self.four_ferm_tensor[x][y][z][w] = oper

        for x in range(dim):
            for y in range(dim):
                oper = self.destroy_oper(x).dag()*self.destroy_oper(y)
                self.two_ferm_tensor[x][y] = oper

    def molecule_data(self, distance):

        filename="openfermion_data/{}/{}".format(util.getMoleculeName(self.atom1, self.atom2), distance)
        geometry = [(self.atom1, (0, 0, 0)), (self.atom2, (0, 0, distance))]
        multiplicity = 1
        basis = "sto-3g"
        charge = 0
        molecule = MolecularData(geometry, basis, multiplicity, charge, filename=filename)

        if(self.temp): 
            molecule = run_psi4(molecule, run_mp2 = True, run_cisd = True, run_ccsd = True, run_fci = True, delete_input = False)
            print("New hdf5 file created for the molecule {} at distance {}".format(self.molecule_name, distance))
        else: molecule.load()

        # Set Hamiltonian parameters.
        active_space_start = 1
        active_space_stop = int(molecule.n_orbitals) // 2

        if(not active_space_stop == active_space_start):
            mol_ham = molecule.get_molecular_hamiltonian(occupied_indices=range(active_space_start),\
                active_indices=range(active_space_start, active_space_stop))
            
        else:
            mol_ham = molecule.get_molecular_hamiltonian()

        nuclear_repulsion = mol_ham.constant
        one_body_integrals = mol_ham.one_body_tensor
        two_body_integrals = mol_ham.two_body_tensor

        # one_body_integrals = h2_molecule.one_body_integrals
        # two_body_integrals = h2_molecule.two_body_integrals
        # nuclear_repulsion = h2_molecule.nuclear_repulsion

        return nuclear_repulsion, one_body_integrals, two_body_integrals

    def sec_quant_ham(self, distance):

        self.n_qubits = 4

        #definition of the one body operator and two body operator
        one_body = np.zeros((4, 4))
        two_body = np.zeros((4, 4, 4, 4))
        nuclear_repulsion = 0

        nuclear_repulsion, one_body, two_body = self.molecule_data(distance)

        #calculation of the part of the hamiltonian with one body operator
        first_part = []
        for x in range(4):
            for y in range(4):

                calc = one_body[x][y]*self.two_ferm_tensor[x][y]
                first_part.append(calc)

        first_part = sum(first_part)

        #calculation of the part of the hamiltonian with two body operator
        second_part = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    for w in range(4):
                        calc = two_body[x][y][z][w]*self.four_ferm_tensor[x][y][z][w]
                        second_part.append(calc)

        second_part = sum(second_part)

        ham = first_part + second_part + nuclear_repulsion

        np_ham = ham.full()

        ham = qt.Qobj(np_ham)

        self.n_qubits = 4

        return ham

    def calc_reduced_ham(self, distance):

        ham = self.sec_quant_ham(distance)
        ham = qt.Qobj(ham, dims = [[2, 2, 2, 2], [2, 2, 2, 2]])

        #Applying the projection for the selection of a fixed number of electron
        proj = self.proj_2el(opposite_spin=True)

        def swap(matrix, row1, row2, swap_row = True):

            if(not swap_row): matrix = matrix.dag()

            np_matrix = matrix.full()

            temp = list(np_matrix[row1])
            np_matrix[row1] = list(np_matrix[row2])
            np_matrix[row2] = temp

            final_matrix = qt.Qobj(np_matrix, dims = [[2, 2, 2, 2], [2, 2, 2, 2]])

            if(not swap_row): final_matrix = final_matrix.dag()

            return final_matrix
        
        ham = proj.dag() * ham * proj

        ham = swap(ham, 3, 7)
        ham = swap(ham, 12, 8)
        ham = swap(ham, 3, 7, swap_row = False)
        ham = swap(ham, 12, 8, swap_row = False)

        ham = swap(ham, 6, 7)
        ham = swap(ham, 8, 9)
        ham = swap(ham, 6, 7, swap_row = False)
        ham = swap(ham, 8, 9, swap_row = False)

        # print(ham)

        #shifting of the hamiltonian in 2 blocks
        shift_plus = self.shift_operator()
    
        ham = shift_plus.dag() * ham * shift_plus + shift_plus * ham * shift_plus.dag()

        #reduction of dimensionality
        self.n_qubits -= 1

        mat = []
        for i in range(2**self.n_qubits):
            mat.append(ham.full()[i][:2**self.n_qubits])

        ham = qt.Qobj(mat, dims = [[2]*self.n_qubits, [2]*self.n_qubits])

        # applying the shift operator again for dimensionality reduction
        shift_plus = self.shift_operator()
        ham = shift_plus.dag() * ham * shift_plus + shift_plus * ham * shift_plus.dag()

        #reduction of dimensionality
        self.n_qubits -= 1

        mat = []
        for i in range(2**self.n_qubits):
            mat.append(ham.full()[i][:2**self.n_qubits])

        ham = qt.Qobj(mat, dims = [[2]*self.n_qubits, [2]*self.n_qubits])

        if(self.one_qubit_simplification):

            ham = swap(ham, 1, 3)
            ham = swap(ham, 1, 3, swap_row = False)

            ham = qt.Qobj(ham, dims = [[2]*self.n_qubits, [2]*self.n_qubits])

        # print(ham)
        # print(ham.eigenenergies())

        # exit()

        return ham

    #Tensor product among the matrix in input and unitary matrixes proportionally to the number of qubits
    def full_mat(self, matrix, qubit_pos):

        if(matrix == 'X'):
            matrix = qt.sigmax()
        elif(matrix == 'Y'):
            matrix = qt.sigmay()
        elif(matrix == 'Z'):
            matrix = qt.sigmaz()

        if qubit_pos < 0 or qubit_pos > self.n_qubits-1:
            raise ValueError('qubit position must be > 0 or < n_qubits-1, but is {}'.format(qubit_pos))

        return qt.tensor([qt.qeye(2)]*qubit_pos + [matrix] + [qt.qeye(2)]*(self.n_qubits-qubit_pos-1))

    def destroy_oper(self, num_orbital):
        
        oper = (self.full_mat('X', num_orbital) + complex(0, 1)*self.full_mat('Y', num_orbital))/2

        for i in range(num_orbital):
            oper = self.full_mat('Z', i) * oper

        return oper

    def number_op(self):

        mat_list = []

        for i in range(0, self.n_qubits):
            operator = self.destroy_oper(i).dag() * self.destroy_oper(i)
            mat_list.append(operator)

        num_op = sum(mat_list)

        return num_op

    def proj_2el(self, opposite_spin = True):

        proj = 1

        if(opposite_spin):
            n_up = self.destroy_oper(0).dag()*self.destroy_oper(0) + self.destroy_oper(2).dag()*self.destroy_oper(2)
            n_down = self.destroy_oper(1).dag()*self.destroy_oper(1) + self.destroy_oper(3).dag()*self.destroy_oper(3)

            proj = n_up * (2 - n_up) * n_down * (2 - n_down)
        else:

            n_tot = sum([self.destroy_oper(i).dag()*self.destroy_oper(i) for i in range(4)])

            N = 2
            for j in range(0, 5):

                piece = (n_tot - j)/(N - j) if j != 2 else 1
                proj = proj * piece

        return proj

    def shift_operator(self):

        shift_op = 1/2 *(self.full_mat('X', 1) + self.full_mat('X', 1)*self.full_mat('X', 0) \
        - complex(0, 1)*self.full_mat('Y', 1) + complex(0,1)*self.full_mat('Y', 1)*self.full_mat('X', 0) )

        shift_op_minus = 1/2 *(self.full_mat('X', 1) + self.full_mat('X', 1)*self.full_mat('X', 0) \
        + complex(0, 1)*self.full_mat('Y', 1) - complex(0,1)*self.full_mat('Y', 1)*self.full_mat('X', 0) )

        return shift_op

    def reorder_op(self):

        reorder_op = 1/2*(1 + self.full_mat('Z', 0)*self.full_mat('Z', 2) \
        - self.full_mat('Z', 0)*self.full_mat('X', 1)*self.full_mat('Z', 2) \
        + self.full_mat('X', 1))

        return reorder_op

    def exchange_indices(self, old_tensor):

        dim = 8

        new_tensor = np.zeros((dim, dim, dim, dim))

        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for l in range(dim):
                        new_tensor[i][j][k][l] = old_tensor[k][j][l][i]
                        print(new_tensor[i][j][k][l] - old_tensor[i][j][k][l])

        # print("new", new_tensor)
        # print("old", old_tensor)

        return new_tensor
