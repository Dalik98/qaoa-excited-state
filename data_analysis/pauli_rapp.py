import qutip as qt
import numpy as np
from numpy import kron
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MultipleLocator

import util
from qaoa import Qaoa
from qu_chemistry import qu_chemistry


np.set_printoptions(precision=5)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

#Tensor product among the matrix in input and unitary matrixes proportionally to the number of qubits
def full_mat(self, matrix, qubit_pos):

    if qubit_pos < 0 or qubit_pos > self.n_qubits-1:
        raise ValueError('qubit position must be > 0 or < n_qubits-1, but is {}'.format(qubit_pos))

    return qt.tensor([qt.qeye(2)]*qubit_pos + [matrix] + [qt.qeye(2)]*(self.n_qubits-qubit_pos-1))

def HS(M1, M2):
    """Hilbert-Schmidt-Product of two matrices M1, M2"""
    return (np.dot(M1.conjugate().transpose(), M2)).trace()

def c2s(c):
    """Return a string representation of a complex number c"""
    if c == 0.0:
        return "0"
    if c.imag == 0:
        return "%g" % c.real
    elif c.real == 0:
        return "%gj" % c.imag
    else:
        return "%g+%gj" % (c.real, c.imag)

def decompose(H):

    coeff_array = []
    tensor = []

    """Decompose Hermitian 4x4 matrix H into Pauli matrices"""
    sx = np.array([[0, 1],  [ 1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j],[1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0],  [0, -1]], dtype=np.complex128)
    id = np.array([[1, 0],  [ 0, 1]], dtype=np.complex128)
    S = [id, sx, sy, sz]
    labels = ['I', 'X', 'Y', 'Z']
    for i in range(4):
        for j in range(4):
            # if(i == 1 or j == 1): continue
            label = labels[i] + ' $\otimes$ ' + labels[j]
            a_ij = 0.25 * HS(kron(S[i], S[j]), H)
            
            if a_ij != 0.0:
                print("%s\t*\t( %s )" % (c2s(a_ij), label))
                pass

            coeff_array.append(a_ij.real)
            tensor.append(label)

    return coeff_array, tensor

def coeff_vs_dist(atom1, atom2):
    q_chem = qu_chemistry(n_qubits = 4, atom1=atom1, atom2=atom2, temp = True)

    energies = []
    coeff_list = []
    tensor_list = []
    for distance in [4]:

        mat = q_chem.calc_reduced_ham(distance)
        print("Creation of the pauli rappresentation for the hamiltonian at distance:{:0.2f}".format(distance))
        # print(mat)
        # for state, energy in zip(mat.eigenstates()[1], mat.eigenenergies()):
        #     print(energy)
        #     print(state, "\n\n")
        numpy_mat = np.array(mat)

        coeffs, tensor = decompose(np.array(numpy_mat))
        coeff_list.append(coeffs)
        tensor_list.append(tensor)

    # print(coeff_list[0], "\n\n\n")
    # print(coeff_list, "\n\n\n")
    coeff_list = [list(i) for i in zip(*coeff_list)]

    tensor_temp = []
    fig, ax = plt.subplots(1,1)
    
    for i, (coeff, tensor) in enumerate(zip(coeff_list, tensor_list)):

        if(np.all((np.array(coeff).round(5)) == 0)): continue
        plt.plot(q_chem.distances, coeff, '.', label=tensor[i])
        tensor_temp.append(tensor[i])

    ax.set_xlabel("Internuclear distance (Ang)")
    ax.set_ylabel("Coefficents")
    ax.set_title("Coefficients of pauli representation vs internuclear distance")

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_ylim([-1.5, 3])

    plt.legend()
    plt.savefig(util.getFolderName(atom1 = atom1, atom2 = atom2) + '/extra/coeff_vs_dist.png')
    print("File created: {}".format(util.getFolderName(atom1 = atom1, atom2 = atom2) + '/extra/coeff_vs_dist.png'))
