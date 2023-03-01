import util
import numpy as np

np.set_printoptions(precision=5)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

main_folder = util.getFolderName(0, "Li", "H")
diag_file_path = main_folder + '_sigmay/diag'
qaoa_file_paths = main_folder + "_sigmay/qaoa_p5/average"

eigenstates_p, distances = util.read_eigenstates_data([diag_file_path, qaoa_file_paths])

accepted_dist = [1.13383559316, 1.98421228803, 2.8345889828999997, 4.25188347435, 6.14160946295]

for eigenstates, p in zip(eigenstates_p, [0, 5]):
    print(p)
    for eigenstate, distance in zip(eigenstates, distances):

        if(distance in accepted_dist):
            print(distance, "\n", np.array(eigenstate), "\n\n")