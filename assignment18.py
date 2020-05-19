#!/usr/bin/env python
from assignment8 import StressStrainConverter
import numpy as np
import scipy.integrate

from PyTrilinos import Epetra

class EpetraParallelToughness(StressStrainConverter):

    def __init__(self, filename, comm):
        super().__init__(filename)
        
        self.comm = comm
        self.rank = comm.MyPID()
        self.size = comm.NumProc()

        if self.rank == 0:
            self.convert_to_true_stress_and_strain()
        else:
            self.true_stress = np.array([], dtype=np.double)
            self.true_strain = np.array([], dtype=np.double)

        
    def compute_toughness(self):
        
        my_toughness = scipy.integrate.trapz(self.true_stress, self.true_strain)

        return self.comm.SumAll(my_toughness)


if __name__ == "__main__":

    from PyTrilinos import Epetra

    comm = Epetra.PyComm()

    T = EpetraParallelToughness('data.dat', comm)

    if comm.MyPID() == 0:
        print(T.compute_toughness())
