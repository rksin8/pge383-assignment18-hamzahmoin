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

        unbalanced_map = Epetra.Map(-1, self.true_stress.shape[0], 0, self.comm)
        #print(unbalanced_map)
        unbalanced_stress = Epetra.Vector(Epetra.Copy, unbalanced_map, self.true_stress)
        unbalanced_strain = Epetra.Vector(Epetra.Copy, unbalanced_map, self.true_strain)
        
        balanced_map = self.create_balanced_map(unbalanced_map)
        #print(balanced_map)
        self.true_stress = Epetra.Vector(balanced_map)
        self.true_strain = Epetra.Vector(balanced_map)
        
        importer = Epetra.Import(balanced_map, unbalanced_map)
        
        self.true_stress.Import(unbalanced_stress, importer, Epetra.Insert)
        self.true_strain.Import(unbalanced_strain, importer, Epetra.Insert)
        
    def create_balanced_map(self, unbalanced_map):
        
        temp_map = Epetra.Map(unbalanced_map.NumGlobalElements(), 0, self.comm)
        
        my_global_element_list = temp_map.MyGlobalElements()
        
        if self.rank < (self.size-1):
            my_global_element_list = np.append(my_global_element_list, [temp_map.MaxMyGID() + 1])
            
        return Epetra.Map(-1, list(my_global_element_list), 0, self.comm)
        
        
    def compute_toughness(self):
        
        my_toughness = scipy.integrate.trapz(self.true_stress, self.true_strain)

        return self.comm.SumAll(my_toughness)


if __name__ == "__main__":

    from PyTrilinos import Epetra

    comm = Epetra.PyComm()

    T = EpetraParallelToughness('data.dat', comm)

    if comm.MyPID() == 0:
        print(T.compute_toughness())
