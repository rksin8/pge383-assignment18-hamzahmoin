#!/usr/bin/env python
import numpy as np

from PyTrilinos import Epetra
from PyTrilinos import AztecOO
from PyTrilinos import Teuchos
from PyTrilinos import Isorropia

class OneDimLaplace(object):

    def __init__(self, comm, number_of_elements=10):

        self.comm = comm
        self.rank = comm.MyPID()
        self.size = comm.NumProc()

        if self.rank == 0:
            number_of_rows = number_of_elements
        else:
            number_of_rows = 0
        
        unbalanced_map = Epetra.Map(-1, number_of_rows, 0, self.comm)

        self.A = Epetra.CrsMatrix(Epetra.Copy, unbalanced_map, 3)
        self.x = Epetra.Vector(unbalanced_map) 
        self.b = Epetra.Vector(unbalanced_map) 
        
        print(self.b)
       
        for gid in unbalanced_map.MyGlobalElements():
            if gid == 0: 
                self.A.InsertGlobalValues(gid,[1],[gid])
                self.b[0] = -1
            elif gid == (number_of_elements - 1): 
                self.A.InsertGlobalValues(gid,[1],[gid])
                self.b[-1] = 1
            else: 
                self.A.InsertGlobalValues(gid,[-1,2,-1],[gid-1,gid,gid+1])

        self.A.FillComplete()

    def load_balance(self):

        parameter_list = Teuchos.ParameterList() 
        parameter_sublist = parameter_list.sublist("ZOLTAN")
        parameter_sublist.set("DEBUG_LEVEL", "0")
        partitioner = Isorropia.Epetra.Partitioner(self.A, parameter_list) 
        redistributor = Isorropia.Epetra.Redistributor(partitioner) 
        self.A = redistributor.redistribute(self.A)
        self.x = redistributor.redistribute(self.x)
        self.b = redistributor.redistribute(self.b)
        return

    def solve(self):

        linear_problem = Epetra.LinearProblem(self.A, self.x, self.b) 
        solver = AztecOO.AztecOO(linear_problem) 
        solver.Iterate(10000, 1.e-5) 
        return

    def get_solution(self):
        return self.x

if __name__ == "__main__":

    from PyTrilinos import Epetra

    comm = Epetra.PyComm()

    solver = OneDimLaplace(comm)
    solver.load_balance()
    solver.solve()

    solver.get_solution()
