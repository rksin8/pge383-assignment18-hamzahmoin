#!/usr/bin/env python
import os
import unittest
from PyTrilinos import Epetra

from assignment18 import EpetraParallelToughness

class TestEpetraParallelToughness(unittest.TestCase):

    def setUp(self):

        comm = Epetra.PyComm()
        self.T = EpetraParallelToughness('data.dat', comm)

    def test_compute_toughness(self):

        toughness = self.T.compute_toughness()

        self.assertAlmostEqual(toughness, 70836.14348345132, 2)


if __name__ == '__main__':
    unittest.main()
