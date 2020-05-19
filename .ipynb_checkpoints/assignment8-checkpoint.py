
# coding: utf-8

# # Assignment 8
# 
# This repository contains a file `data_1.dat` that is actual tabular data from a stress-strain test conducted in a standard test frame (MTS machine).  The columns of data have the following format
# 
# |time|Axial Displacement|Axial (engineering) Strain| Axial Force |
# |:-:|:-:|:-:|:-:|
# | (s) | (in) | (in/in) | (lbf) |
# 
# Complete the function definition below for the function named `convert_to_true_stress_and_strain()`.  This function should take the name of a file (e.g. `data_1.dat`) as an argument and return a 2-tuple ($\varepsilon_T$, $\sigma_T$), where $\varepsilon_T$ is the *true* strain and $\sigma_T$ is the *true* stress.
# 
# Recall that the definition of engineering stress is
# 
# $$
# \sigma_E = \frac P A_o
# $$
# 
# where $P$ is the axial force and $A_o$ is the original cross-sectional area of the sample.  The cross-sectional area can be computed from the width and thickness dimensions that are stored on the 3rd line of the header of the MTS data file.  Look for `Wo=` and `Thicko=`.

# Your function should parse these values out computationally (not manually) because we expect them to be in every MTS header the same way and woud like the function to work on multiple data files.
# 
# The conversion between engineering $\varepsilon_E$ and true strain is
# 
# $$
# \varepsilon_T = \ln(1+\varepsilon_E).
# $$
# 
# The conversion between engineering and true stress is
# 
# $$
# \sigma_T = \sigma_E (1+\varepsilon_E).
# $$
# 
# Use Numpy data structures and broadcasted operations to make your code compact and efficient.
# 
# ## Please Note
# 
# You may write your code directly in the Jupyter Notebook below or convert the notebook to a plain Python file with the command
# 
# ```bash
# jupyter nbconvert --to python assignment7.ipynb
# ```
# 
# which will create a file `assignment7.py`.  If you commit this file to the repository, only the Python file will be tested (not the Jupyter notebook).  However, if you do not commit this file into the repository, the Jupyter notebook will be converted automatically on Travis before testing.  

# In[1]:


import numpy as np
import linecache
import scipy.integrate

class StressStrainConverter():
    
    def __init__(self, filename):
        
        self.filename = filename
        

    def extract_dimensions(self):

        line = linecache.getline(self.filename, 3).split('=')

        self.width = float(line[1].split('"')[0])

        self.thickness = float(line[2].split('"')[0]) 

        return
    

    def convert_to_true_stress_and_strain(self):
        
        self.extract_dimensions()

        eng_strain, force = np.loadtxt(self.filename, skiprows=5, usecols=(2,3)).T 

        self.true_strain = np.log(1 + eng_strain)

        self.true_stress = force / self.width / self.thickness * (1 + eng_strain)

        return 
    
    
class Toughness(StressStrainConverter):
    
    def compute_toughness_simps(self):
        
        self.convert_to_true_stress_and_strain()
        
        return scipy.integrate.simps(self.true_stress, self.true_strain)
    
    def compute_toughness_trapz(self):
        
        self.convert_to_true_stress_and_strain()
        
        return scipy.integrate.trapz(self.true_stress, self.true_strain)

