# Funkcje testowe, do pobrania z:
#https://gitlab.com/luca.baronti/python_benchmark_functions
#https://github.com/thieu1995/opfunu/tree/master/opfunu/cec_based
#!/usr/bin/env python3

import math
from scipy.stats import multivariate_normal
import numpy as np
import logging
from abc import ABC, abstractmethod
from . import functions_info_loader as fil

__author__      = 'Luca Baronti'
__maintainer__  = 'Luca Baronti'
__license__     = 'GPLv3'
__version__     = '1.1.3'


# ### <--- BELOW CLASS COPY PASTE FROM LUCA BARONTI REPO --->
class BenchmarkFunction(ABC):
	def __init__(self, name, n_dimensions=2, opposite=False):
		if type(n_dimensions)!=int:
			raise ValueError(f"The number of dimensions must be an int, found {n_dimensions}:{type(n_dimensions)}.")
		if n_dimensions==0:
			raise ValueError(f"Functions can be created only using a number of dimensions greater than 0, found {n_dimensions}.")
		self._name=name
		self._n_dimensions=n_dimensions
		self.opposite=opposite
		self.parameters=[]
		self.info=None

	def load_info(self, force_reload=False):
		if force_reload or self.info is None:
			self.info=fil.FunctionInfo(self._name)
		return self.info

	def __call__(self, point, validate=True):
		if validate:
			self._validate_point(point)
		if type(point) is fil.Optimum:
			point=point.position
		if self.opposite:
			return - self._evaluate(point)
		else:
			return self._evaluate(point)

class Hypersphere(BenchmarkFunction):
	def __init__(self, n_dimensions=2, opposite=False):
		super().__init__("Hypersphere", n_dimensions, opposite)
	def _evaluate(self,point):
		return sum([pow(x,2) for x in point])
	def _evaluate_gradient(self, point):
		return [2.0*x for x in point]
	def _evaluate_hessian(self, point):
		H = np.zeros((self._n_dimensions,self._n_dimensions))
		np.fill_diagonal(H,2.0)
		return H


class Hyperellipsoid(BenchmarkFunction): # rotated hyperellipsoid
	def __init__(self, n_dimensions=2, opposite=False):
		super().__init__("Hyperellipsoid", n_dimensions, opposite)
	def _evaluate(self,point):
		ret = 0.0
		for i in range(self._n_dimensions):
			for j in range(i+1):
				ret += pow(point[j],2)
		return ret


class Rastrigin(BenchmarkFunction):
	def __init__(self, n_dimensions=2, opposite=False):
		super().__init__("Rastrigin", n_dimensions, opposite)
	def _evaluate(self,point):
		ret = sum([pow(p,2) - 10.0*math.cos(2.0*math.pi*p) for p in point]) + 10.0*len(point)
		return ret
