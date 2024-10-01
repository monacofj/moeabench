from __future__ import annotations

import bisect
import random
from typing import Callable, Generator, List, Tuple

from typing_extensions import override


class Problem:
	"""
	Describes a problem whose objective functions must be minimized.
	"""

	@property
	def objectives(self) -> int:
		"""
		Returns the number of objectives in this test problem.
		"""
		raise NotImplementedError()

	@property
	def params(self) -> int:
		"""
		Returns the number of parameters in this test problem.
		"""
		raise NotImplementedError()

	def range_of(self, param: int) -> List[Tuple[float, float]]:
		"""
		Returns the list of ranges containing acceptable values for the
		parameter at the given index. The returned list must contain one or more
		non-overlapping ranges, in the form of (beg, end) tuples, sorted by
		their beggining value. `-inf` and `inf` are valid range bounds.

		Throws:
		- `IndexError`: If `param` is greater than or equal to the `self.params`.
		"""
		raise NotImplementedError()

	def value_of(self, objective: int, params: List[float]) -> float:
		"""
		Returns the value of the given objective function, when evaluated with
		the given parameter list.

		Throws:
		- `IndexError`: If `objective` is greater than or equal to `self.objectives`.
		- `ValueError`: If `len(params)` is not the same as `self.params`.
		"""
		raise NotImplementedError()

	def optimum_sdf(self, objs: List[float]) -> float:
		"""
		Returns the signed distance between the given point in objective space
		and the Pareto-optimal front.

		Throws:
		- `ValueError`: If `len(objs)` is not the same as `self.objectives`.
		"""
		raise NotImplementedError()


class DTLZ1(Problem):
	"""
	The DTLZ1 family of test problems, defined in Section 6.7.1.
	"""

	def __init__(self, objectives: int, params: int, g: Callable[[List[float]], float] | None = None):
		self._objectives = objectives
		self._params = params
		self._target_g = g if g is not None else self._g

	def _g(self, x: List[float]) -> float:
		# Make sure the length of the vector we've been given corresponds to the
		# value `k`, as required by the definition of DTLZ1.
		assert len(x) == self._params - self._objectives + 1

		from math import cos, pi
		return 100 * (len(x) + sum([(xi - 0.5) ** 2 - cos(20 * pi * (xi - 0.5)) for xi in x]))

	@override
	@property
	def objectives(self) -> int:
		return self._objectives

	@override
	@property
	def params(self) -> int:
		return self._params

	@override
	def range_of(self, param: int) -> List[Tuple[float, float]]:
		if param >= self._params:
			# Not strictly necessary here, but still good not to fail silently.
			raise IndexError(f"requested parameter {param} is outside the valid parameter range 0 <= i < {self._params}")

		# All parameters in DTLZ1 are confined to the range 0 <= x <= 1.
		return [(0, 1)]

	@override
	def value_of(self, objective: int, params: List[float]) -> float:
		if objective >= self._objectives:
			raise IndexError(f"requested objective {objective} is outside the valid objective range 0 <= i < {self._objectives}")
		if len(params) != self._params:
			raise ValueError(f"parameter list has unexpected length: expected {self._params}, got {len(params)}")

		# Split the parameter vector into the part used by the objective function
		# and the part used by `g(x)`.
		lo = params[:-(self._params - self._objectives + 1)]
		hi = params[len(lo):]
		top = self._objectives - objective - 1

		from math import prod
		a = prod(lo[:top])
		b = (1 - lo[top]) if objective > 0 else 1
		c = 1 + self._target_g(hi)

		return a * b * c / 2

	@override
	def optimum_sdf(self, objs: List[float]) -> float:
		# The Pareto-optimal surface for DTLZ1 is the hyperplane defined by
		# `sum[objs] = 0.5`. We return the signed distance to it.
		return sum(objs) - 0.5

class Point:
	_begin: int
	_end: int
	_points: List[float]

	def __init__(self, points: List[float], begin: int, end: int):
		self._points = points
		self._begin = begin
		self._end = end

	@property
	def value(self) -> List[float]:
		return self._points[self._begin:self._end]

	def update(self, value: List[float]) -> None:
		if len(value) != self._end - self._begin:
			raise ValueError(f"size mismatch. expected {self._end - self._begin}, got {len(value)}")

		for i, x in enumerate(value):
			self._points[self._begin + i] = x

class Engine:
	_problem: Problem
	_points: int
	_points_curr: List[float]
	_points_orig: List[float]
	_generation: int

	def __init__(self, problem: Problem, points: int, rand: Callable[[], float] = random.random):
		"""
		Creates a new engine with the given problem and given number of points.
		Optionally, one may specify a random function, that will be used to
		generate the initial population, and that must only return values in
		the `0 <= x < 1` range.
		"""

		if points <= 0 or problem.params <= 0:
			raise ValueError(f"expected positive point count (got {points}) and positive problem parameter count (got {problem.params})")


		self._problem = problem
		self._points = points
		self._generation = 0

		self._points_curr = [0.0] * (self._problem.params * points)

		# Scatter the initial points uniformily over the ranges allowed by the
		# problem. Might be slow.
		for j in range(problem.params):
			los = []
			ranges: List[float] = []
			for i, (lo, hi) in enumerate(problem.range_of(j)):
				if lo >= hi:
					raise ValueError("problem returned empty/negative range [{lo},{hi}[ for parameter {i}")
				ranges.append((ranges[-1] if len(ranges) > 0 else 0) + (hi - lo))
				los.append(lo)

			for i in range(points):
				r = rand()
				if r < 0 or r >= 1:
					raise ValueError(f"provided random callable returned value {r}, outside the valid range 0 <= x < 1")

				sample = r * ranges[-1]
				index = bisect.bisect(ranges, sample)

				# Should never fail unless `r` is outside the `0 <= x < 1` range,
				# and we've hopefully already checked for that.
				assert index < len(ranges)

				base = ranges[index - 1] if index > 0 else 0
				offset = r - base

				self._points_curr[i * problem.params + j] = los[index] + offset

		# Create the first generation by copying from the set of points we've
		# just generated.
		self._points_orig = self._points_curr.copy()

	@property
	def points(self) -> Generator[Point]:
		a = self._problem.params
		yield from (Point(self._points_curr, i * a, (i + 1) * a) for i in range(self._points))

	def rollback(self) -> None:
		"""
		Indicates to the engine that the optimizer wants to move the value of
		`self.points` back to the start of this generation.
		self.
		"""
		self._points_curr = self._points_orig.copy()

	def commit(self) -> None:
		"""
		Indicates to the engine that the optimizer is done moving the points in
		the current generation, and that the current value of `self.points` should
		be the start of the new generation.
		"""
		self._points_orig = self._points_curr.copy()
		self._generation += 1

	def optimum_dist(self, point: int) -> float:
		"""
		Returns the signed distance between the point in the objective space
		that corresponds to the point with the given index and the Pareto-optimal
		front for the problem in the engine.
		"""
		pnt = self._points_orig[point * self._problem.params:(point + 1) * self._problem.params]
		obj = [self._problem.value_of(j, pnt) for j in range(self._problem.objectives)]

		return self._problem.optimum_sdf(obj)

	def plot(self, dim0: int, dim1: int | None = None, dim2: int | None = None) -> None:
		"""
		Uses MatPlotLib to plot the current generation in any one, two, or three
		dimensions of the objective space.
		"""

		dims = [x for x in (dim0, dim1, dim2) if x is not None]
		if len(dims) != 3:
			# We'll do it later.
			raise NotImplementedError()

		if dims[0] >= self._problem.objectives:
			raise ValueError(f"dim0 ({dim0}) is out of bounds for {self._problem.objectives} objective space dimensions")
		if dims[1] >= self._problem.objectives:
			raise ValueError(f"dim1 ({dim1}) is out of bounds for {self._problem.objectives} objective space dimensions")
		if dims[2] >= self._problem.objectives:
			raise ValueError(f"dim1 ({dim1}) is out of bounds for {self._problem.objectives} objective space dimensions")

		import matplotlib.pyplot as plt
		from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]

		fig = plt.figure()
		fig = plt.figure(figsize=(10, 15))
		ax: Axes3D = fig.add_subplot(111, projection='3d')

		rejected = []
		accepted = []

		for i in range(self._points):
			point = self._points_orig[i * self._problem.params:(i + 1) * self._problem.params]
			obj = [self._problem.value_of(j, point) for j in range(self._problem.objectives)]
			tgt = [obj[dims[0]], obj[dims[1]], obj[dims[2]]]

			# No reason to use 0.1 as the threshold to accept values.
			# TODO: Pick a better mechanism for rejecting values.
			if abs(self.optimum_dist(i)) < 0.1:
				accepted.append(tgt)
			else:
				rejected.append(tgt)

		import numpy

		accepted_np = numpy.asarray(accepted)
		rejected_np = numpy.asarray(rejected)

		if len(accepted) > 0:
			ax.scatter(accepted_np[:,0], accepted_np[:,1], accepted_np[:,2], color='red')
		if len(rejected) > 0:
			ax.scatter(rejected_np[:,0], rejected_np[:,1], rejected_np[:,2], color='gray')

		ax.view_init(elev=360, azim=25)
		plt.show()


