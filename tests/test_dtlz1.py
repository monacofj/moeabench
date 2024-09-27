
from optbench import DTLZ1, Engine


def test_dtlz1_set_optimum():
	"""
	Tests the DTLZ1 problem class and the Engine by setting all values to the
	known-optimal solution, and checking whether we can confirm that our points
	lie on the Pareto-optimal front.

	Ideally tessted over many different numbers of points, objectives and
	parameters.
	"""
	points = 1500
	objectives = 3
	parameters = 4

	engine = Engine(DTLZ1(objectives, parameters), points)

	# Set the last `k` points (x_M) to 0.5. This moves all of the points to the
	# Pareto-optimal front in objective space.
	for p, point in enumerate(engine.points):
		value = point.value
		for i in range(parameters - objectives + 1):
			value[-(i + 1)] = 0.5
		point.update(value)
	engine.commit()

	# Check if we know that all points are on the front.
	for i in range(points):
		assert abs(engine.optimum_dist(i)) < 0.1



