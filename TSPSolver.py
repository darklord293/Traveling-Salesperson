#!/usr/bin/python3

from which_pyqt import PYQT_VER

if PYQT_VER == "PYQT5":
    from PyQt5.QtCore import QLineF, QPointF
else:
    raise Exception("Unsupported Version of PyQt: {}".format(PYQT_VER))


import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


class SubProblem:
    def __init__(self, cost, matrix, path):
        self.cost = cost  # Cost of the subproblem
        self.matrix = (
            matrix  # Matrix representing the reduced cost matrix of the subproblem
        )
        self.path = path  # List representing the current path taken in the subproblem


class BSSF:
    # bssf class holding cost and path
    def __init__(self, score, route):
        self.cost = score
        self.route = route


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    """ <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	"""

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Valid tour found
                foundTour = True
        end_time = time.time()
        results["cost"] = bssf.cost if foundTour else math.inf
        results["time"] = end_time - start_time
        results["count"] = count
        results["soln"] = bssf
        results["max"] = None
        results["total"] = None
        results["pruned"] = None
        return results

    """ <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	"""

    def greedy(self, time_allowance=60.0):
        cities = self._scenario.getCities()
        cost, matrix = self.create_initial_matrix(cities)  # extract initial cost

        results = {
            "cost": math.inf,
            "time": 0.0,
            "soln": None,
            "count": 0,
            "max": None,
            "total": None,
            "pruned": None,
        }

        start_time = time.time()

        # Iterate over possible starting nodes
        for initial_node in range(len(cities)):
            greedy_matrix = matrix.copy()
            greedy_cost = cost
            path = [initial_node]

            while len(path) != len(cities):
                # Check timeout
                if time.time() - start_time >= time_allowance:
                    return results

                min_val, col_index = self.find_next_city(greedy_matrix, path)
                # No valid path found
                if min_val == np.inf:
                    break

                self.update_greedy_matrix(greedy_matrix, col_index, path)
                greedy_cost += min_val
            # complete tour found
            else:
                greedy_cost += greedy_matrix[path[-1], initial_node]
                cities_path = [cities[i] for i in path]
                solution = TSPSolution(cities_path)

                if solution.cost < results["cost"]:
                    results["cost"] = solution.cost
                    results["time"] = time.time() - start_time
                    results["soln"] = solution
                    results["count"] = 1

        return results

    def find_next_city(self, matrix, path):
        # find next city with minimum edge from current city
        min_val = np.min(matrix[path[-1], :])
        col_index = np.argmin(matrix[path[-1]])
        return min_val, col_index

    def update_greedy_matrix(self, matrix, col_index, path):
        # update the matrix and prevent revisit of the city
        matrix[path[-1], :] = np.inf
        matrix[:, col_index] = np.inf
        matrix[col_index, path[-1]] = np.inf
        path.append(col_index)

    """ <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	"""

    def branchAndBound(self, time_allowance=60.0):
        results = {"pruned": 0, "total": 0, "max": 0, "count": 0}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        # use priority queue to keep track of next subproblem to search
        priority_queue = []
        # dictionary mapping priority hashing to sub problem
        priority_dict = {}
        cost, matrix = self.create_initial_matrix(cities)
        # initial BSSF using greedy approach
        greedy_bssf = self.greedy()
        if greedy_bssf["cost"] != np.inf:
            foundTour = True
            bssf = BSSF(
                greedy_bssf["cost"], [city._index for city in greedy_bssf["soln"].route]
            )
        else:
            bssf = BSSF(np.inf, None)
        start_time = time.time()
        timed_out = False
        initial_node = 0
        cur_path = [initial_node]
        initial_problem = SubProblem(cost, matrix, cur_path)
        results["total"] += 1
        # update the priority queue
        self.add_to_queue(initial_problem, priority_queue, priority_dict)
        while len(priority_queue) != 0:
            # sort the priority queue
            priority_queue.sort(reverse=True)
            cur_problem_score = priority_queue.pop()
            cur_problem = priority_dict[cur_problem_score]
            for j in range(ncities):
                # Check timeout
                if time.time() - start_time > time_allowance:
                    timed_out = True
                    break
                # expand the subproblem
                problemChild = self.expand_subproblem(cur_problem, j, results)
                if problemChild is None:
                    continue
                results["total"] += 1
                if len(problemChild.path) == ncities:
                    if (
                        problemChild.matrix[problemChild.path[-1], problemChild.path[0]]
                        != np.inf
                    ):
                        foundTour = True
                        problemChild.cost += problemChild.matrix[
                            problemChild.path[-1], problemChild.path[0]
                        ]
                        # update the BSSF
                        if problemChild.cost < bssf.cost:
                            results["count"] += 1
                            bssf.cost = problemChild.cost
                            bssf.route = problemChild.path
                            continue
                        # prune the child
                        else:
                            results["pruned"] += 1
                    else:
                        results["pruned"] += 1
                elif problemChild.cost >= bssf.cost:
                    results["pruned"] += 1
                else:
                    self.add_to_queue(problemChild, priority_queue, priority_dict)
                    if len(priority_queue) > results["max"]:
                        results["max"] = len(priority_queue)
            if timed_out:
                break

        citiesPath = [cities[index] for index in bssf.route]
        bandbSolution = TSPSolution(citiesPath)

        end_time = time.time()
        # update the result
        results["cost"] = bandbSolution.cost if foundTour else math.inf
        results["time"] = end_time - start_time
        results["soln"] = bandbSolution
        return results

    def expand_subproblem(self, subProblem, col, results):
        if subProblem.matrix[subProblem.path[-1], col] == np.inf:
            return None

        matrix_copy = subProblem.matrix.copy()
        matrix_copy[subProblem.path[-1]] = np.inf
        matrix_copy[:, col] = np.inf
        matrix_copy[col, subProblem.path[-1]] = np.inf

        reductionCost = self.reduce_cost_matrix(matrix_copy)

        new_cost = (
            reductionCost
            + subProblem.cost
            + subProblem.matrix[subProblem.path[-1], col]
        )
        new_path = subProblem.path.copy()
        new_path.append(col)
        return SubProblem(new_cost, matrix_copy, new_path)

    def reduce_cost_matrix(self, matrix):
        lower_bound = 0
        for i in range(len(matrix)):
            row_min = min(matrix[i, :])
            if row_min != np.inf:
                matrix[i] -= row_min
                lower_bound += row_min
        for j in range(len(matrix)):
            col_min = min(matrix[:, j])
            if col_min != np.inf:
                matrix[:, j] -= col_min
                lower_bound += col_min
        return lower_bound

    def create_initial_matrix(self, cities):
        initMatrix = np.full((len(cities), len(cities)), np.inf)
        reductionCost = 0
        for i, city_i in enumerate(cities):
            for j, city_j in enumerate(cities):
                if i != j:
                    initMatrix[i, j] = city_i.costTo(city_j)
        reductionCost = self.reduce_cost_matrix(initMatrix)
        return reductionCost, initMatrix

    def add_to_queue(self, subProblem, priority_queue, priority_dict):
        hash_value = subProblem.cost / len(subProblem.path)
        priority_queue.append(hash_value)
        priority_dict[hash_value] = subProblem

    """ <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	"""

    def fancy(self, time_allowance=60.0):
        pass
