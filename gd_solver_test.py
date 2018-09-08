import csv
import os
from gd_solver import GDSolver


def csv_to_array(file_relative_path):
    """convert a csv file to a 2d array"""
    script_dir = os.path.dirname(__file__)
    full_path = os.path.join(script_dir, file_relative_path)
    arr = []
    with open(full_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',',
                            quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            arr.append([row[0], row[1]])
    return arr


def test_case1():
    """perform test number 1"""
    dataset = csv_to_array('test_datasets/test_dataset_1.csv')
    solver = GDSolver()

    # set configs
    solver.set_max_iteration(10000)
    solver.set_alpha(0.03)
    solver.set_theta0_guess(100)
    solver.set_theta1_guess(100)
    solver.set_alpha(0.03)
    solver.set_convergence_limit(0.00001)
    solver.set_debug(False)
    solver.set_dataset(dataset)

    results = solver.solve()
    assert round(results[0], 1) == 0.0
    assert round(results[1], 1) == 1.0


def test_case2():
    """perform test number 2"""
    dataset = csv_to_array('test_datasets/test_dataset_2.csv')
    solver = GDSolver()

    # set configs
    solver.set_max_iteration(10000)
    solver.set_alpha(0.03)
    solver.set_theta0_guess(2)
    solver.set_theta1_guess(2)
    solver.set_alpha(0.03)
    solver.set_convergence_limit(0.00001)
    solver.set_debug(False)
    solver.set_dataset(dataset)

    results = solver.solve()
    assert round(results[0], 1) == 0.0
    assert round(results[1], 1) == 1.0


def test_case3():
    """perform test number 3"""
    dataset = csv_to_array('test_datasets/test_dataset_3.csv')
    solver = GDSolver()

    # set configs
    solver.set_max_iteration(10000)
    solver.set_alpha(0.03)
    solver.set_theta0_guess(2)
    solver.set_theta1_guess(2)
    solver.set_alpha(0.03)
    solver.set_convergence_limit(0.00001)
    solver.set_debug(False)
    solver.set_dataset(dataset)

    results = solver.solve()
    assert round(results[0], 1) == 4.0
    assert round(results[1], 1) == -0.5
