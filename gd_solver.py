"""
Single Variable Implementation of the Gradient Descent Algorithm

Instance methods provided to set the learning rate (alpha), dataset, and maximum
number of iterations to perform before exiting with a "No Converagence" exception.
Default alpha is set to 1, default max_iterations are set to 1000.
"""

class GDSolver:
    """Single Variable Implementation of the Gradient Descent Algorithm"""

    def __init__(self):
        self.alpha = 0.01
        self.convergence_limit = 0.00001
        self.max_iterations = 1000
        self.dataset = []
        self.debug = False
        self.theta0_guess = 0
        self.theta1_guess = 0

    def set_convergence_limit(self, convergence_limit):
        """set the convergence limit"""
        self.convergence_limit = convergence_limit

    def set_theta0_guess(self, theta0_guess):
        """set initial guess for theta0"""
        self.theta0_guess = theta0_guess
    
    def set_theta1_guess(self, theta1_guess):
        """set initial guess for theta1"""
        self.theta1_guess = theta1_guess

    def set_debug(self, debug):
        """log if debugging"""
        self.debug = debug

    def set_alpha(self, alpha):
        """set the the learning rate"""
        self.alpha = alpha

    def set_max_iteration(self, max_iterations):
        """set the max iterations"""
        self.max_iterations = max_iterations

    def set_dataset(self, dataset):
        """dataset must be an array of (x,y) tuples, with length m"""

        # check to make sure dataset is a valid type
        assert isinstance(dataset, (tuple, list)), 'Dataset must be an array'
        assert len(dataset) > 0, 'Dataset must have a non-zero length'
        assert isinstance(dataset[0], (tuple, list)
                          ), 'Dataset items must be arrays'
        assert len(dataset[0]) == 2, 'Dataset must be an array of (x,y) tuples'

        # set the dataset
        self.dataset = dataset

        if self.debug: 
            print(f'm =  {len(self.dataset)}')

    def solve(self):
        """solver for theta0 and theta1"""
        if not len(self.dataset) > 0:
            raise Exception('Dataset required to perform algorithm')

        # initialize loop variables
        iterations = 0
        theta0 = self.theta0_guess
        theta1 = self.theta1_guess
        delta0 = float('inf')
        delta1 = float('inf')

        # loop algorithm until convergence
        while abs(delta0) > self.convergence_limit and abs(delta1) > self.convergence_limit:

            if self.debug: 
                print(f'Starting iteration {iterations}. theta0 = {theta0}, theta1 = {theta1}')

            delta0 = self.alpha * self.__dtheta0j(theta0, theta1)
            delta1 = self.alpha * self.__dtheta1j(theta0, theta1)
            theta0 = theta0 - delta0
            theta1 = theta1 - delta1

            # count iterations and throw exception if no converagence is
            # achieved within the maximum amount of iterations
            iterations += 1
            if iterations >= self.max_iterations:
                raise Exception(
                    f'Could not converage within {self.max_iterations} iterations!')

        # return results
        return (theta0, theta1, iterations)

    def __dtheta0j(self, theta0, theta1):
        """get the partial derivative of the cost function wrt theta0"""

        # map the value of the cost function for each item in the dataset
        get_cost = lambda xy: (theta0 + theta1 * xy[0]) - xy[1]
        costs = map(get_cost, self.dataset)

        # if self.debug: 
        #     print(f'Starting iteration {iterations}. theta0 = {theta0}, theta1 = {theta1}')

        # apply 1/m multiplier and return the sum of the costs
        m = len(self.dataset)
        return 1 / m * sum(costs)

    def __dtheta1j(self, theta0, theta1):
        """get the partial derivative of the cost function wrt theta1"""
        
        # map the value of the cost function for each item in the dataset
        # theta1 gets an extra "x" multiplier
        get_cost = lambda xy: ((theta0 + theta1 * xy[0]) - xy[1]) * xy[0]
        costs = map(get_cost, self.dataset)

        # apply 1/m multiplier and return the sum of the costs
        m = len(self.dataset)
        return 1 / m * sum(costs)
