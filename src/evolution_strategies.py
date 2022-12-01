from datetime import datetime

import numpy as np
import ioh

from utils import ProgressBar

class EvolutionStrategies:
    
    def __init__(
        self,
        problem: ioh.ProblemType,
        pop_size: int,
        mu_: int,
        lambda_: int,
        tau_: float,
        sigma_: float,
        budget: int = 5_000,
        minimize: bool = False,
        recombination: str = 'd',
        individual_sigmas: bool = False,
        run_id: any = None,
        verbose: bool = False
        ) -> None:
        
        """ Sets all parameters """

        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        kwargs = locals(); kwargs.pop('self')
        self.validate_parameters(**kwargs)

        self.problem = problem
        self.pop_size = pop_size
        self.mu_ = mu_
        self.lambda_ = lambda_
        self.tau_ = tau_
        self.sigma_prop = sigma_  # what gets passed as sigma_ should be interpreted as the proportion wrt the bounds
        self.minimize = minimize
        self.budget = budget
        self.isig = individual_sigmas
        self.run_id = str(run_id)
        self.verbose = verbose

        self.n_dimensions = problem.meta_data.n_variables
        self.lb = problem.bounds.lb[0]
        self.ub = problem.bounds.ub[0]

        if self.pop_size == self.lambda_:
            self.selection_kind = ','
        else:  # pop_size is mu + lambda
            self.selection_kind = '+'

        self.recombination = dict(
            d = self.recombination_discrete,
            i = self.recombination_intermediate,
            dg = self.recombination_discrete_global,
            ig = self.recombination_intermediate_global
        )[recombination]

        self.n_generations = self.budget // self.pop_size
        self.history = np.zeros(self.n_generations)
        if self.verbose:
            self.progress = ProgressBar(self.n_generations, p_id=self.run_id)

        self.f_opt = np.inf  # problem is always a minimization one
        self.x_opt = None

        return


    def optimize(self, return_history: bool = False) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, np.ndarray]:
        """
        Runs the optimization algorithm and returns the best candidate solution found and its fitness.
        If return_history is set to True, it will also return the history
        of the best fitness value found in each population.
        """

        self.initialize_population()
        improvement = lambda x, y: x < y if self.minimize else x > y

        for gen in range(self.n_generations):
            self.population, self.pop_sigmas, f_opt_in_pop = self.evaluate_population()
            self.history[gen] = f_opt_in_pop

            if improvement(f_opt_in_pop, self.f_opt):
                self.f_opt = f_opt_in_pop
                self.x_opt = self.population[0]

            # deterministic selection            
            parents = self.population[:self.mu_]
            parents_sigmas = self.pop_sigmas[:self.mu_]

            offspring = np.zeros((self.lambda_, self.n_dimensions))
            offspring_sigmas = np.zeros((self.lambda_, self.n_dimensions))
            for i in range(self.lambda_):
                offspring[i], offspring_sigmas[i] = self.recombination(parents, parents_sigmas)
            
            offspring, offspring_sigmas = self.mutate(offspring, offspring_sigmas)

            if self.selection_kind == '+':
                self.population = np.concatenate((parents, offspring), axis=0)
                self.pop_sigmas = np.concatenate((parents_sigmas, offspring_sigmas), axis=0)
            else:  # selection kind is ,
                self.population = offspring
                self.pop_sigmas = offspring_sigmas

            if self.verbose:
                self.progress(gen)

        if self.verbose:
            print(f'f_opt: {self.f_opt:.5f}')
            print(f'x_opt: {self.x_opt}')

        if return_history:
            return self.x_opt, self.f_opt, self.history
        else:
            return self.x_opt, self.f_opt


    def evaluate_population(self) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Evaluates the fitness of all candidate solutions in the population.
        Returns the candidate solutions ranked by fitness values, along with their sigmas and the highest fitness value.
        """
        pop_fitness = np.array([self.problem(x) for x in self.population])
        if self.minimize:
            ranking = np.argsort(pop_fitness)
        else:
            ranking = np.argsort(pop_fitness)[::-1]

        return self.population[ranking], self.pop_sigmas[ranking], np.max(pop_fitness)


    def initialize_population(self) -> None:
        """ Initializes population and sigmas with random values between lower and upper bounds """

        self.population = np.random.uniform(
            self.lb,
            self.ub,
            (self.pop_size, self.n_dimensions)
        )

        sigma = self.sigma_prop * (self.ub - self.lb)

        if self.isig:  # every parameter has its own sigma associated with it
            self.pop_sigmas = np.random.uniform(
                self.lb * sigma,
                self.ub * sigma,
                (self.pop_size, self.n_dimensions)
            )
        else:  # every parameter has the same sigma associated with it, but it still differs from candidate to candidate
            self.pop_sigmas = np.repeat(
                np.random.uniform(
                    self.lb * sigma,
                    self.ub * sigma,
                    self.pop_size
                ),
                self.n_dimensions
            ).reshape(self.pop_size, self.n_dimensions)

        return


    def mutate(self, individuals: np.ndarray, sigmas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Mutates the individuals in the given 2D array (will also work for 1D array).
        Returns both the updated individuals as well as the updated sigmas.
        """
        
        mutated = individuals + sigmas
        mutated = np.clip(mutated, self.lb, self.ub)  # clip to bounds
        
        updated_sigmas = sigmas * np.exp(self.tau_ * np.random.normal(0, 1, sigmas.shape))

        return mutated, updated_sigmas


    def recombination_discrete(self, parents: np.ndarray, parents_sigmas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ Chooses two parents and returns one child, randomly choosing a parameter from each parent """
        
        pi1, pi2 = np.random.choice(parents.shape[0], 2, replace=False)
        c, cs = np.zeros(self.n_dimensions), np.zeros(self.n_dimensions)
        for i in range(self.n_dimensions):
            if np.random.uniform() < 0.5:
                c[i], cs[i] = parents[pi1, i], parents_sigmas[pi1, i]
            else:
                c[i], cs[i] = parents[pi2, i], parents_sigmas[pi2, i]

        return c, cs
    
    def recombination_intermediate(self, parents: np.ndarray, parents_sigmas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ Chooses two parents and returns one child, averaging the parameters of the two parents """

        pi1, pi2 = np.random.choice(parents.shape[0], 2, replace=False)
        c, cs = np.zeros(self.n_dimensions), np.zeros(self.n_dimensions)
        for i in range(self.n_dimensions):
            c[i], cs[i] = (parents[pi1, i] + parents[pi2, i]) / 2, (parents_sigmas[pi1, i] + parents_sigmas[pi2, i]) / 2

        return c, cs
    
    def recombination_discrete_global(self, parents: np.ndarray, parents_sigmas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ Randomly chooses each parameter from all parents """

        c, cs = np.zeros(self.n_dimensions), np.zeros(self.n_dimensions)
        for i in range(self.n_dimensions):
            pi = np.random.choice(parents.shape[0])
            c[i], cs[i] = parents[pi, i], parents_sigmas[pi, i]

        return c, cs
    
    def recombination_intermediate_global(self, parents: np.ndarray, parents_sigmas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ Averages all parameters of all parents """

        c, cs = np.mean(parents, axis=0), np.mean(parents_sigmas, axis=0)

        return c, cs


    def validate_parameters(
        self,
        problem: ioh.ProblemType,
        pop_size: int,
        mu_: int,
        lambda_: int,
        tau_: float,
        sigma_: float,
        minimize: bool,
        budget: int,
        recombination: str,
        individual_sigmas: bool,
        run_id: any,
        verbose: bool
        ) -> None:

        """ Validates all parameters passed to the constructor """
        
        assert isinstance(problem, ioh.ProblemType), "problem must be an instance of <ioh.ProblemType>"

        assert isinstance(pop_size, int), "population size must be an integer"
        assert pop_size in range(0, 500), "population size must be between 0 and 500"

        assert isinstance(mu_, int), "mu_ must be an integer"
        assert mu_ > 0, "mu_ must be greater than 0"
        assert mu_ < pop_size, "mu_ must be less than population size"

        assert isinstance(lambda_, int), "lambda_ must be an integer"
        assert lambda_ > 0, "lambda_ must be greater than 0"
        assert lambda_ <= pop_size, "lambda_ must be less than or equal to population size"

        assert pop_size == lambda_ + mu_ or pop_size == lambda_, "population size must be" + \
        " either number of parents + number of offspring or just number of offspring"

        assert isinstance(tau_, float), "tau_ must be a float"
        assert tau_ > 0, "tau_ must be greater than 0"
        assert tau_ < 1, "tau_ must be less than 1"

        assert isinstance(sigma_, float), "sigma_ must be a float"
        assert sigma_ > 0, "sigma_ must be greater than 0"
        assert sigma_ < 1, "sigma_ must be less than 1"

        assert isinstance(minimize, bool), "min must be a boolean"

        assert isinstance(budget, int), "budget must be an integer"
        assert budget > 0, "budget must be greater than 0"
        assert budget < 10_000_000, "budget must be less than 100 million"

        assert recombination in ['d', 'i', 'dg', 'ig'], "recombination must be one of the following: 'd', 'i', 'dg', 'ig'"

        assert isinstance(individual_sigmas, bool), "individual_sigmas must be a boolean"
        
        assert len(str(run_id)) > 0, "run_id must be representable as a string"

        assert isinstance(verbose, bool), "verbose must be a boolean"

        return
