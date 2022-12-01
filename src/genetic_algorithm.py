from datetime import datetime

import numpy as np
import ioh

from utils import ProgressBar

class GeneticAlgorithm:

    def __init__(
        self,
        problem: ioh.ProblemType,
        pop_size: int,
        mu_: int,
        lambda_: int,
        budget: int = 5_000,
        minimize: bool = False,
        selection: str = 'rw',
        recombination: str = 'kp',
        mutation: str = 'u',
        xp: int = None,
        mut_rate: float = None,
        mut_nb: int = None,
        run_id: any = None,
        verbose: bool = False
        ) -> None:

        """ Sets all parameters. """

        if run_id is None:
            run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
        kwargs = locals(); kwargs.pop('self')
        self.validate_parameters(**kwargs)

        self.problem = problem
        self.pop_size = pop_size
        self.mu_ = mu_
        self.lambda_ = lambda_
        self.budget = budget
        self.minimize = minimize
        self.run_id = str(run_id)
        self.verbose = verbose

        self.n_dimensions = problem.meta_data.n_variables

        if self.pop_size == self.lambda_:
            self.selection_kind = ','
        else:  # pop_size is mu + lambda
            self.selection_kind = '+'

        self.selection = dict(
            rw = self.selection_roulette_wheel,
            ts = self.selection_tournament,
            rk = self.selection_rank,
            su = self.selection_stochastic_universal
        )[selection]

        self.recombination = dict(
            kp = self.recombination_k_point,
            u = self.recombination_uniform
        )[recombination]

        if recombination == 'kp':
            self.xp = xp if xp is not None else 1
        else:
            self.xp = None
        
        self.mutation = dict(
            u = self.mutation_uniform,
            b = self.mutation_bitflip
        )[mutation]

        if mutation == 'u':
            self.mut_rate = mut_rate if mut_rate is not None else 1 / self.n_dimensions
            self.mut_nb = None
        elif mutation == 'b':
            self.mut_nb = mut_nb if mut_nb is not None else 1
            self.mut_rate = None

        self.n_generations = self.budget // self.pop_size
        self.history = np.zeros((self.n_generations))
        if self.verbose:
            self.progress = ProgressBar(self.n_generations, p_id=self.run_id)

        self.f_opt = -np.inf  # problem is always a maximization one? TODO check this
        self.x_opt = None

        return


    def optimize(self, return_history: bool = False) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, np.ndarray]:
        """
        Runs the optimization algorithm and returns the best candidate solution found and its fitness.
        If return_history is set to True, it will also return the history
        of the best fitness value found in each population.
        """

        self.population = np.random.randint(2, size=(self.pop_size, self.n_dimensions), dtype=int)
        improvement = lambda x, y: x < y if self.minimize else x > y

        for gen in range(self.n_generations):
            self.population, self.pop_fitness = self.evaluate_population()
            f_opt_in_pop = self.pop_fitness[0]
            self.history[gen] = f_opt_in_pop

            if improvement(f_opt_in_pop, self.f_opt):
                self.f_opt = f_opt_in_pop
                self.x_opt = self.population[0]
            
            parents = self.selection()
            offspring = self.recombination(parents)
            offspring = self.mutation(offspring)

            if self.selection_kind == '+':
                self.population = np.concatenate((parents, offspring), axis=0)
            else:  # selection_kind is ','
                self.population = offspring
            
            if self.verbose:
                self.progress(gen)
        
        if self.verbose:
            print(f'f_opt: {self.f_opt:.2f}')
            print(f'x_opt: {self.x_opt}')

        if return_history:
            return self.x_opt, self.f_opt, self.history
        else:
            return self.x_opt, self.f_opt


    def evaluate_population(self) -> tuple[np.ndarray, float]:
        """
        Evaluates the fitness of the population and returns the best fitness value found.
        Returns the candidate solutions ranked by fitness values, along with the highest fitness value.
        """

        pop_fitness = np.array([self.problem(list(x)) for x in self.population])
        if self.minimize:
            ranking = np.argsort(pop_fitness)
        else:
            ranking = np.argsort(pop_fitness)[::-1]

        return self.population[ranking], pop_fitness[ranking]


    def selection_roulette_wheel(self) -> np.ndarray:
        """ Selects parents using roulette wheel selection. """

        pop_fitness = self.pop_fitness.copy()  # copy because we're going to modify it
        if self.minimize:
            # flip fitness values so lowest "fitness" values get higher probability
            pop_fitness = -pop_fitness
        total_fitness = np.sum(pop_fitness)
        if total_fitness == 0:
            total_fitness = 1e-10
        probs = pop_fitness / total_fitness
        cum_probs = np.cumsum(probs)

        parents = np.zeros((self.mu_, self.n_dimensions), dtype=int)

        for i in range(self.mu_):
            r = np.random.rand()
            for j in range(self.pop_size):
                if r < cum_probs[j]:
                    parents[i] = self.population[j]
                    break

        return parents

    def selection_tournament(self) -> np.ndarray:
        """ Selects parents using tournament selection. """

        parents = np.zeros((self.mu_, self.n_dimensions), dtype=int)

        for i in range(self.mu_):
            pool = np.random.choice(self.pop_size, 2)  # TODO add parameter to determine pool size
            # TODO define which arg[min/max] to use in __init__ to reduce if-statements while iterating
            if self.minimize:
                parents[i] = self.population[pool[np.argmin(self.pop_fitness[pool])]]
            else:
                parents[i] = self.population[pool[np.argmax(self.pop_fitness[pool])]]

        return parents
    
    def selection_rank(self) -> np.ndarray:
        """ Selects parents using rank selection. """

        parents = np.zeros((self.mu_, self.n_dimensions), dtype=int)
        # TODO define this in init once to save on performance
        probs = np.arange(self.pop_size, 0, -1) / (self.pop_size * (self.pop_size + 1) / 2)
        cum_probs = np.cumsum(probs)

        for i in range(self.mu_):
            r = np.random.rand()
            for j in range(self.pop_size):
                if r < cum_probs[j]:
                    parents[i] = self.population[j]
                    break
        
        return parents

    def selection_stochastic_universal(self) -> np.ndarray:
        """ Selects parents using stochastic universal sampling. """

        pop_fitness = self.pop_fitness.copy()  # copy because we're going to modify it
        if self.minimize:
            # convert all zeros in pop_fitness to 1e-10 to avoid division by zero
            pop_fitness = np.where(pop_fitness == 0, 1e-10, pop_fitness)
            # invert fitness values so lowest "fitness" values get higher prob
            pop_fitness = 1 / pop_fitness
        total_fitness = np.sum(pop_fitness)
        cum_fitness = np.cumsum(pop_fitness)

        parents = np.zeros((self.mu_, self.n_dimensions), dtype=int)
        fitness_step = total_fitness / self.mu_
        pointer = np.random.uniform(0, fitness_step)

        for i in range(self.mu_):
            for j in range(self.pop_size):
                if pointer < cum_fitness[j]:
                    parents[i] = self.population[j]
                    pointer += fitness_step
                    break

        return parents


    def recombination_k_point(self, parents: np.ndarray) -> np.ndarray:
        """ Recombines parents using k-point crossover to produce all offspring. """

        offspring = np.zeros((self.lambda_, self.n_dimensions), dtype=int)

        for i in range(0, self.lambda_, 2):
            pi1, pi2 = np.random.choice(self.mu_, 2, replace=False)
            points = np.random.choice(range(1, self.n_dimensions), self.xp, replace=False)
            points.sort()

            prev_point = 0
            for point in points:
                offspring[i][prev_point:point] = parents[pi1][prev_point:point]
                offspring[i+1][prev_point:point] = parents[pi2][prev_point:point]
                prev_point = point
                temp = pi1; pi1 = pi2; pi2 = temp  # swap parent ids

            offspring[i][prev_point:] = parents[pi1][prev_point:]
            offspring[i+1][prev_point:] = parents[pi2][prev_point:]

        return offspring

    def recombination_uniform(self, parents: np.ndarray) -> np.ndarray:
        """ Recombines parents using uniform crossover to produce all offspring. """

        offspring = np.zeros((self.lambda_, self.n_dimensions), dtype=int)

        for i in range(0, self.lambda_, 2):
            pi1, pi2 = np.random.choice(self.mu_, 2, replace=False)
            mask = np.random.choice(2, self.n_dimensions, replace=True)
            offspring[i] = parents[pi1] * mask + parents[pi2] * (1 - mask)
            offspring[i+1] = parents[pi2] * mask + parents[pi1] * (1 - mask)

        return offspring

    
    def mutation_uniform(self, offspring: np.ndarray) -> np.ndarray:
        """ Mutates the offspring using uniform mutation. """

        for i in range(self.lambda_):
            for j in range(self.n_dimensions):
                if np.random.rand() < self.mut_rate:
                    offspring[i][j] = not offspring[i][j]

        return offspring
    
    def mutation_bitflip(self, offspring: np.ndarray) -> np.ndarray:
        """ Mutates the offspring using bitflip mutation. """

        for i in range(self.lambda_):
            points = np.random.choice(range(self.n_dimensions), self.mut_nb, replace=False)
            offspring[i][points] = not offspring[i][points]

        return offspring


    def validate_parameters(
        self,
        problem: ioh.ProblemType,
        pop_size: int,
        mu_: int,
        lambda_: int,
        budget: int,
        minimize: bool,
        selection: str,
        recombination: str,
        mutation: str,
        xp: int,
        mut_rate: float,
        mut_nb: int,
        run_id: any,
        verbose: bool
        ) -> None:

        """ Validates all parameters passed to the constructor. """

        dims = problem.meta_data.n_variables

        assert isinstance(problem, ioh.ProblemType), "problem must be an instance of <ioh.ProblemType>"

        assert isinstance(pop_size, int), "pop_size must be an integer"
        assert pop_size in range(0, 500), "pop_size must be between 0 and 500"

        assert isinstance(mu_, int), "mu_ must be an integer"
        assert mu_ > 0, "mu_ must be greater than 0"
        assert mu_ < pop_size, "mu_ must be less than population size"

        assert isinstance(lambda_, int), "lambda_ must be an integer"
        assert lambda_ > 0, "lambda_ must be greater than 0"
        assert lambda_ <= pop_size, "lambda_ must be less than or equal to population size"
        assert lambda_ % 2 == 0, "lambda_ must be an even number"

        assert pop_size == lambda_ + mu_ or pop_size == lambda_, "population size must be" + \
        " either number of parents + number of offspring or just number of offspring"

        assert isinstance(budget, int), "budget must be an integer"
        assert budget > 0, "budget must be greater than 0"
        assert budget <= 100_000_000, "budget must be less than or equal to 100 million"

        assert isinstance(minimize, bool), "minimize must be a boolean"

        assert selection in ['rw', 'ts', 'rk', 'su'], "selection must be one of the following: 'rw', 'ts', 'rk', 'su'"

        assert recombination in ['kp', 'u'], "recombination must be one of the following: 'u', 'kp'"
        if recombination == 'kp':
            if xp is not None:
                assert isinstance(xp, int), "for kp crossover, xp must be an integer"
                assert xp > 0, "for kp crossover, xp must be greater than 0"
                assert xp < dims, "for kp crossover, xp must be less than problem dimension"

        assert mutation in ['u', 'b'], "mutation must be one of the following: 'u', 'b'"
        if mutation == 'u':
            if mut_rate is not None:
                assert isinstance(mut_rate, float), "for u, mut_rate must be a float"
                assert mut_rate > 0, "for u, mut_rate must be greater than 0"
                assert mut_rate < 1, "for u, mut_rate must be less than 1"
        if mutation == 'b':
            if mut_rate is not None:
                assert isinstance(mut_rate, int), "for n, mut_rate must be an integer"
                assert mut_rate > 0, "for n, mut_rate must be greater than 0"
                assert mut_rate < dims, "for n, mut_rate must be less than problem dimension"
        
        assert len(str(run_id)) > 0, "run_id must be representable as a string"

        assert isinstance(verbose, bool), "verbose must be a boolean"
        
        return
