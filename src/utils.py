import os
from argparse import ArgumentParser
from warnings import warn
from time import perf_counter
from datetime import datetime

from numpy import sqrt


def get_directories(parent_file: str) -> dict[str, str]:
    """Returns a tuple of directories to be used in the program."""
    src = os.path.dirname(os.path.abspath(parent_file))
    root = os.sep.join(src.split(os.sep)[:-1])
    data = os.path.join(root, 'data')
    results = os.path.join(root, 'results')
    csv = os.path.join(results, 'csv')
    pkl = os.path.join(results, 'pkl')
    plots = os.path.join(results, 'plots')
    
    dirs: dict[str, str] = {}
    for directory in (data, results, csv, pkl, plots):
        basename = os.path.basename(directory)
        if not os.path.exists(directory):
            os.mkdir(directory)
            print()
            warn(f'Created empty {basename} directory at "{directory}".')
            print()
        dirs[basename] = directory + os.sep
    return dirs


class ParseWrapper:

    def __init__(self, parser: ArgumentParser) -> None:

        BOLD = lambda x: f'\033[1m{x}\033[0m'

        # "basic" evolutionary parameters
        parser.add_argument('-o', dest='optimizer', type=str, default='GA',
                            choices=['GA', 'ES'],
                            help="Optimizer to use.")
        parser.add_argument('-b', dest='budget', type=int, default=5_000,
                            help=f'Number of function evaluations ({BOLD("b")}udget): [1, 1 million].')
        parser.add_argument('-d', dest='dimension', type=int, default=None,
                            help="Dimension of the problem: [1, 100].")
        # --- TODO make it so that m & l automatically scale to p --- #
        parser.add_argument('-p', dest='population_size', type=int, default=100,
                            help="Population size: [10, 1000].")
        parser.add_argument('-m', dest='mu_', type=int, default=40,
                            help="Number of parents: [2, pop_size-1].")
        parser.add_argument('-l', dest='lambda_', type=int, default=100,
                            help="Number of offspring: [2, pop_size].")
        # ----------------------------------------------------------- #
        parser.add_argument('-s', dest='sigma_', type=float, default=0.01,
                            help=f'({BOLD("ES only")} Initial mutation strength (sigma): [0.001, 1].')
        parser.add_argument('-t', dest='tau_', type=float, default=0.1,
                            help=f'({BOLD("ES only")}) Perturbation rate for sigma: [0.01, 1].')

        # operator methods
        parser.add_argument('--sel', dest='selection', type=str, default='rw',
                            choices=['rw', 'ts', 'rk', 'su'],
                            help=f'''
                                ({BOLD("GA only")})
                                Selection method:
                                {BOLD("r")}oulette {BOLD("w")}heel selection,
                                {BOLD("t")}ournament {BOLD("s")}election,
                                {BOLD("r")}an{BOLD("k")} selection,
                                {BOLD("s")}tochastic {BOLD("u")}niversal sampling.
                            ''')
        parser.add_argument('--mut', dest='mutation', type=str, default='u',
                            choices=['u', 'b'],
                            help=f'''
                                ({BOLD("GA only")})
                                Mutation method:
                                {BOLD("u")}niform,
                                {BOLD("b")}itflip.
                            ''')
        parser.add_argument('--rec', dest='recombination', type=str, default=None,
                            choices=['kp', 'u', 'd', 'i', 'dg', 'ig'],
                            help=f'''
                                Recombination method.
                                ({BOLD("GA")}):
                                {BOLD("kp")}oint,
                                {BOLD("u")}niform.
                                ({BOLD("ES")}):
                                {BOLD("d")}iscrete,
                                {BOLD("i")}ntermediate,
                                {BOLD("d")}iscrete {BOLD("g")}lobal,
                                {BOLD("i")}ntermediate {BOLD("g")}lobal.
                                Default is {BOLD("kp")}oint for {BOLD("GA")} and {BOLD("d")}iscrete for {BOLD("ES")}.
                            ''')
        
        # "advanced" evolutionary parameters
        parser.add_argument('--is', dest='individual_sigmas', action='store_true',
                            help=f'({BOLD("ES only")}) Use individual mutation strengths (sigmas).')
        parser.add_argument('--lb', dest='lower_bound', type=float,
                            help=f'({BOLD("ES only")}) Lower bound of the problem. Must be < ub.')
        parser.add_argument('--ub', dest='upper_bound', type=float,
                            help=f'({BOLD("ES only")}) Upper bound of the problem. Must be > lb.')
        parser.add_argument('--xp', dest='xp', type=int, default=1,
                            help=f'({BOLD("GA w/ kp only")}) Number of crossover points in recombination.')
        parser.add_argument('--mb', dest='mut_b', type=int, default=1,
                            help=f'({BOLD("GA w/ b only")}) Number of bits to flip in mutation.')
        parser.add_argument('--mr', dest='mut_r', type=float, default=None,
                            help=f'''
                                ({BOLD("GA w/ u only")}) Mutation rate: [0, 1].
                                Default will be set to 1/popsize.
                            ''')

        # experiment level parameters
        parser.add_argument('-P', '--pid', type=int, default=1,
                            help=f'IOH problem ID: {BOLD("GA")}: [1, 24], {BOLD("ES")}: [1, 25],.')
        parser.add_argument('-I', '--run_id', type=str, default=None,
                            help=f"""
                                Identifier for the current run.
                                Default will be set to <GA/ES>_<problem_id>_<date_time>.
                            """)
        parser.add_argument('-R', '--repetitions', type=int, default=1,
                            help="Number of repetitions for each experiment: [1, 100].")
        parser.add_argument('-S', '--seed', type=int, default=42,
                            help="Seed for the random number generator: [0, 999999].")
        parser.add_argument('-M', '--minimize', action='store_true',
                            help="Minimize the objective function.")
        parser.add_argument('-V', '--verbose', type=int, default=1,
                            help="Determines how much is logged to stdout: [0, 2].")
        parser.add_argument('-O', '--overwrite', action='store_true',
                            help="Overwrite existing data if already present.")
        parser.add_argument('--plot', action='store_true',
                            help="Plot the results with matplotlib.")

        self.args = vars(parser.parse_args())
        
        # resolve default Nones
        if self.args['recombination'] is None:
            self.args['recombination'] = 'kp' if self.args['optimizer'] == 'GA' else 'd'
        if self.args['optimizer'] == 'GA' and self.args['mutation'] == 'u':
            if self.args['mut_r'] is None:
                self.args['mut_r'] = 1 / self.args['population_size']
        if self.args['run_id'] is None:
            self.args['run_id'] = self.args['optimizer'] + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.validate_args()

    def __call__(self) -> dict[str, any]:
        print('\nExperiment will be ran with the following parameters:')
        for arg, value in self.args.items():
            print(f'{arg:>20} | {value}')
        return self.args

    def validate_args(self) -> None:
        
        # "basic" evolutionary parameters
        assert self.args['budget'] in range(1, 1_000_001), "Budget must be in [1, 1 million]."

        dim = self.args['dimension']
        if dim is not None:
            assert dim in range(1, 101), "Dimension must be in [1, 100]."
        else:
            dim = 121  # dirty fix for the rest of the test cases

        assert self.args['population_size'] in range(10, 1001), "Population size must be in [10, 1000]."

        population_size = self.args['population_size']
        assert self.args['mu_'] in range(2, population_size), "Number of parents must be in [2, pop_size-1]."

        assert self.args['lambda_'] in range(2, population_size+1), "Number of offspring must be in [2, pop_size]."

        mu_, lambda_ = self.args['mu_'], self.args['lambda_']
        assert mu_ + lambda_ == population_size or lambda_ == population_size, \
            "mu + lambda must be popsize or lambda must be popsize."
        
        sigma_ = self.args['sigma_']
        assert sigma_ >= 0.001 and sigma_ <= 1, "Initial mutation strength (sigma) must be in [0.001, 1]."

        tau_ = self.args['tau_']
        assert tau_ >= 0.01 and tau_ <= 1, "Perturbation rate for sigma must be in [0.01, 1]."

        # operator methods
        rec = self.args['recombination']
        if self.args['optimizer'] == 'GA':
            assert rec in ['kp', 'u'], "For GA, Recombination method must be in [kp, u]."
        else:
            assert rec in ['d', 'i', 'dg', 'ig'], \
                "For ES, Recombination method must be in [d, i, dg, ig]."

        # "advanced" evolutionary parameters
        if self.args['optimizer'] == 'ES':
            
            lb, ub = self.args['lower_bound'], self.args['upper_bound']
            if lb is not None:
                assert ub is not None, "Upper bound must be specified if lower bound is specified."
                assert lb < ub, "Lower bound must be < upper bound."
            if ub is not None:
                assert lb is not None, "Lower bound must be specified if upper bound is specified."
                assert ub > lb, "Upper bound must be > lower bound."

        else:  # GA
            if rec == 'kp':
                assert self.args['xp'] in range(1, dim), "Number of crossover points must be in [1, dim-1]."
            
            if self.args['mutation'] == 'u':
                if self.args['mut_r'] is not None:
                    assert self.args['mut_r'] > 0 and self.args['mut_r'] < 1, \
                        "Mutation rate must be between 0 and 1."
            else:  # b
                assert self.args['mut_b'] in range(1, dim), \
                    "Number of bits to flip must be between 0 and problem dimension."

        # experiment level parameters
        if self.args['optimizer'] == 'GA':
            assert self.args['pid'] in range(1, 26), "For GA, problem ID must be in [1, 25]."
        else:  # ES
            assert self.args['pid'] in range(1, 25), "For ES, problem ID must be in [1, 24]."
        
        if self.args['run_id'] is not None:
            assert len(self.args['run_id']) <= 50, "Identifier must be less than 50 characters long."

        assert self.args['repetitions'] in range(1, 101), "Number of repetitions must be in [1, 100]."

        assert self.args['seed'] in range(0, 1_000_000), "Seed must be an integer in [0, 999_999]."
        
        assert self.args['verbose'] in range(0, 3), "Verbosity must be an integer in [0, 2]."
        
        # misc
        if self.args['optimizer'] == 'GA':
            assert dim == int(sqrt(dim) + .5) ** 2, "Dimension must be a perfect square for GA."
        
        return


class ProgressBar:
    frames = [f'\033[32m\033[1m{s}\033[0m' for s in ['╀', '╄', '┾', '╆', '╁', '╅', '┽', '╃']]   # spinner frames
    done_char = '\033[32m\033[1m━\033[0m'   # green bold ━, reset after
    todo_char = '\033[31m\033[2m─\033[0m'   # red faint ─, reset after
    spin_frame = 0

    def __init__(self, n_iters: int, p_id: str) -> None:
        self.n_iters = n_iters
        self.len_n_iters = len(str(n_iters))
        print(p_id)
        print('\r' + 50 * self.todo_char + ' ' + self.frames[0] + ' 0%', end='')
        self.start_ts = perf_counter()

    def __call__(self, iteration: int) -> None:
        """Updates and displays a progress bar on the command line"""
        percentage = 100 * (iteration+1) // self.n_iters            # floored percentage
        if percentage == 100 * iteration // self.n_iters: return    # prevent printing same line multiple times
        steps = 50 * (iteration+1) // self.n_iters                  # chars representing progress
        self.spin_frame += 1

        spin_char = self.frames[self.spin_frame%8]
        bar = (steps)*self.done_char + (50-steps)*self.todo_char        # the actual bar
        
        runtime = perf_counter() - self.start_ts
        if iteration+1 == self.n_iters:             # flush last suffix with spaces and place carriage at newline
            suffix = ' completed in ' + f'{runtime:.2f} sec'  + ' ' * 50 + '\n'
        else:                                       # print iteration number
            percentage_float = (100 * (iteration+1) / self.n_iters)
            eta = (100-percentage_float) / percentage_float * runtime
            suffix = f' {str(iteration+1).zfill(self.len_n_iters)}/{self.n_iters} (ETA {eta:.1f} sec) '
        
        print('\r' + bar + suffix, end='')
        return
