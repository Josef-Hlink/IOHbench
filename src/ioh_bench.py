import os
import argparse
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ioh import get_problem

from genetic_algorithm import GeneticAlgorithm
from evolution_strategies import EvolutionStrategies
from utils import get_directories, ParseWrapper, ProgressBar


def main():
    
    global DIRS, ARGS, PROB
    DIRS = get_directories(__file__)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ARGS = ParseWrapper(parser)()
    
    # --- FIXME ES is somehow hardcoded into path ----
    if ARGS['plot']:
        plot_target = 'ES' + os.sep + ARGS['run_id']
        plots_path = DIRS['plots'] + plot_target
        if not os.path.exists(plots_path):
            os.mkdir(plots_path)
        DIRS['plots'] = plots_path + os.sep
    
    np.random.seed(ARGS['seed'])
    problem_type = 'PBO' if ARGS['optimizer'] == 'GA' else 'Real'
    if ARGS['dimension'] is not None:
        PROB = get_problem(ARGS['pid'], dimension=ARGS['dimension'], problem_type=problem_type)
    else:
        PROB = get_problem(ARGS['pid'], problem_type=problem_type)

    print(f'{ARGS["optimizer"]}: {PROB.meta_data.name}')
    if ARGS['verbose'] == 1 and ARGS['repetitions'] > 1:
        progress_1 = ProgressBar(ARGS['repetitions'], ARGS['run_id'])
    
    df = pd.DataFrame(columns=list(range(ARGS['repetitions'])))
    df.index.name = 'generation'

    tic = perf_counter()
    for i in range(ARGS['repetitions']):
        
        res = run_experiment(i, ARGS['repetitions'])  # no real args, because everything is already in global ARGS
        df[i] = res

        if ARGS['verbose'] == 1 and ARGS['repetitions'] > 1:
            progress_1(i)
        
    toc = perf_counter()
    print(f'\nTotal time elapsed: {toc - tic:.3f} seconds')

    if ARGS['plot']:
        fig = create_plot(df)
        fig.savefig(DIRS['plots'] + f'{ARGS["run_id"]}_{ARGS["pid"]}.png')
    # --- TODO rename this to something like save --- #
    if ARGS['overwrite']:
        df.to_csv(DIRS['csv'] + f'{ARGS["run_id"]}_{ARGS["pid"]}.csv', index=True)
    # ----------------------------------------------- #

def run_experiment(i: int, n_reps: int) -> pd.Series:
    
    history = pd.Series(dtype=np.float64)
    history.index.name = 'generation'

    if ARGS['optimizer'] == 'GA':
        optimizer = GeneticAlgorithm(
            problem = PROB,
            pop_size = ARGS['population_size'],
            mu_ = ARGS['mu_'],
            lambda_ = ARGS['lambda_'],
            budget = ARGS['budget'],
            minimize = ARGS['minimize'],
            selection = ARGS['selection'],
            recombination = ARGS['recombination'],
            mutation = ARGS['mutation'],
            xp = ARGS['xp'],
            mut_rate = ARGS['mut_r'],
            mut_nb = ARGS['mut_b'],
            run_id = f'Repetition {i+1}/{n_reps}...',
            verbose = True if ARGS['verbose'] == 2 else False
        )
    else:  # ES
        optimizer = EvolutionStrategies(
            problem = PROB,
            pop_size = ARGS['population_size'],
            mu_ = ARGS['mu_'],
            lambda_ = ARGS['lambda_'],
            tau_ = ARGS['tau_'],
            sigma_ = ARGS['sigma_'],
            budget = ARGS['budget'],
            minimize = ARGS['minimize'],
            recombination = ARGS['recombination'],
            individual_sigmas = ARGS['individual_sigmas'],
            run_id = f'Repetition {i+1}/{n_reps}...',
            verbose = True if ARGS['verbose'] == 2 else False
        )
    _, _, history = optimizer.optimize(return_history=True)
    return history


def create_plot(df: pd.DataFrame) -> plt.Figure:
    
    fig, ax = plt.subplots()
    ax.set_title(f'{ARGS["optimizer"]}: {PROB.meta_data.name}')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_yscale('log')
    ax.grid(True)
    ax.plot(df.mean(axis=1), label='Mean')
    ax.plot(df.median(axis=1), label='Median')
    ax.plot(df.min(axis=1), label='Min')
    ax.plot(df.max(axis=1), label='Max')
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    main()
