from typing import Any, Dict

import optuna


def sample_ga_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Larger Hyperparameter space to find good search space for GA."""
    population_size = trial.suggest_int('population_size', 5, 50)
    num_parents_mating = trial.suggest_int('num_parents_mating', 1, population_size)
    return {
        'population_size': population_size,
        'num_generations': trial.suggest_int('num_generations', 10, 400),
        'mutation_probability': trial.suggest_float('mutation_probability', 0.005, 0.3),
        'num_parents_mating': num_parents_mating,
        'crossover_type': trial.suggest_categorical('crossover_type', ['single_point', 'two_points']),
        'keep_elitism': trial.suggest_int('keep_elitism', 1, 5),
        'save_best_solutions': True,
        'save_solutions': True,
        'keep_parents': trial.suggest_int('keep_parents', -1, num_parents_mating)
    }


def sample_bo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Larger Hyperparameter space to find good search space for BO."""
    ret = {
        "acq_func": trial.suggest_categorical("acq_func", ["EI", "LCB", "PI", "gp_hedge", "EIps", "PIps"]),
        "n_initial_points": trial.suggest_int("n_inital_points", 5, 100),
        "initial_point_generator": trial.suggest_categorical("initial_point_generator",
                                                             ["random", "sobol", "halton", "hammersly", "lhs"]),
        "ask_strategy": trial.suggest_categorical("ask_strategy", ["cl_min", "cl_mean", "cl_max", "default"]),
        "base_estimator": "GP"  # trial.suggest_categorical("base_estimator", ["GP", "RF", "ET", "GBRT"])
    }
    if ret["base_estimator"] in {"GBRT", "RF", "ET"}:
        ret["acq_optimizer"] = "sampling"
    else:
        ret["acq_optimizer"] = trial.suggest_categorical("acq_optimizer", ["sampling", "lbfgs", "auto"])
    if ret["ask_strategy"] != "default":  # With batch size 1 there are no cl_* strategies
        ret["batch_size"] = trial.suggest_int("batch_size", 2, 50)
    if ret["acq_func"] == "LCB":
        ret["acq_func_kwargs"] = {"kappa": trial.suggest_float("kappa", 1.5, 2.5)}
    if ret["acq_func"] in {"EI", "PI", "EIps", "PIps"}:
        ret["acq_func_kwargs"] = {"xi": trial.suggest_float("xi", 1e-4, 1e-1)}
    return ret


algorithm_to_search_space = {
    "GAOptimizer": sample_ga_params,
    "BOOptimizer": sample_bo_params,
}
