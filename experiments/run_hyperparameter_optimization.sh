# /bin/bash
# Run both of these from the main folder with ./experiments/run_hyperparameter_optimization.sh

python src/base_opt/base_opt/hyperparameter_optimization.py \
  --eval-set eval \
  --n-trials 400 \
  --parallel-trials 10 \
  --timeout 240 \
  --storage sqlite:///data/optuna.db \
  --algorithm GAOptimizer \
  --study-name GAopimization -optimize

python src/base_opt/base_opt/hyperparameter_optimization.py \
  --eval-set eval \
  --n-trials 400 \
  --parallel-trials 10 \
  --timeout 240 \
  --storage sqlite:///data/optuna.db \
  --algorithm BOOptimizer \
  --study-name BOopimization -optimize

python src/base_opt/base_opt/hyperparameter_optimization.py \
  --eval-set eval \
  --n-trials 400 \
  --parallel-trials 40 \
  --timeout 240 \
  --storage sqlite:///data/optuna.db \
  --algorithm AdamOptimizer \
  --study-name AdamOpimization_prune_longer -optimize
