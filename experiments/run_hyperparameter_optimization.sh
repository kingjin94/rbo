python src/base_opt/base_opt/hyperparameter_optimization.py \
  --eval-set eval \
  --n-trials 400 \
  --parallel-trials 10 \
  --storage sqlite:///../data/database.db \
  --algorithm GAOptimizer \
  --study-name GAopimization -optimize

python src/base_opt/base_opt/hyperparameter_optimization.py \
  --eval-set eval \
  --n-trials 400 \
  --parallel-trials 10 \
  --storage sqlite:///../data/database.db \
  --algorithm BOOptimizer \
  --study-name BOopimization -optimize
