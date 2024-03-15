# Simple test set evaluation (total 30 * 2 * 4 * 5 = 1200 = ~80h)
parallel --bar --header : --results data/outdir_test_1200/outdir_simple_1200 -j 10 python src/base_opt/base_opt/evaluate_algorithm.py \
  --eval-set test_simple \
  --algorithm {Alg} \
  --action-space {Space} \
  --cost-function cyc \
  --seed {Seed} \
  --raw-data data/outdir_test_1200/outdir_simple_1200/Space/{Space}/Alg/{Alg}/Seed/{Seed}_raw.csv \
  --solution-storage data/solutions/ \
  --timeout 1200 ::: Space xyz xyz_rotvec ::: Alg BOOptimizer GAOptimizer RandomBaseOptimizer DummyOptimizer ::: Seed {1..5} > /dev/null

# Hard/Real-world test set evaluation (total 127 * 2 * 4 * 5 = 5080 runs a 240 sec = ~339 h)
parallel --bar --header : --results data/outdir_test_1200/outdir_hard_1200 -j 40 python src/base_opt/base_opt/evaluate_algorithm.py \
  --eval-set {Set} \
  --algorithm {Alg} \
  --action-space {Space} \
  --cost-function cyc \
  --seed {Seed} \
  --raw-data data/outdir_test_1200/outdir_hard_1200/Space/{Space}/Set/{Set}/Alg/{Alg}/Seed/{Seed}_raw.csv \
  --solution-storage data/solutions/ \
  --timeout 1200 \
  --reward-fail -50 ::: Space xyz xyz_rotvec ::: Set test_hard test_realworld ::: Alg BOOptimizer GAOptimizer RandomBaseOptimizer DummyOptimizer ::: Seed {1..5} > /dev/null
