Keep this folder and its parents in the repository even if they are empty, so coverage reports can be built locally.

If you run (a subset) of the coverage commands:
```bbash
 - coverage run --rcfile=../.coveragerc -m pytest tests/
 - coverage report --rcfile=../.coveragerc
 - coverage html -i --rcfile ../.coveragerc --data-file ../ci/coverage/report.coverage
 - coverage xml -i --rcfile ../.coveragerc --data-file ../ci/coverage/report.coverage
```
the according reports will be created here.
