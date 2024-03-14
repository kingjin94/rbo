#!/bin/sh

RED='\033[0;31m'
NC='\033[0m' # No Color

printf "${RED}Adding ${PWD} to git safe directory${NC}\n"
git config --global --add safe.directory $PWD  # TODO Current directory
printf "${RED}Updating base opt, mcs, and timor-python${NC} before running $@"
pip install -r requirements.txt  # Update base opt, mcs, timor
exec "$@"
