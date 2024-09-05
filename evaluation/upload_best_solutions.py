from cobra.solution.solution import submit_solution
from pathlib import Path
from tqdm import tqdm
from time import sleep


def main():
    files = list(Path(__file__).parent.parent.joinpath('data').joinpath('best_solutions').rglob('*.json'))
    for file in tqdm(files):
        print("Upload ", file)
        submit_solution(file, "matthias.mayer@tum.de")
        sleep(10)


if __name__ == '__main__':
    main()