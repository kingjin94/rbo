import json
from pathlib import Path
import pickle

from timor.utilities.json_serialization_formatting import compress_json_vectors
from tqdm import tqdm


def main():
    """
    Prepare the best solutions for each algorithm and action space for upload to the cobra benchmark website.

    Sets version, author, email, affilitation, publication, notes, and tags
    """
    # Load solution dict
    with open(Path(__file__).parent.joinpath('best_solutions.pkl'), 'rb') as f:
        uuid_list = pickle.load(f)  # Dict Alg name, action_space to solution uuid

    for (alg, action_space), uuids in tqdm(uuid_list.items()):
        for uuid in uuids:
            print(f"Process {alg} {action_space} {uuid}")
            opt_domain = "Position" if action_space == 'xyz' else "Position + Orientation"
            # Load solution
            sol = json.load(Path(__file__).parent.joinpath('solutions').joinpath(f"solution-{uuid}.json").open('rb'))
            # Edit solution
            sol["version"] = "2022"
            sol["author"] = ["Matthias Mayer", ]
            sol["email"] = ["matthias.mayer@tum.de", ]
            sol["affiliation"] = ["Technical University of Munich", ]
            sol["publication"] = "Mayer et al., 'Decreasing Robotic Cycle Time via Base-Pose Optimization', 2024"
            sol["notes"] = f"This solution was found by {alg} optimizing on {opt_domain}"
            sol["tags"] = ["BPO24", "Base-Pose-Optimization", alg, opt_domain]
            # Save solution
            with Path(__file__).parent.joinpath('solutions_out').joinpath(f"solution-{uuid}.json").open('w') as f:
                f.write(compress_json_vectors(json.dumps(sol, indent=2)))


if __name__ == '__main__':
    main()
