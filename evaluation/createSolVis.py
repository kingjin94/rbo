import json
from pathlib import Path

import numpy as np
import timor.task.Task
import timor.task.Solution
from timor.utilities.visualization import color_visualization

import cobra

sol = json.load(Path('sol_edge_case.json').open())
task_id = sol['taskID']
task_file = cobra.task.get_task(id=task_id)
task = timor.task.Task.Task.from_json_file(task_file)
sol = timor.task.Solution.SolutionTrajectory.from_json_data(sol, {task.id: task})

v = sol.visualize()
# Prettyfy, i.e., color modules
colors = {
    '105': np.array([156 / 255, 0 / 255, 0 / 255, 1.]),
    '24': np.array([127 / 255, 127 / 255, 127 / 255, 1.]),
    '25': np.array([127 / 255, 127 / 255, 127 / 255, 1.]),
    '2': np.array([248 / 255, 135 / 255, 1 / 255, 1.]),
    '1': np.array([248 / 255, 135 / 255, 1 / 255, 1.]),
    'GEP2010IL': np.array([250 / 255, 182 / 255, 1 / 255, 1.]),
}
color_visualization(v, sol.module_assembly, colors)
# Background white
settings = (
    ('/Background', "visible", False),
    ('/Lights/SpotLight', "visible", True),
    ('/Lights/SpotLight/<object>', "intensity", .2),
    ('/Lights/PointLightPositiveX', "visible", False),
    ('/Axes', "visible", False),
    ('/Grid', "visible", False),
)
for key, prop, value in settings:
    v.viewer[key].set_property(prop, value)

# Show trajectory
sol.trajectory.visualize(
    v, assembly=sol.module_assembly,
    line_color=np.broadcast_to(np.asarray([36, 196, 226]) / 255,
                               (2 * len(sol.trajectory) - 2, 3)).T)

print("""
Manual work:
* Find good perspective
* Use Save - save_image to store png version
* Optional: Stack multiple at these, e.g., to show the robot at each goal pose in GIMP/PS
* Optional: Overlay spline e.g. to highlight coordinate systems, the base pose tolerance, or trajectory in GIMP/inkscape
* Optional: Overlay labels with tikz in latex
""")

input("Press Enter to quit...")
