
import sys
import os
import datetime
import rhinoscriptsyntax as rs
if not base_path:
    base_path = r"C:\Evolver"
sys.path.append(os.path.join(base_path,"surface_evolver_grasshopper"))
sys.path.append(os.path.join(base_path,"surface_evolver_grasshopper", "py_lib"))
import se_grasshopper_entery
se_grasshopper_entery.reload_all_modules()

if run_on_change and input_mesh:


    if not volume_factor:
        volume_factor = 0.5
    if not interactive:
        interactive = False
    if not g_input:
        g_input = 20
    if not r_input:
        r_input = 2
    if not input_boundary_conditions:
        input_boundary_conditions = []

    arguments = {
        "VOLUME_FACTOR": volume_factor,
        "INTER_ACTIVE": interactive,
        "G_INPUT": g_input,
        "R_INPUT": r_input,
        "BASE_PATH": base_path,
        "input_mesh": input_mesh,
        "input_boundary_conditions": input_boundary_conditions,
        "result_mesh": {"verts":[], "faces":[]},
        "result_fixed": {"verts":[], "faces":[]}
    }
    
    se_grasshopper_entery.run_SE(arguments)
    result_mesh = rs.AddMesh(arguments["result_mesh"]["verts"], arguments["result_mesh"]["faces"])
    if arguments["result_fixed"]["verts"] and arguments["result_fixed"]["faces"] :
        result_fixed = rs.AddMesh(arguments["result_fixed"]["verts"], arguments["result_fixed"]["faces"])
else:
    print("Not running script until 'RUN_ON_CHANGE' is true and 'input_mesh' is set")
