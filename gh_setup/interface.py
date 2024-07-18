
import sys
import os
import datetime
import rhinoscriptsyntax as rs
if not BASE_PATH:
    BASE_PATH = r"C:\Evolver"
sys.path.append(os.path.join(BASE_PATH,"surface_evolver_grasshopper"))
sys.path.append(os.path.join(BASE_PATH,"surface_evolver_grasshopper", "py_lib"))
import se_grasshopper_entery
se_grasshopper_entery.reload_all_modules(BASE_PATH)

if RUN_ON_CHANGE and input_mesh:


    if not VOLUME_FACTOR:
        VOLUME_FACTOR = 0.5
    if not INTER_ACTIVE:
        INTER_ACTIVE = False
    if not G_INPUT:
        G_INPUT = 20
    if not R_INPUT:
        R_INPUT = 2
    arguments = {
        "VOLUME_FACTOR": VOLUME_FACTOR,
        "INTER_ACTIVE": INTER_ACTIVE,
        "G_INPUT": G_INPUT,
        "R_INPUT": R_INPUT,
        "BASE_PATH": BASE_PATH,
        "input_mesh": input_mesh,
        "result_mesh": {"verts":[], "faces":[]}
    }
    
    se_grasshopper_entery.run_SE(arguments)
    result_mesh = rs.AddMesh(arguments["result_mesh"]["verts"], arguments["result_mesh"]["faces"])
else:
    print("Not running script until 'RUN_ON_CHANGE' is true and 'input_mesh' is set")
