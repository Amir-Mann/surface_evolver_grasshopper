
if RUN_ON_CHANGE:
    import importlib.util
    import sys
    import os
    import rhinoscriptsyntax as rs

    if not VOLUME_FACTOR:
        VOLUME_FACTOR = 0.5
    if not INTER_ACTIVE:
        INTER_ACTIVE = False
    if not G_INPUT:
        G_INPUT = 20
    if not R_INPUT:
        R_INPUT = 2
    if not BASE_PATH:
        BASE_PATH = r"C:\Evolver"
    arguments = {
        "VOLUME_FACTOR": VOLUME_FACTOR,
        "INTER_ACTIVE": INTER_ACTIVE,
        "G_INPUT": G_INPUT,
        "R_INPUT": R_INPUT,
        "BASE_PATH": BASE_PATH,
        "input_mesh": input_mesh,
        "result_mesh": {"verts":[], "faces":[]}
    }
    
    sys.path.append(os.path.join(BASE_PATH,"surface_evolver_grasshopper"))
    sys.path.append(os.path.join(BASE_PATH,"surface_evolver_grasshopper", "py_lib"))
    import main
    main.run_SE(arguments)
    result_mesh = rs.AddMesh(arguments["result_mesh"]["verts"], arguments["result_mesh"]["faces"])
else:
    print("Not running script until 'RUN_ON_CHANGE'")