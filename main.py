
import os
import subprocess
from py_lib.se_setup import get_fe_str
from py_lib.reconstruct_mesh import reconstruct_mesh

## Execution ##

def run_SE(arguments):
    
    arguments['TEMP_FE_PATH'] = os.path.join(arguments['BASE_PATH'], r"temp_fe_file_for_grasshopper_script.fe")
    arguments['TEMP_DMP_PATH'] = os.path.join(arguments['BASE_PATH'], r"temp_fe_file_for_grasshopper_script.dmp")
    arguments['SE_PATH'] = os.path.join(arguments['BASE_PATH'], r"evolver.exe")
        
    fe_file_str = get_fe_str(arguments)
    with open(f"{arguments['TEMP_FE_PATH']}", "w") as temp_fe:
        temp_fe.write(fe_file_str)

    subprocess.run([f"{arguments['SE_PATH']}", f"{arguments['TEMP_FE_PATH']}"])

    if not os.path.isfile(arguments['TEMP_DMP_PATH']):
        print("Surface Evolver Failed")
        #clean_temps()
        return

    with open(f"{arguments['TEMP_DMP_PATH']}", "r") as temp_dmp:
        results_text = temp_dmp.read()
    
    reconstruct_mesh(arguments, results_text)
    return 