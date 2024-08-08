

import os
import pathlib

from py_lib.load_mesh import get_mesh_topology_for_fe


LEFT_QURLY_BRACKET = r"{"
RIGHT_QURLY_BRACKET = r"}"


## Optimization ##

def get_fe_str(arguments):
    fe_file_str, estimated_volume = get_mesh_topology_for_fe(arguments)

    fe_file_str += f"read // Take and run SE commands from this file\n"
    fe_file_str += f"G 0; //\n"
    fe_file_str += f"set body target {-estimated_volume * 0.5 * arguments['VOLUME_FACTOR']} where id == 1 // Sets the volume\n"
    if arguments['INTER_ACTIVE']:
        fe_file_str += f"s // Open graphics window\nq\n"
        fe_file_str += f'read "{os.path.join(arguments["SEFIT_LIB_PATH"], "sefit_parameter_list.ses")}"\n'.replace('\\', '\\\\')
        fe_file_str += f'read "{os.path.join(arguments["SEFIT_LIB_PATH"], "computation_kernel.ses")}"\n'.replace('\\', '\\\\')
        fe_file_str += f'read "{os.path.join(arguments["SEFIT_LIB_PATH"], "eigenvalue_study.ses")}"\n'.replace('\\', '\\\\')
        fe_file_str += f'read "{os.path.join(arguments["SEFIT_LIB_PATH"], "convergence_operation.ses")}"\n'.replace('\\', '\\\\')
        
        fe_file_str += f''


    fe_file_str += f"optimize_step := {LEFT_QURLY_BRACKET} g; // A general function looking for minimum\n"
    fe_file_str += f"    g {arguments['G_INPUT']};\n"
    fe_file_str += f"    t .2;\n"
    fe_file_str += f"    V 5;\n"
    fe_file_str += f"    hessian_seek;\n"
    fe_file_str += f"    hessian_seek;\n"
    fe_file_str += f"    o;\n"
    fe_file_str += f"{RIGHT_QURLY_BRACKET}\n"
    
    path_for_fe = arguments['TEMP_DMP_PATH'].replace("\\", "\\\\")
    fe_file_str += f'dump_file := {LEFT_QURLY_BRACKET} dump "{path_for_fe}" {RIGHT_QURLY_BRACKET}\n'

    if not arguments['INTER_ACTIVE']:
        for r in range(arguments['R_INPUT']):
            fe_file_str += "r; // refine edges \n" if r != 0 else ""#"u; // equiangulation, tries to polish up the triangulation.\n"
            fe_file_str += "optimize_step;\n"

        fe_file_str += f"g {arguments['G_INPUT']}; // finall settling down\n"
        fe_file_str += "dump_file\n"
        fe_file_str += "q\n"
        fe_file_str += "q\n"
        
    
    return fe_file_str

def clean_temps(arguments):
    if os.path.isfile(arguments['TEMP_FE_PATH']):
        pathlib.Path.unlink(arguments['TEMP_FE_PATH'])
    if os.path.isfile(arguments['TEMP_DMP_PATH']):
        pathlib.Path.unlink(arguments['TEMP_DMP_PATH'])
