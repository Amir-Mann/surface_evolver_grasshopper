

import os
import pathlib

from py_lib.load_mesh import get_mesh_topology_for_fe


LEFT_QURLY_BRACKET = r"{"
RIGHT_QURLY_BRACKET = r"}"


## Optimization ##

def get_fe_str(arguments):
    fe_file_str, x_length, y_length, z_length = get_mesh_topology_for_fe(arguments)
    estimated_volume = x_length * y_length * z_length
    initial_target_length = (x_length ** 2 + y_length ** 2 + z_length ** 2) ** 0.5 / 5

    fe_file_str += f"read // Take and run SE commands from this file\n"
    fe_file_str += f"G 0; //\n"
    fe_file_str += f"set body target {-estimated_volume * 0.5 * arguments['VOLUME_FACTOR']} where id == 1 // Sets the volume\n"
    if arguments['INTER_ACTIVE']:
        fe_file_str += f"s // Open graphics window\nq\n"

    fe_file_str += f"optimize_step := {LEFT_QURLY_BRACKET} g; // A general function looking for minimum\n"
    fe_file_str += f"    g {arguments['G_INPUT']};\n"
    fe_file_str += f"    hessian_seek;\n"
    fe_file_str += f"    hessian_seek;\n"
    fe_file_str += f"    o;\n"
    fe_file_str += f"{RIGHT_QURLY_BRACKET}\n"
    fe_file_str += f"target_length := {initial_target_length:.2f};\n"
    fe_file_str += f"remesh_step := {LEFT_QURLY_BRACKET}t target_length/16*9; l target_length/16*25; V 2; u{RIGHT_QURLY_BRACKET}\n"

    if not arguments['INTER_ACTIVE']:
        for r in range(arguments['R_INPUT']):
            if r != 0:
                fe_file_str += "remesh_step; remesh_step;\n"
                fe_file_str += "target_length := target_length / 2;\n"
                fe_file_str += "remesh_step; remesh_step;\n"
            else:
                fe_file_str += "remesh_step; remesh_step;\n"
            fe_file_str += "optimize_step;\n"

        fe_file_str += f"g {arguments['G_INPUT']}; // finall settling down\n"
        path_for_fe = arguments['TEMP_DMP_PATH'].replace("\\", "\\\\")
        fe_file_str += f'dump "{path_for_fe}" // Save results\n'
    
        fe_file_str += "quit;\n"
        fe_file_str += "q;\n"
    return fe_file_str

def clean_temps(arguments):
    if os.path.isfile(arguments['TEMP_FE_PATH']):
        pathlib.Path.unlink(arguments['TEMP_FE_PATH'])
    if os.path.isfile(arguments['TEMP_DMP_PATH']):
        pathlib.Path.unlink(arguments['TEMP_DMP_PATH'])
