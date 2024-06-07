

import os
import pathlib


R_INPUT = 2
G_INPUT = 20
TEMP_FE_PATH = "temp_fe_file_for_grasshopper_script.fe"
TEMP_DMP_PATH = "temp_fe_file_for_grasshopper_script.dmp"
SE_PATH = r"C:\Evolver\evolver.exe"
LEFT_QURLY_BRACKET = r"{"
RIGHT_QURLY_BRACKET = r"}"
INTER_ACTIVE = False
SAMPLE_TEXT = """
vertices //     coordinates
  1      0.877383  0.000000  1.000000
  2      0.877383  0.000000  0.000000
  3      0.877383  0.000000  -1.000000
  4      2.057645  0.000000  -1.000000
  5      2.057645  0.000000  1.000000
  6      -0.438691  -0.759836  1.000000
  7      -0.438691  -0.759836  0.000000
  8      -0.438691  -0.759836  -1.000000
  9      -1.028822  -1.781973  -1.000000
 10      -1.028822  -1.781973  1.000000
 11      -0.438691  0.759836  1.000000
 12      -0.438691  0.759836  0.000000
 13      -0.438691  0.759836  -1.000000
 14      -1.028822  1.781973  -1.000000
 15      -1.028822  1.781973  1.000000

edges  // endpoints
  1       1    2
  2       2    3
  3       3    4
  4       4    5
  5       5    1
  6       1    6
  7       2    7
  8       3    8
  9       4    9
 10       5   10
 11       6    7
 12       7    8
 13       8    9
 14       9   10
 15      10    6
 16       6   11
 17       7   12
 18       8   13
 19       9   14
 20      10   15
 21      11   12
 22      12   13
 23      13   14
 24      14   15
 25      15   11
 26      11    1
 27      12    2
 28      13    3
 29      14    4
 30      15    5

faces //   edges    
  1      1   2   3   4   5 
  2      6  11  -7  -1 
  3      7  12  -8  -2 
  4      8  13  -9  -3 
  5     10 -14  -9   4 
  6      6 -15 -10   5 
  7     11  12  13  14  15 
  8     16  21 -17 -11 
  9     17  22 -18 -12 
 10     18  23 -19 -13 
 11     20 -24 -19  14 
 12     16 -25 -20  15 
 13     21  22  23  24  25 
 14     26   1 -27 -21 
 15     27   2 -28 -22 
 16     28   3 -29 -23 
 17     30  -4 -29  24 
 18     26  -5 -30  25 
 19    -26  -16   -6  
 20    -27  -17   -7  
 21    -28  -18   -8  

bodies    //     facets 
  1     -1   -2   -3   -4    5    6    7     volume  3.000000  
  2     -7   -8   -9  -10   11   12   13     volume  3.000000  
  3    -13  -14  -15  -16   17   18    1     volume  3.000000  
  4     19    2    8   14  -20     volume  1.000000
  5     20    3    9   15  -21     volume  1.000000

"""

def get_mesh_topology_for_fe():
    return SAMPLE_TEXT

def get_fe_str():
    fe_file_str = get_mesh_topology_for_fe()

    fe_file_str += f"read // Take and run SE commands from this file\n"
    if INTER_ACTIVE:
        fe_file_str += f"s // Open graphics window\n q // quit graphics command window\n"

    fe_file_str += f"optimize_step := {LEFT_QURLY_BRACKET} g; // A general function looking for minimum\n"
    fe_file_str += f"    g {G_INPUT};\n"
    fe_file_str += f"    t .2;\n"
    fe_file_str += f"    V 5;\n"
    fe_file_str += f"    hessian_seek;\n"
    fe_file_str += f"    hessian_seek;\n"
    fe_file_str += f"{RIGHT_QURLY_BRACKET}\n"

    for r in range(R_INPUT):
        fe_file_str += "r; // refine edges \n" if r != 0 else "u; // equiangulation, tries to polish up the triangulation.\n"
        fe_file_str += "optimize_step;\n"

    fe_file_str += f"g {G_INPUT}; // finall settling down\n"
    fe_file_str += f'dump "{TEMP_DMP_PATH}" // Save results\n'
    if not INTER_ACTIVE:
        fe_file_str += "quit;\n"
        fe_file_str += "q;\n"
    return fe_file_str

def clean_temps():
    return
    if os.path.isfile(TEMP_FE_PATH):
        pathlib.Path.unlink(TEMP_FE_PATH)
    if os.path.isfile(TEMP_DMP_PATH):
        pathlib.Path.unlink(TEMP_DMP_PATH)

def run_SE():
    fe_file_str = get_fe_str()
    with open(f"{TEMP_FE_PATH}", "w") as temp_fe:
        temp_fe.write(fe_file_str)

    os.system(f"{SE_PATH} {TEMP_FE_PATH}")

    if not os.path.isfile(TEMP_DMP_PATH):
        print("Surface Evolver Failed")
        clean_temps()
        return

    with open(f"{TEMP_DMP_PATH}", "r") as temp_dmp:
        results_text = temp_dmp.read()
    
    send_back_to_grasshopper(results_text)
    clean_temps()
    return

def send_back_to_grasshopper(results_text):
    pass 

run_SE()