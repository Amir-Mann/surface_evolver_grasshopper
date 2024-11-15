
# Fluid Simulation in Rhino using Surface Evolver

This project was created for the [Bercovici Fluidic Technologies LAB](https://www.fluidic.technology/) to enable fluid dynamics simulation within the 3D modeling software [Rhino](https://www.rhino3d.com/). The tool utilizes [Surface Evolver](https://kenbrakke.com/evolver/evolver.html) to predict the equilibrium state of a polymer bounded by specific solid boundary conditions, floating within another liquid, as demonstrated in [this video](https://www.youtube.com/watch?v=MZLemDf43mk).

![](https://github.com/Amir-Mann/surface_evolver_grasshopper/blob/4002bd9266f4472a42d0c997f43c14e80c786623/assets/FluidicPrintingDemo.png?raw=true)

*From left to right: An above view of the shape floating, a side view of the shape, and a view of the printed shape after being hardened and removed from its container. Credit: Bercovici Fluidic Technologies Lab.*

[Grasshopper](https://www.grasshopper3d.com/) is a graphical algorithm editor integrated with Rhino. The tool is implemented as a Grasshopper script available in this repository.

![](https://github.com/Amir-Mann/surface_evolver_grasshopper/blob/491eed9aee365237a63bf57a52bc0faa90627ac5/assets/FirstIteration.png?raw=true)

*From left to right: The input-conditioned shape to the tool and the output, settled shape.*

## Main Capabilities

The tool accepts as input a Rhino triangle mesh with boundaries represented either by holes in the mesh or by a separate mesh, where all faces are subsurfaces of the main mesh. 

The workflow is as follows:
1. The tool loads the mesh as the initial fluid state, with the boundaries serving as boundary conditions, into a Surface Evolver command language script.
2. The script incorporates:
   - The mesh itself.
   - A convergence algorithm to settle the fluid into its final form.
   - A remeshing algorithm based on [Botsch et al.](https://dl.acm.org/doi/abs/10.1145/1057432.1057457), applied during the convergence process.
   - Both algorithms are implemented directly in Surface Evolver's command language.
3. Surface Evolver runs as a subprocess of the Grasshopper script, ultimately outputting the resulting mesh to a file.
4. The resulting meshes (final state and boundary conditions for future iterations) are reloaded into Grasshopper, enabling direct baking back into Rhino's visual editor.

Thanks to the tool's flexible structure, it supports iterative workflows. For example, the settled shape can be used to impose new boundary conditions and adjust the shape's volume for subsequent simulations.

![](https://github.com/Amir-Mann/surface_evolver_grasshopper/blob/main/assets/FinalSmooth3.png?raw=true)

![](https://github.com/Amir-Mann/surface_evolver_grasshopper/blob/main/assets/FinalWireframe.png?raw=true)

*From left to right: The original shape input, the shape after one iteration of the tool with the new boundary condition marked in red, and the resulting shape after another iteration with reduced volume and the new condition.*

 ## How to Install the Tool
 
1. **Create a Base Directory**  
   Create an empty directory on your computer. For example, on Windows, you can create `C:\Evolver`. This directory will be referred to as `base_path`. Alternatively, this directory can be created automatically during the Surface Evolver installation in step 2 by specifying it as the installation path.

2. **Install Surface Evolver**  
   Download and install Surface Evolver from [the official site](https://kenbrakke.com/evolver/evolver.html#download), following the instructions provided on the website or by the installer.

3. **Download the Repository**  
   Download all the code from this repository into a folder named `surface_evolver_grasshopper` inside `base_path`. If you have git installed, this can be done by running the following command in the `base_path` directory:  
   ```bash
   git clone https://github.com/Amir-Mann/surface_evolver_grasshopper.git
   ```  
   Alternatively, you can download the repository as a zip file, extract it, and place the extracted folder in `base_path`.

4. **Verify the Setup**  
   Ensure that the following two files are present:  
   - `<base_path>\evolver.exe` or `<base_path>\evolver` (this opens the Evolver terminal).  
   - `<base_path>\surface_evolver_grasshopper\se_grasshopper_entery.py`.

5. **Open the Script in Grasshopper**  
   Open Rhino, then launch Grasshopper. From Grasshopper, open the script file located at:  
   `<base_path>\surface_evolver_grasshopper\gh_setup\se_tool.gh`

## How to Use the Tool

After loading the script into Grasshopper, you will see a "battery" that serves as the main tool interface.

### Input Fields

The battery has the following input fields:

1. **input_mesh**  
   The rough starting state of the fluid. This must be a Grasshopper mesh item and a triangular mesh. Topologically, the mesh can have any shape, but it is recommended to use simple spheres or prisms with few holes at most. For more complex shapes, it is advisable to use a separate mesh for boundary conditions. Note that the behavior of Surface Evolver with non-manifold meshes is unexplored.

2. **input_boundary_conditions**  
   Additional boundary conditions (other than holes in the input_mesh) where the fluid is constrained. This should be a Grasshopper mesh item, where all faces are subfaces of the input mesh. Rhino meshes do not need to be continuous, allowing multiple boundaries to be represented with a single mesh (rather than a Grasshopper mesh list). The default value is no additional boundary conditions.

3. **run_on_change**  
   A boolean specifying whether to run the simulation whenever inputs to the battery change. Typically, it is easier to turn this option off before making changes and turn it back on only when calculating the settled state. If left on, input changes may be slow because Grasshopper waits for the Surface Evolver process to complete before continuing GUI interactions. Note: After turning off `run_on_change`, the result meshes will become empty. It is recommended to bake (`Right-click → Bake → OK`) the results before disabling this option.

4. **g_input**  
   An integer specifying the number of gradient steps to take during each step of the convergence algorithm. Values between 10 and 20 usually suffice for simple shapes, while more complex shapes may require up to 100 for greater accuracy. The default value is 25.

5. **r_input**  
   An integer specifying the number of refinement steps the algorithm should take. Each refinement step halves the target size of each edge in the mesh. The original target size is:  
   $$\frac{||\text{mesh\_diagonal}||_2}{6} = \frac{\sqrt{(x_{\text{max}} - x_{\text{min}})^2 + (y_{\text{max}} - y_{\text{min}})^2 + (z_{\text{max}} - z_{\text{min}})^2}}{6}$$  
   Fluid convergence steps and remeshing are performed at each level of refinement. As a result, the runtime increases significantly with higher `r_input` values. The default value is 3.

6. **volume_factor**  
   A float specifying the volume of liquid to simulate. The target shape's volume in the simulator is calculated as:  
   $$0.5 \cdot \text{volume\_factor} \cdot (x_{\text{max}} - x_{\text{min}}) \cdot (y_{\text{max}} - y_{\text{min}}) \cdot (z_{\text{max}} - z_{\text{min}})$$  
   The default value is 1.

7. **interactive**  
   A boolean to enable interactive mode for Surface Evolver. See the "Interactive Mode" subsection below. The default value is `False`.

8. **base_path**  
   The path to the folder where the installation process was completed. The path should match the operating system's format. The default value is `C:\Evolver`.

### Output Fields

The battery provides the following output fields:

1. **out**  
   A plain text Grasshopper object containing the Python log of the run. Typical messages include ` ` (success) or `Not running the script until "run_on_change" is true and "input_mesh" is set.` when the script is waiting for inputs.

2. **result_mesh**  
   A Rhino mesh object representing the full resulting shape from the simulation. This shape typically forms one topological sphere or, in cases of separation during the simulation, a few topological spheres.

3. **result_fixed**  
   A Rhino mesh object containing all fixed faces provided as input, including those specified by `input_boundary_conditions` and holes filled in and set as boundaries.

### Interactive Mode

**Note:** The macOS version appears to have issues with this feature.

Interactive mode introduces a small modification to the Surface Evolver script, allowing it to run with built-in visualization. In this mode, the automatic convergence scheme and automatic exit are disabled. 

Interactive mode is useful for debugging the input mesh, examining the convergence process, and manually adjusting simulation properties like gravity or volume. 

When the script launches the Surface Evolver process in interactive mode, both a visualization window and a terminal window should appear. The terminal window will display the following help message:

```
This is an interactive mode for the MannHadad Surface Evolver plugin for Grasshopper
There should be an open window displaying the current state of the evolver, on light blue background.
When you finished optimizing your model, use 'dump_file' function to save, then you can safely close this window, or exit by writing the 'quit' command twice.
Until this process concludes, Grasshopper and Rhino would stay stuck.

Useful commands by the MannHadad:
    'optimize_step'     - Runs a single optimization step, towards the minimum energy shape. Ignores mesh quality.
    'remesh_step'       - Optimizes mesh quality, Should sattle down in 3-5 iterations but might break the optimal shape.
    'target_length := target_length / 2' - Making the target edge length smaller for the remeshing steps.
    'dump_file'         - Save
    'print_help'        - display this help message

Useful commands by Surface Evolver:
    'r'       - Divides all edges lengths by half.
    'g <num>' - Takes <num> gradient steps.
    'quit'    - quit the current simulation, use twice to exit the simulator all together.
    'set body target <volume> where id == 1' - Set the main body's volume to <volume> (in our testing usually a negative number around -25000).
    *see https://kenbrakke.com/evolver/html/commands.htm for more command language       documentation.
    *see https://kenbrakke.com/evolver/html/single.htm   for more single letter commands documentation.

Useful commands by SE-FIT:
    'MAX_STEPS := <num>'    - Set the maximum allowed iteration of a spesific convergence step inside SE-FIT to run. Added code to SE-FIT
    'ps_conv_run_rough'     - Run SE-FIT's rough convergence loop.
    'ps_conv_run_fine'      - Run SE-FIT's fine convergence loop.
    *see surface_evolver_grasshopper.se_lib.SE_FIT_lib.convergence_operation.ses for other operations and SE-FIT settings
    *DISCLAIMER: Stopping an SE-FIT function call using Ctrl-C can break the program, the SE-FIT Gui offers better interface for such fine control.
```

The [SE-FIT](https://www.se-fit.com/) commands are imported from their Surface Evolver code. All credit for these functionalities goes to them.

### Useful Rhino Tools:
1. **SelectionFilterFaces**  
   Allows selecting faces for the boundary mesh.

2. **ExtractMeshFaces**  
   Use with `MakeCopy = Yes`. This duplicates the selected faces as a sub-mesh, which can then be used for boundary conditions.

3. **Join**  
   Joins two meshes into one.

4. **TriangulateMesh**  
   Useful for converting a mesh into a triangular format.

5. **MeshSplit**  
   Useful for selecting a subpart of a mesh to use as a boundary condition.

6. **MeshIntersect**  
   Useful for isolating specific geometries within a mesh to use as boundary conditions.