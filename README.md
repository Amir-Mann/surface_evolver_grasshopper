BASIC GH COMPONENT

How  to setup:
1. Place the evolver.exe in the path as follows:   C:\Evolver\evolver.exe
2. Open holes_script.gh from grasshopper. Can be found at C:\Evolver\surface_evolver_grasshopper\gh_setup
3. for other OS systems, use any similar pathing, and change the PathToEvolverDir component to the OS formated path.
4. Remember to disable the run_on_change button at time of changing conditions


How to use:
First iteration:
1. Load the appropriate mesh and connect it to the StartingShapeMesh
2. Recommended parameters:
    -GradientSteps: 20-40
    -RefinmentIterations: 3
    -VolumeFactor: 0.5

Second Iteration:
1. New boundary conditions: Choose few meshes and join them with the FixedFacesOutput from the first iteration.
    Connect them to BoundaryConditionsMesh(s).
2. Connect the FullShapeOutput from the first iteration to StartingShapeMesh
3. Recommended parameters:
    -GradientSteps: 20-40
    -RefinmentIterations: 1


Common bugs:
1. AttributeError: 'NoneType' object has no attribute 'faces'.
    Fix: Disconnect input_boundary_conditions mesh
2. The volume seems to really increase on the second run, try to decrease it as needed.
3. Second iteration not running: Maybe BoundaryConditionsMesh(s) is not included in StartingShapeMesh as required.
