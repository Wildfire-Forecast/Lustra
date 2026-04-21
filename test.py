import pybullet as p
import pybullet_data
from world_builder import WorldBuilder
import time
import os
import inspect
import numpy as np


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

assets_dir = os.path.join(current_dir, "assets_new")

# --- PyBullet Setup ---
cid = p.connect(p.GUI, options="--disable-example-browser")

# Hushing the b3Printf chatter (doesnt work)
p.setInternalSimFlags(0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

p.setGravity(0, 0, -9.8)
data_path = pybullet_data.getDataPath()
p.setAdditionalSearchPath(assets_dir)


# --- Initialize World Builder ---
wb = WorldBuilder(assets_dir)
wb.setup_base_world()
wb.build_biome_world(tile_size=4, grid_range=25)
wb.spawn_fire(center_pos=[5, 5, 1])


cam_target = [8.0, 8.0, 0.0]
p.resetDebugVisualizerCamera(
    cameraDistance=40,
    cameraYaw=45,
    cameraPitch=-35,
    cameraTargetPosition=cam_target
)

print("------ World Building Mode -----")
print("Press 'q' in the PyBullet window to exit.")

# Simplified Main Loop
try:
    while True:
        p.stepSimulation()
        
        keys = p.getKeyboardEvents()
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            break
            
        time.sleep(1 / 240)
except KeyboardInterrupt:
    pass

p.disconnect()