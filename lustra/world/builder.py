import math
import os
import random

import pybullet as p


class WorldBuilder:
    def __init__(self, assets_dir):
        self.assets_dir = assets_dir
        self.area_size = 200
        self.placed_positions = []
        self.seed_x = random.uniform(0, 1000)
        self.seed_y = random.uniform(0, 1000)

    def spawn_asset_by_type(self, asset_type, position):
        mapping = {
            "oak": "oak/oak.urdf",
            "pinusbruita": "pinusbruita/pinusbruita.urdf",
            "tree": "tree/tree.urdf",
            "rock": random.choice(["smallrock/smallrock.urdf", "mediumrock/mediumrock.urdf", "bigrock/bigrock.urdf"]),
            "bush": "bush/bush.urdf",
            "fire": "fire/fire.urdf",
        }

        if asset_type in mapping:
            urdf_path = os.path.join(self.assets_dir, mapping[asset_type])
            rot = p.getQuaternionFromEuler([0, 0, random.uniform(0, 6.28)])
            return self.load_custom_object(urdf_path, position=position, orientation=rot)
        return None

    def build_biome_world(self, tile_size=4, grid_range=20):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        grass_tex = p.loadTexture(os.path.join(self.assets_dir, "grass.png"))
        dirt_tex = p.loadTexture(os.path.join(self.assets_dir, "dirt.png"))

        grass_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[tile_size / 2, tile_size / 2, 0.1],
            rgbaColor=[1, 1, 1, 1],
        )

        dirt_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[tile_size / 2, tile_size / 2, 0.1],
            rgbaColor=[1, 1, 1, 1],
        )

        collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[tile_size / 2, tile_size / 2, 0.1])

        for x in range(-grid_range, grid_range):
            for y in range(-grid_range, grid_range):
                wx, wy = x * tile_size, y * tile_size

                layer1 = (math.sin((wx + self.seed_x) / 40.0) + math.cos((wy + self.seed_y) / 40.0))
                layer2 = (math.sin((wx + self.seed_x) / 8.0) + math.cos((wy + self.seed_y) / 12.0)) * 0.5
                moisture = (layer1 + layer2) / 2.0
                is_wet = moisture > -0.2

                if is_wet:
                    use_v_id = grass_visual
                    use_tex = grass_tex
                    use_color = [0.34, 0.49, 0.27, 1]
                else:
                    use_v_id = dirt_visual
                    use_tex = dirt_tex
                    use_color = [0.45, 0.35, 0.23, 1]

                random_yaw = random.choice([0, 1.57, 3.14, 4.71])
                rand_ori = p.getQuaternionFromEuler([0, 0, random_yaw])
                tile_id = p.createMultiBody(0, collision_id, use_v_id, [wx, wy, -0.1], rand_ori)
                p.changeVisualShape(tile_id, -1, textureUniqueId=use_tex, rgbaColor=use_color)

                roll = random.random()
                if is_wet:
                    if roll < 0.25:
                        asset = random.choice(["oak", "bush", "pinusbruita"])
                        self.spawn_asset_by_type(asset, [wx, wy, 0])
                else:
                    if roll < 0.15:
                        if roll < 0.05:
                            self.spawn_asset_by_type("tree", [wx, wy, 0])
                        else:
                            asset = random.choice(["rock", "bush"])
                            self.spawn_asset_by_type(asset, [wx, wy, 0])

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    def setup_base_world(self):
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(self.assets_dir)
        return None

    def load_custom_object(self, urdf_filename, position=[0, 0, 0], orientation=[0, 0, 0, 1]):
        try:
            obj_id = p.loadURDF(
                urdf_filename,
                basePosition=position,
                baseOrientation=orientation,
                useFixedBase=True,
            )

            obj_folder = os.path.dirname(urdf_filename)
            base_name = os.path.basename(urdf_filename).lower()

            texture_rules = {
                "oak": [{"file": "trunk.png", "keywords": ["trunk"]}, {"file": "leaf_pattern.png", "keywords": ["leaf", "leaves"]}],
                "pinusbruita": [{"file": "trunk.png", "keywords": ["trunk"]}, {"file": "leaf_pattern.png", "keywords": ["leaf", "leaves"]}],
                "tree": [{"file": "trunk.png", "keywords": [""]}],
                "rock": [{"file": "rock.png", "keywords": [""]}],
                "bush": [{"file": "bush.png", "keywords": [""]}],
            }

            active_rule = next((rules for key, rules in texture_rules.items() if key in base_name), None)

            if active_rule:
                visual_data = p.getVisualShapeData(obj_id)
                for rule in active_rule:
                    tex_path = os.path.join(obj_folder, rule["file"])
                    if not os.path.exists(tex_path):
                        continue

                    tex_id = p.loadTexture(tex_path)
                    for part in visual_data:
                        link_index = part[1]
                        part_name = part[4].decode("utf-8").lower()
                        if any(k in part_name for k in rule["keywords"]):
                            p.changeVisualShape(obj_id, link_index, textureUniqueId=tex_id, rgbaColor=[1, 1, 1, 1])

            return obj_id
        except Exception as e:
            print(f"Error loading {urdf_filename}: {e}")
            return None

    def spawn_fire(self, center_pos, offset=1.5):
        plus_offsets = [
            [0, 0],           
            [offset, 0],      
            [-offset, 0],     
            [0, offset],      
            [0, -offset]      
        ]
        
        spawned_ids = []
        
        for dx, dy in plus_offsets:
            spawn_pos = [center_pos[0] + dx, center_pos[1] + dy, center_pos[2]]
            
            fire_id = self.spawn_asset_by_type("fire", spawn_pos)
            
            if fire_id is not None:
                spawned_ids.append(fire_id)
                
        return spawned_ids

