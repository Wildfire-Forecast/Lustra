import inspect
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectPaths:
    root_dir: str
    assets_dir: str
    captured_images_dir: str
    model_path: str


def get_project_paths() -> ProjectPaths:
    package_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    root_dir = os.path.dirname(package_dir)
    return ProjectPaths(
        root_dir=root_dir,
        assets_dir=os.path.join(root_dir, "assets_new"),
        captured_images_dir=os.path.join(root_dir, "captured_images"),
        model_path=os.path.join(root_dir, "last.pt"),
    )

