import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.case1_dataset_builder import build_predictor_dataset_parallel, save_predictor_dataset, validate_dataset_shapes, validate_dataset_labels
from src.config import make_config
from src.simulate import build_robot, make_reference, make_simulator    

cfg = make_config(
    urdf="xarm6.urdf",
    dt=0.001,
    T=20.0,
    D=0.2,
    Ts=0.05,
    predictor_tolerance=1e-8,
    max_picard_iters=50,
    inner_predictor_discretization_steps=4,
)


dataset = build_predictor_dataset_parallel(
    cfg,
    n_rollouts=20,
    stride=20,
    seed=0,
    max_workers=10,
)

save_predictor_dataset(dataset, cfg, "dataset/singlestep_predictor_dataset_small.npz")
robot = build_robot(cfg["urdf"])
ref = make_reference(robot, cfg)
sim = make_simulator(robot, cfg, ref)
validate_dataset_shapes(dataset, robot, cfg)
validate_dataset_labels(dataset, sim, cfg)