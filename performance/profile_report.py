import time
import torch
import numpy as np

from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy


data_config = load_data_config("fourier_gr1_arms_waist")
modality_config = data_config.modality_config()
modality_transform = data_config.transform()

policy = Gr00tPolicy(
    model_path="/root/models/gr00t",
    modality_config=modality_config,
    modality_transform=modality_transform,
    embodiment_tag="gr1",
    denoising_steps=4,
)

test_obs = {
    "video.ego_view": np.zeros((1, 256, 256, 3), dtype=np.uint8),
    "state.left_arm": np.random.rand(1, 7),
    "state.right_arm": np.random.rand(1, 7),
    "state.left_hand": np.random.rand(1, 6),
    "state.right_hand": np.random.rand(1, 6),
    "state.waist": np.random.rand(1, 3),
    "annotation.human.action.task_description": ["do your thing!"],
}

for _ in range(10):
    policy.get_action(test_obs)


with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    policy.get_action(test_obs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
