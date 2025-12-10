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

total_time = 0
num_runs = 1000
for _ in range(num_runs):
    start = time.time()
    action = policy.get_action(test_obs)
    torch.cuda.synchronize()  # 等待 GPU 任务完成
    end = time.time()
    print(f"单次推理时间{(end - start)*1000}")
    total_time += (end - start)

avg_latency = total_time / num_runs * 1000  # 转换为毫秒
print(f"端到端平均延迟：{avg_latency:.2f} ms")
print(f"吞吐量：{1000/avg_latency:.2f} 帧/秒")
