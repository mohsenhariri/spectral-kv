import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

input_dir = "/scratch/pioneer/users/mxh1029/dump/kv_cache/kv_cache_hf"  
output_dir = os.path.join(os.getcwd(), "figures_surface_all_heads")
os.makedirs(output_dir, exist_ok=True)

pt_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".pt"))

for pt_file in pt_files:
    file_path = os.path.join(input_dir, pt_file)
    print(f" Processing {pt_file} ...")

    try:
        data = torch.load(file_path, map_location="cpu")
    except Exception as e:
        print(f"Failed to load {pt_file}: {e}")
        continue

    if not (isinstance(data, list) and isinstance(data[0], tuple)):
        print(f" Unexpected structure in {pt_file}, skipping.")
        continue

    try:
        key_tensor = data[0][0]  # [1, num_heads, seq_len, head_dim]
        key_tensor = key_tensor.squeeze(0).float()  # [num_heads, seq_len, head_dim]

        num_heads, seq_len, head_dim = key_tensor.shape

        for head_index in range(num_heads):
            activation = key_tensor[head_index]  
            act_np = activation.abs().numpy().T  

            X = np.arange(seq_len)
            Y = np.arange(head_dim)
            X, Y = np.meshgrid(X, Y)
            Z = act_np  # [head_dim, seq_len]

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)

            ax.set_xlabel("Sequence Pos", fontsize=10)
            ax.set_ylabel("Head Dim", fontsize=10)
            ax.set_zlabel("Activation Magnitude", fontsize=10)
            ax.set_title(f"{pt_file} - Head {head_index}", fontsize=12)

            ax.view_init(elev=30, azim=120)  

            save_path = os.path.join(output_dir, f"{pt_file.replace('.pt','')}_head{head_index}.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f" Saved head {head_index} to {save_path}")

    except Exception as e:
        print(f" Error processing {pt_file}: {e}")
