import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D


input_dir = "/scratch/pioneer/users/mxh1029/dump/kv_cache/kv_cache_hf"  
output_dir = os.path.join(os.getcwd(), "figures_all_heads_topk")
os.makedirs(output_dir, exist_ok=True)

topk = 2            # Top-k feature dimensions highlighted in each head
max_tokens = 8      # Visualize the first few tokens

pt_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".pt"))

for pt_file in pt_files:
    file_path = os.path.join(input_dir, pt_file)
    print(f" Processing {pt_file} ...")

    try:
        data = torch.load(file_path, map_location="cpu")
    except Exception as e:
        print(f" failed to load {pt_file}: {e}")
        continue

    if not (isinstance(data, list) and isinstance(data[0], tuple)):
        print(f" Unexpected structure in {pt_file}, skipping.")
        continue

    try:
        key_tensor = data[0][0]  # [1, num_heads, seq_len, head_dim]
        key_tensor = key_tensor.squeeze(0).float()  
        num_heads, seq_len, head_dim = key_tensor.shape
        token_indices = list(range(min(seq_len, max_tokens)))

        for head_index in range(num_heads):
            activation = key_tensor[head_index]  
            abs_activation = activation.abs()    

            max_magnitudes = abs_activation.max(dim=0).values
            top_dims = torch.topk(max_magnitudes, topk).indices.tolist()

            dz_bg = np.zeros((head_dim, len(token_indices)))
            for dim_idx in top_dims:
                dz_bg[dim_idx, :] = abs_activation[token_indices, dim_idx].numpy()

            # Background gray colunm
            X_bg, Y_bg = np.meshgrid(token_indices, np.arange(head_dim))
            Z_bg = np.zeros_like(X_bg)
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.bar3d(X_bg.ravel(), Y_bg.ravel(), Z_bg.ravel(),
                     0.4, 0.4, dz_bg.ravel(), color='lightgray', alpha=0.5)

            # Blue highlighted column
            X_top, Y_top, Z_top = [], [], []
            for token_idx in token_indices:
                for dim_idx in top_dims:
                    X_top.append(token_idx)
                    Y_top.append(dim_idx)
                    Z_top.append(abs_activation[token_idx, dim_idx].item())
            ax.bar3d(X_top, Y_top, np.zeros_like(Z_top),
                     0.4, 0.4, Z_top, color='royalblue', alpha=1.0)

            
            ax.set_xlabel("Token Index", fontsize=10)
            ax.set_ylabel("Feature Dim", fontsize=10)
            ax.set_zlabel("Activation Magnitude", fontsize=10)
            ax.set_title(f"{pt_file} - Head {head_index}", fontsize=11)
            ax.set_xticks(token_indices)
            ax.set_yticks(sorted(top_dims))
            ax.view_init(elev=30, azim=120)

            save_path = os.path.join(output_dir, f"{pt_file.replace('.pt','')}_head{head_index}_top{topk}.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f" Saved Head {head_index} to {save_path}")

    except Exception as e:
        print(f" Error processing {pt_file}: {e}")
