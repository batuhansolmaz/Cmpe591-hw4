import torch
import numpy as np
import matplotlib.pyplot as plt
from homework4 import CNP, Hw5Env, bezier
import os

def evaluate_cnp(model, trajectories, n_tests=100, n_context_max=10, n_target_max=10, device="cpu"):
    """
    model: your trained CNP/CNMP
    trajectories: list of np.arrays, each of shape (T, 5) = [e_y, e_z, o_y, o_z, height]
    Returns: two lists of MSEs (one per trial) for end-effector and for object.
    """
    mse_ee_list = []
    mse_obj_list = []
    model.eval()
    with torch.no_grad():
        for _ in range(n_tests):
            # pick random trajectory
            states = trajectories[np.random.randint(len(trajectories))]
            T = states.shape[0]
            # sample random sizes
            n_context = np.random.randint(1, min(n_context_max, T) + 1)
            n_target  = np.random.randint(1, min(n_target_max, T) + 1)
            # sample indices
            idx_all = np.arange(T)
            context_idx = np.random.choice(idx_all, size=n_context, replace=False)
            remain = np.setdiff1d(idx_all, context_idx)
            target_idx = np.random.choice(remain, size=n_target, replace=False)
            # build tensors
            # x = [t_normalized, height], y = [e_y, e_z, o_y, o_z]
            t = np.linspace(0, 1, T)  # normalize time to [0,1]
            # context
            cx = np.stack([t[context_idx], states[context_idx,4]], axis=1)    # (n_ctx,2)
            cy = states[context_idx, :4]                                      # (n_ctx,4)
            obs = np.concatenate([cx, cy], axis=1)[None]                      # (1,n_ctx,6)
            # target
            tx = np.stack([t[target_idx], states[target_idx,4]], axis=1)[None] # (1,n_tgt,2)
            ty = states[target_idx, :4][None]                                 # (1,n_tgt,4)

            # to torch
            obs_t   = torch.from_numpy(obs).float().to(device)
            tx_t    = torch.from_numpy(tx).float().to(device)
            ty_t    = torch.from_numpy(ty).float().to(device)

            # forward
            mean, std = model(obs_t, tx_t)               # mean: (1,n_tgt,4)
            # compute MSE per dimension
            err = (mean - ty_t).pow(2).mean(dim=2).squeeze(0)  # (n_tgt,)
            # but we want separate EE vs object:
            # dims 0,1 = EE ; dims 2,3 = object
            ee_errs  = (mean[..., 0:2] - ty_t[..., 0:2]).pow(2).mean(dim=(1,2)).item()
            obj_errs = (mean[..., 2:4] - ty_t[..., 2:4]).pow(2).mean(dim=(1,2)).item()

            mse_ee_list.append(ee_errs)
            mse_obj_list.append(obj_errs)

    return mse_ee_list, mse_obj_list

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1) Collect data
    data_file = 'demonstration_data.npz'
    if os.path.exists(data_file):
        print(f"Loading demonstrations from {data_file}")
        data = np.load(data_file, allow_pickle=True)
        states_arr = data['states_arr']
    else:
        try:
            print("Collecting demonstrations...")
            env = Hw5Env(render_mode="offscreen")
            states_arr = []
            for i in range(100):
                env.reset()
                p_1 = np.array([0.5, 0.3, 1.04])
                p_2 = np.array([0.5, 0.15, np.random.uniform(1.04, 1.4)])
                p_3 = np.array([0.5, -0.15, np.random.uniform(1.04, 1.4)])
                p_4 = np.array([0.5, -0.3, 1.04])
                points = np.stack([p_1, p_2, p_3, p_4], axis=0)
                curve = bezier(points)

                env._set_ee_in_cartesian(curve[0], rotation=[-90, 0, 180], n_splits=100, max_iters=100, threshold=0.05)
                states = []
                for p in curve:
                    env._set_ee_pose(p, rotation=[-90, 0, 180], max_iters=10)
                    states.append(env.high_level_state())
                states = np.stack(states)
                states_arr.append(states)
                print(f"Collected {i+1}/100 trajectories.", end="\r")
            
            print("\nDemonstration collection complete!")
            np.savez(data_file, states_arr=states_arr)
        except Exception as e:
            print(f"Error collecting demonstrations: {e}")
            print("Generating synthetic data...")
            
            # Generate synthetic data
            states_arr = []
            for i in range(100):
                # Random object height
                obj_height = np.random.uniform(0.03, 0.1)
                
                # Generate time steps and trajectory
                steps = 100
                time_steps = np.linspace(0, 1, steps)
                
                # End-effector trajectory
                ee_y = 0.3 * np.cos(time_steps * 2 * np.pi) 
                ee_z = 1.04 + 0.2 * np.sin(time_steps * 2 * np.pi)
                
                # Object trajectory
                obj_y = np.zeros_like(ee_y)
                obj_z = np.ones_like(ee_z) * 1.04
                
                # Simple physics: object moves when end-effector is close
                for t in range(1, steps):
                    dist = np.sqrt((ee_y[t-1] - obj_y[t-1])**2 + (ee_z[t-1] - obj_z[t-1])**2)
                    if dist < 0.15:
                        dir_y = ee_y[t-1] - obj_y[t-1]
                        dir_z = ee_z[t-1] - obj_z[t-1]
                        norm = np.sqrt(dir_y**2 + dir_z**2) + 1e-6
                        scale = 0.05 * (1.0 - obj_height / 0.1)
                        obj_y[t] = obj_y[t-1] + scale * dir_y / norm
                        obj_z[t] = obj_z[t-1] + scale * dir_z / norm
                    else:
                        obj_y[t] = obj_y[t-1]
                        obj_z[t] = obj_z[t-1]
                
                # Create states array [ee_y, ee_z, obj_y, obj_z, obj_height]
                states = np.column_stack([ee_y, ee_z, obj_y, obj_z, np.ones(steps) * obj_height])
                states_arr.append(states)
                print(f"Generated {i+1}/100 synthetic trajectories.", end="\r")
            
            print("\nSynthetic data generation complete!")
            np.savez("synthetic_data.npz", states_arr=states_arr)
    
    print(f"Loaded {len(states_arr)} trajectories")
    
    # 2) Instantiate model & optimizer
    cnp = CNP(in_shape=(2, 4), hidden_size=128, num_hidden_layers=3).to(device)
    optimizer = torch.optim.Adam(cnp.parameters(), lr=1e-3)
    
    # 3) Training hyperparams
    num_epochs = 50
    n_context_max = 10
    n_target_max = 10
    
    # 4) Training loop
    print("Training model...")
    for epoch in range(1, num_epochs+1):
        cnp.train()
        running_loss = 0.0
        for states in states_arr:
            T = states.shape[0]
            # random sizes
            nc = np.random.randint(1, min(n_context_max, T)+1)
            nt = np.random.randint(1, min(n_target_max, T)+1)
            idx = np.arange(T)
            ctx_idx = np.random.choice(idx, size=nc, replace=False)
            tgt_idx = np.random.choice(np.setdiff1d(idx, ctx_idx), size=nt, replace=False)
            
            # build tensors
            t = np.linspace(0, 1, T)
            cx = np.stack([t[ctx_idx], states[ctx_idx, 4]], axis=1)
            cy = states[ctx_idx, :4]
            obs = np.concatenate([cx, cy], axis=1)[None]  # (1,nc,6)
            
            tx = np.stack([t[tgt_idx], states[tgt_idx, 4]], axis=1)[None]  # (1,nt,2)
            ty = states[tgt_idx, :4][None]  # (1,nt,4)
            
            obs_t = torch.from_numpy(obs).float().to(device)
            tx_t = torch.from_numpy(tx).float().to(device)
            ty_t = torch.from_numpy(ty).float().to(device)
            
            # forward + loss
            loss = cnp.nll_loss(obs_t, tx_t, ty_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(states_arr)
        print(f"Epoch {epoch:2d} – NLL Loss: {avg_loss:.4f}")
    
    # Save model
    torch.save(cnp.state_dict(), 'cnp_model.pt')
    
    # 5) After training, evaluate:
    print(f"Testing model with 100 random cases...")
    ee_errs, obj_errs = evaluate_cnp(cnp, states_arr,
                                     n_tests=100,
                                     n_context_max=n_context_max,
                                     n_target_max=n_target_max,
                                     device=device)
    
    # compute statistics
    ee_mean, ee_std = np.mean(ee_errs), np.std(ee_errs)
    obj_mean, obj_std = np.mean(obj_errs), np.std(obj_errs)
    
    print("Test Results:")
    print(f"End-effector MSE: {ee_mean:.6f} ± {ee_std:.6f}")
    print(f"Object MSE: {obj_mean:.6f} ± {obj_std:.6f}")
    
    # bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(['End-effector', 'Object'], [ee_mean, obj_mean], yerr=[ee_std, obj_std], capsize=10)
    plt.ylabel('Mean Squared Error')
    plt.title('Test Errors (Mean and Std)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('test_errors.png')
    plt.show() 