import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

import environment


class CNP(torch.nn.Module):
    def __init__(self, in_shape, hidden_size, num_hidden_layers, min_std=0.1):
        super(CNP, self).__init__()
        self.d_x = in_shape[0]
        self.d_y = in_shape[1]

        self.encoder = []
        self.encoder.append(torch.nn.Linear(self.d_x + self.d_y, hidden_size))
        self.encoder.append(torch.nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            self.encoder.append(torch.nn.Linear(hidden_size, hidden_size))
            self.encoder.append(torch.nn.ReLU())
        self.encoder.append(torch.nn.Linear(hidden_size, hidden_size))
        self.encoder = torch.nn.Sequential(*self.encoder)

        self.query = []
        self.query.append(torch.nn.Linear(hidden_size + self.d_x, hidden_size))
        self.query.append(torch.nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            self.query.append(torch.nn.Linear(hidden_size, hidden_size))
            self.query.append(torch.nn.ReLU())
        self.query.append(torch.nn.Linear(hidden_size, 2 * self.d_y))
        self.query = torch.nn.Sequential(*self.query)

        self.min_std = min_std

    def nll_loss(self, observation, target, target_truth, observation_mask=None, target_mask=None):
        '''
        The original negative log-likelihood loss for training CNP.
        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor that contains context
            points.
            d_x: the number of query dimensions
            d_y: the number of target dimensions.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor that contains query dimensions
            of target (query) points.
            d_x: the number of query dimensions.
            note: n_context and n_target does not need to be the same size.
        target_truth : torch.Tensor
            (n_batch, n_target, d_y) sized tensor that contains target
            dimensions (i.e., prediction dimensions) of target points.
            d_y: the number of target dimensions
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should be
            used in aggregation. Used for batch input.
        target_mask : torch.Tensor
            (n_batch, n_target) sized tensor indicating which entries should be
            used for loss calculation. Used for batch input.
        Returns
        -------
        loss : torch.Tensor (float)
            The NLL loss.
        '''
        mean, std = self.forward(observation, target, observation_mask)
        dist = torch.distributions.Normal(mean, std)
        nll = -dist.log_prob(target_truth)
        if target_mask is not None:
            # sum over the sequence (i.e. targets in the sequence)
            nll_masked = (nll * target_mask.unsqueeze(2)).sum(dim=1)
            # compute the number of entries for each batch entry
            nll_norm = target_mask.sum(dim=1).unsqueeze(1)
            # first normalize, then take an average over the batch and dimensions
            loss = (nll_masked / nll_norm).mean()
        else:
            loss = nll.mean()
        return loss

    def forward(self, observation, target, observation_mask=None):
        '''
        Forward pass of CNP.
        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor where d_x is the number
            of the query dimensions, d_y is the number of target dimensions.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor where d_x is the number of
            query dimensions. n_context and n_target does not need to be the
            same size.
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should be
            used in aggregation.
        Returns
        -------
        mean : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the mean
            prediction.
        std : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the standard
            deviation prediction.
        '''
        h = self.encode(observation)
        r = self.aggregate(h, observation_mask=observation_mask)
        h_cat = self.concatenate(r, target)
        query_out = self.decode(h_cat)
        mean = query_out[..., :self.d_y]
        logstd = query_out[..., self.d_y:]
        std = torch.nn.functional.softplus(logstd) + self.min_std
        return mean, std

    def encode(self, observation):
        h = self.encoder(observation)
        return h

    def decode(self, h):
        o = self.query(h)
        return o

    def aggregate(self, h, observation_mask):
        # this operation is equivalent to taking mean but for
        # batched input with arbitrary lengths at each entry
        # the output should have (batch_size, dim) shape

        if observation_mask is not None:
            h = (h * observation_mask.unsqueeze(2)).sum(dim=1)  # mask unrelated entries and sum
            normalizer = observation_mask.sum(dim=1).unsqueeze(1)  # compute the number of entries for each batch entry
            r = h / normalizer  # normalize
        else:
            # if observation mask is none, we assume that all entries
            # in the batch has the same length
            r = h.mean(dim=1)
        return r

    def concatenate(self, r, target):
        num_target_points = target.shape[1]
        r = r.unsqueeze(1).repeat(1, num_target_points, 1)  # repeating the same r_avg for each target
        h_cat = torch.cat([r, target], dim=-1)
        return h_cat


class CNMP(CNP):
    """
    Conditional Neural Process that takes a condition as input.
    Extends CNP by adding a condition that modifies the decoder.
    """
    def __init__(self, in_shape, hidden_size, num_hidden_layers, condition_dim, min_std=0.1):
        super(CNMP, self).__init__(in_shape, hidden_size, num_hidden_layers, min_std)
        
        # Create a new query (decoder) network to include condition
        query_layers = []
        query_layers.append(torch.nn.Linear(hidden_size + self.d_x + condition_dim, hidden_size))
        query_layers.append(torch.nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            query_layers.append(torch.nn.Linear(hidden_size, hidden_size))
            query_layers.append(torch.nn.ReLU())
        query_layers.append(torch.nn.Linear(hidden_size, 2 * self.d_y))
        self.query = torch.nn.Sequential(*query_layers)
        
        self.condition_dim = condition_dim
    
    def forward(self, observation, target, condition, observation_mask=None):
        """
        Forward pass of CNMP that takes a condition as input.
        
        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor
        condition : torch.Tensor
            (n_batch, condition_dim) sized tensor containing the condition
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should be used

        Returns
        -------
        mean : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the mean prediction.
        std : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the standard deviation prediction.
        """
        h = self.encode(observation)
        r = self.aggregate(h, observation_mask=observation_mask)
        h_cat = self.concatenate_with_condition(r, target, condition)
        query_out = self.decode(h_cat)
        mean = query_out[..., :self.d_y]
        logstd = query_out[..., self.d_y:]
        std = torch.nn.functional.softplus(logstd) + self.min_std
        return mean, std
    
    def concatenate_with_condition(self, r, target, condition):
        """
        Concatenate representation with target and condition.
        
        Parameters
        ----------
        r : torch.Tensor
            (n_batch, hidden_size) sized tensor
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor
        condition : torch.Tensor
            (n_batch, condition_dim) sized tensor
            
        Returns
        -------
        h_cat : torch.Tensor
            (n_batch, n_target, hidden_size + d_x + condition_dim) sized tensor
        """
        num_target_points = target.shape[1]
        r = r.unsqueeze(1).repeat(1, num_target_points, 1)
        condition = condition.unsqueeze(1).repeat(1, num_target_points, 1)
        h_cat = torch.cat([r, target, condition], dim=-1)
        return h_cat
    
    def nll_loss(self, observation, target, target_truth, condition, observation_mask=None, target_mask=None):
        """
        Negative log-likelihood loss for CNMP.
        
        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor
        target_truth : torch.Tensor
            (n_batch, n_target, d_y) sized tensor
        condition : torch.Tensor
            (n_batch, condition_dim) sized tensor
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor
        target_mask : torch.Tensor
            (n_batch, n_target) sized tensor
            
        Returns
        -------
        loss : torch.Tensor (float)
            The NLL loss.
        """
        mean, std = self.forward(observation, target, condition, observation_mask)
        dist = torch.distributions.Normal(mean, std)
        nll = -dist.log_prob(target_truth)
        if target_mask is not None:
            nll_masked = (nll * target_mask.unsqueeze(2)).sum(dim=1)
            nll_norm = target_mask.sum(dim=1).unsqueeze(1)
            loss = (nll_masked / nll_norm).mean()
        else:
            loss = nll.mean()
        return loss


class Hw5Env(environment.BaseEnv):
    def __init__(self, render_mode="gui") -> None:
        self._render_mode = render_mode
        self.viewer = None
        self._init_position = [0.0, -np.pi/2, np.pi/2, -2.07, 0, 0, 0]
        self._joint_names = [
            "ur5e/shoulder_pan_joint",
            "ur5e/shoulder_lift_joint",
            "ur5e/elbow_joint",
            "ur5e/wrist_1_joint",
            "ur5e/wrist_2_joint",
            "ur5e/wrist_3_joint",
            "ur5e/robotiq_2f85/right_driver_joint"
        ]
        self.reset()
        self._joint_qpos_idxs = [self.model.joint(x).qposadr for x in self._joint_names]
        self._ee_site = "ur5e/robotiq_2f85/gripper_site"

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        obj_pos = [0.5, 0.0, 1.5]
        height = np.random.uniform(0.03, 0.1)
        self.obj_height = height
        environment.create_object(scene, "box", pos=obj_pos, quat=[0, 0, 0, 1],
                                  size=[0.03, 0.03, height], rgba=[0.8, 0.2, 0.2, 1],
                                  name="obj1")
        return scene

    def state(self):
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="frontface")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=0).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return pixels / 255.0

    def high_level_state(self):
        ee_pos = self.data.site(self._ee_site).xpos[1:]
        obj_pos = self.data.body("obj1").xpos[1:]
        return np.concatenate([ee_pos, obj_pos, [self.obj_height]])


def bezier(p, steps=100):
    t = np.linspace(0, 1, steps).reshape(-1, 1)
    curve = np.power(1-t, 3)*p[0] + 3*np.power(1-t, 2)*t*p[1] + 3*(1-t)*np.power(t, 2)*p[2] + np.power(t, 3)*p[3]
    return curve


def prepare_data_for_cnmp(states_arr):
    """
    Prepare data for CNMP from collected trajectories.
    
    Parameters
    ----------
    states_arr : list of np.ndarray
        List of arrays where each row contains [ey, ez, oy, oz, h]
        
    Returns
    -------
    data : list of dictionaries
        List of dictionaries with keys 't', 'coords', 'height'
    """
    data = []
    
    for states in states_arr:
        # Verify the shape of the states array
        if states.shape[1] < 5:
            print(f"WARNING: State array has {states.shape[1]} columns, expected at least 5")
            continue
            
        # Add time dimension
        times = np.linspace(0, 1, len(states)).reshape(-1, 1)
        
        # Extract coordinates (ey, ez, oy, oz)
        coords = states[:, :4]
        
        # Extract height (constant for the trajectory)
        height = states[0, 4]
        
        data.append({
            't': times,
            'coords': coords,
            'height': height
        })
    
    return data


def split_train_test(data, test_ratio=0.2):
    """Split data into train and test sets."""
    np.random.shuffle(data)
    split_idx = int(len(data) * (1 - test_ratio))
    return data[:split_idx], data[split_idx:]


def sample_context_target(trajectory, n_context, n_target):
    """
    Sample context and target points from a trajectory.
    
    Parameters
    ----------
    trajectory : dict
        Dictionary with keys 't', 'coords', 'height'
    n_context : int
        Number of context points to sample
    n_target : int
        Number of target points to sample
        
    Returns
    -------
    context_x : np.ndarray
        (n_context, 1) array of context times
    context_y : np.ndarray
        (n_context, 4) array of context coordinates (ey, ez, oy, oz)
    target_x : np.ndarray
        (n_target, 1) array of target times
    target_y : np.ndarray
        (n_target, 4) array of target coordinates (ey, ez, oy, oz)
    height : float
        Height of the object
    """
    total_points = len(trajectory['t'])
    
    # Sample context indices
    context_idx = np.random.choice(total_points, size=n_context, replace=False)
    context_x = trajectory['t'][context_idx]
    context_y = trajectory['coords'][context_idx]
    
    # Sample target indices (can overlap with context)
    target_idx = np.random.choice(total_points, size=n_target, replace=False)
    target_x = trajectory['t'][target_idx]
    target_y = trajectory['coords'][target_idx]
    
    height = trajectory['height']
    
    return context_x, context_y, target_x, target_y, height


def train_cnmp(train_data, hidden_size=64, num_hidden_layers=3, num_epochs=100, batch_size=16):
    """
    Train a CNMP on the given data.
    
    Parameters
    ----------
    train_data : list of dict
        Training data
    hidden_size : int
        Hidden size of the neural networks
    num_hidden_layers : int
        Number of hidden layers
    num_epochs : int
        Number of epochs to train for
    batch_size : int
        Batch size
        
    Returns
    -------
    model : CNMP
        Trained CNMP model
    loss_history : list
        List of average loss values for each epoch
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define model
    # Query dimension (t) = 1, Target dimension (ey, ez, oy, oz) = 4, Condition dimension (h) = 1
    model = CNMP(in_shape=[1, 4], hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, 
                 condition_dim=1, min_std=0.01)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # For tracking loss history
    loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx in range(0, len(train_data), batch_size):
            batch_data = train_data[batch_idx:batch_idx + batch_size]
            batch_size_actual = len(batch_data)
            
            # Random number of context and target points for each trajectory
            n_contexts = np.random.randint(1, 20, size=batch_size_actual)
            n_targets = np.random.randint(1, 20, size=batch_size_actual)
            
            # Prepare batch
            context_x_batch = []
            context_y_batch = []
            target_x_batch = []
            target_y_batch = []
            heights_batch = []
            
            for i, traj in enumerate(batch_data):
                cx, cy, tx, ty, h = sample_context_target(traj, n_contexts[i], n_targets[i])
                context_x_batch.append(cx)
                context_y_batch.append(cy)
                target_x_batch.append(tx)
                target_y_batch.append(ty)
                heights_batch.append(h)
            
            # Find max lengths for padding
            max_context_len = max(len(cx) for cx in context_x_batch)
            max_target_len = max(len(tx) for tx in target_x_batch)
            
            # Create observation mask
            observation_mask = torch.zeros(batch_size_actual, max_context_len, device=device)
            
            # Pad and convert to tensors
            padded_context_x = torch.zeros(batch_size_actual, max_context_len, 1, device=device)
            padded_context_y = torch.zeros(batch_size_actual, max_context_len, 4, device=device)
            padded_target_x = torch.zeros(batch_size_actual, max_target_len, 1, device=device)
            padded_target_y = torch.zeros(batch_size_actual, max_target_len, 4, device=device)
            heights = torch.zeros(batch_size_actual, 1, device=device)
            
            for i in range(batch_size_actual):
                cx_len = len(context_x_batch[i])
                tx_len = len(target_x_batch[i])
                
                padded_context_x[i, :cx_len, 0] = torch.tensor(context_x_batch[i].flatten(), device=device)
                padded_context_y[i, :cx_len] = torch.tensor(context_y_batch[i], device=device)
                padded_target_x[i, :tx_len, 0] = torch.tensor(target_x_batch[i].flatten(), device=device)
                padded_target_y[i, :tx_len] = torch.tensor(target_y_batch[i], device=device)
                observation_mask[i, :cx_len] = 1.0
                heights[i, 0] = torch.tensor(heights_batch[i], device=device)
            
            # Combine context inputs
            context = torch.cat([padded_context_x, padded_context_y], dim=2)
            
            # Forward pass and loss calculation
            optimizer.zero_grad()
            loss = model.nll_loss(context, padded_target_x, padded_target_y, heights, observation_mask)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_size_actual
        
        avg_epoch_loss = epoch_loss / len(train_data)
        loss_history.append(avg_epoch_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}")
    
    return model, loss_history


def evaluate_cnmp(model, test_data, num_tests=100):
    """
    Evaluate CNMP on test data.
    
    Parameters
    ----------
    model : CNMP
        Trained CNMP model
    test_data : list of dict
        Test data
    num_tests : int
        Number of test cases
        
    Returns
    -------
    ee_errors : list
        List of MSE for end-effector predictions
    obj_errors : list
        List of MSE for object predictions
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    ee_errors = []
    obj_errors = []
    
    with torch.no_grad():
        for test_idx in range(num_tests):
            try:
                # Sample a random trajectory
                trajectory_idx = np.random.randint(0, len(test_data))
                trajectory = test_data[trajectory_idx]
                
                # Random number of context and target points
                n_context = np.random.randint(1, 20)
                n_target = np.random.randint(1, 20)
                
                # Sample context and target points
                context_x, context_y, target_x, target_y, height = sample_context_target(
                    trajectory, n_context, n_target)
                
                # Convert to tensors
                context_x_tensor = torch.tensor(context_x, dtype=torch.float32, device=device)
                context_y_tensor = torch.tensor(context_y, dtype=torch.float32, device=device)
                target_x_tensor = torch.tensor(target_x, dtype=torch.float32, device=device)
                target_y_tensor = torch.tensor(target_y, dtype=torch.float32, device=device)
                height_tensor = torch.tensor([[height]], dtype=torch.float32, device=device)
                
                # Combine context
                context = torch.cat([context_x_tensor, context_y_tensor], dim=1).unsqueeze(0)
                target_x_tensor = target_x_tensor.unsqueeze(0)
                target_y_tensor = target_y_tensor.unsqueeze(0)
                
                # Forward pass
                mean, _ = model(context, target_x_tensor, height_tensor)
                
                # Calculate squared errors (batch_size, n_targets, n_dimensions)
                squared_errors = (mean - target_y_tensor) ** 2
                
                # Average over targets to get per-dimension errors (batch_size, n_dimensions)
                avg_squared_errors = squared_errors.mean(dim=1)
                
                # Extract to numpy
                errors = avg_squared_errors.cpu().numpy()[0]  # (n_dimensions,)
                
                # First two dimensions are end-effector (ey, ez)
                ee_mse = np.mean(errors[0:2])
                
                # Last two dimensions are object (oy, oz)
                obj_mse = np.mean(errors[2:4])
                
                # Add to lists if valid
                if np.isfinite(ee_mse):
                    ee_errors.append(ee_mse)
                if np.isfinite(obj_mse):
                    obj_errors.append(obj_mse)
                    
            except Exception as e:
                print(f"Error in test {test_idx}: {e}")
                continue
                
            # Print progress
            if (test_idx + 1) % 10 == 0:
                print(f"Completed {test_idx + 1}/{num_tests} test cases", end="\r")
        
        print("\nEvaluation complete.")
        print(f"Valid error samples - End-effector: {len(ee_errors)}, Object: {len(obj_errors)}")
    
    # Return empty lists if no valid errors were found
    if not ee_errors:
        print("WARNING: No valid end-effector errors found!")
        ee_errors = [0.0]
    if not obj_errors:
        print("WARNING: No valid object errors found!")
        obj_errors = [0.0]
        
    return ee_errors, obj_errors


if __name__ == "__main__":
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
        print(f"Collected {i+1} trajectories.", end="\r")

    fig, ax = plt.subplots(1, 2)
    for states in states_arr:
        ax[0].plot(states[:, 0], states[:, 1], alpha=0.2, color="b")
        ax[0].set_xlabel("e_y")
        ax[0].set_ylabel("e_z")
        ax[1].plot(states[:, 2], states[:, 3], alpha=0.2, color="r")
        ax[1].set_xlabel("o_y")
        ax[1].set_ylabel("o_z")
    plt.savefig('trajectories.png')
    plt.close()

    # Prepare data for CNMP
    data = prepare_data_for_cnmp(states_arr)
    train_data, test_data = split_train_test(data)
    
    print(f"Training CNMP on {len(train_data)} trajectories, testing on {len(test_data)} trajectories")
    
    # Train CNMP
    model, loss_history = train_cnmp(train_data, hidden_size=64, num_hidden_layers=3, num_epochs=100, batch_size=16)
    
    # Evaluate CNMP
    print("\nEvaluating CNMP on 100 random test cases with varying context and target points...")
    ee_errors, obj_errors = evaluate_cnmp(model, test_data, num_tests=100)
    
    # Calculate mean and std of errors
    ee_mean, ee_std = np.mean(ee_errors), np.std(ee_errors)
    obj_mean, obj_std = np.mean(obj_errors), np.std(obj_errors)
    
    # Check for potential NaN/inf values
    if np.isnan(ee_mean) or np.isinf(ee_mean):
        print("WARNING: End-effector mean error is NaN/Inf, setting to 0")
        ee_mean = 0.0
    if np.isnan(ee_std) or np.isinf(ee_std):
        print("WARNING: End-effector std error is NaN/Inf, setting to 0")
        ee_std = 0.0
    if np.isnan(obj_mean) or np.isinf(obj_mean):
        print("WARNING: Object mean error is NaN/Inf, setting to 0")
        obj_mean = 0.0
    if np.isnan(obj_std) or np.isinf(obj_std):
        print("WARNING: Object std error is NaN/Inf, setting to 0")
        obj_std = 0.0
    
    print(f"End-effector MSE (ey, ez): {ee_mean:.6f} ± {ee_std:.6f}")
    print(f"Object MSE (oy, oz): {obj_mean:.6f} ± {obj_std:.6f}")
    
    # Plot errors as separate bars for end-effector and object
    plt.figure(figsize=(10, 6))
    
    # Data for plotting
    labels = ['End-effector', 'Object']
    means = [ee_mean, obj_mean]
    stds = [ee_std, obj_std]
    
    # Position of bars on x-axis
    x = np.arange(len(labels))
    width = 0.5
    
    # Create the bars with error bars
    bars = plt.bar(x, means, width, yerr=stds, capsize=10, 
                  color=['blue', 'red'], alpha=0.7)
    
    # Add labels, title and custom x-axis tick labels
    plt.ylabel('Mean Squared Error', fontsize=14)
    plt.title('CNMP Prediction Error by Component Type', fontsize=16)
    plt.xticks(x, labels, fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of the bars (safely)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if np.isfinite(height) and np.isfinite(x[i]) and height > 0:
            plt.text(x[i], height + 0.005, f'{height:.6f}', 
                    ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('cnmp_errors.png', dpi=300)
    plt.show()

    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(loss_history) + 1), loss_history, linewidth=2, color='blue')
    
    # Add markers at every 10 epochs
    markers_idx = list(range(9, len(loss_history), 10))  # 0-indexed, so epoch 10 is at index 9
    markers_x = [i + 1 for i in markers_idx]  # Convert to 1-indexed for plotting
    markers_y = [loss_history[i] for i in markers_idx]
    plt.plot(markers_x, markers_y, 'ro', markersize=8)
    
    # Add text labels for marker points
    for i, (x, y) in enumerate(zip(markers_x, markers_y)):
        epoch_num = (i + 1) * 10
        plt.annotate(f'Epoch {epoch_num}: {y:.4f}', 
                    xy=(x, y), 
                    xytext=(10, 0), 
                    textcoords='offset points',
                    fontsize=9,
                    va='center')
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Negative Log-Likelihood Loss', fontsize=14)
    plt.title('CNMP Training Loss History', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis ticks to be every 10 epochs
    plt.xticks(list(range(0, len(loss_history) + 1, 10)))
    
    # Adjust y-axis to focus on the interesting part of the loss curve
    plt.ylim(min(loss_history) - 0.1, max(min(5, max(loss_history)), 0))
    
    plt.tight_layout()
    plt.savefig('cnmp_loss_history.png', dpi=300)
    plt.show()
