import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.gridspec import GridSpec
import os
from scipy.linalg import eigh
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE
import matplotlib.animation as animation
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors

def load_data():
    """Load the saved noise data and timesteps"""
    with open('all_timestep_noises.pkl', 'rb') as f:
        all_timestep_noises = pickle.load(f)
    
    with open('t_steps.pkl', 'rb') as f:
        t_steps = pickle.load(f)
    
    return all_timestep_noises, t_steps

def create_visualizations(all_timestep_noises, t_steps, output_dir='diffusion_maps'):
    """Create visualizations of noise distributions across timesteps"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dimensions from the data
    timesteps = sorted(all_timestep_noises.keys())
    
    # Extract actual timestep values (skip the last step which is just for final denoising)
    t_values = t_steps[:-1].cpu().numpy()
    
    # Create noise distributions plot
    plot_noise_distributions(all_timestep_noises, timesteps, t_values, output_dir)
    
    # Create diffusion map visualization
    create_diffusion_map_visualization(all_timestep_noises, timesteps, t_values, output_dir)
    
    # Create t-SNE visualization
    create_tsne_visualization(all_timestep_noises, timesteps, t_values, output_dir)
    
    # Create t-SNE animation
    create_tsne_animation(all_timestep_noises, timesteps, t_values, output_dir, save_frames=False)

def plot_noise_distributions(all_timestep_noises, timesteps, t_values, output_dir):
    """Plot distribution of normalized noise vector magnitudes across timesteps"""
    print("Generating noise distributions plot...")
    
    # Create violin plots for distribution at select timesteps
    sample_indices = np.linspace(0, len(timesteps)-1, 18, dtype=int)
    plt.figure(figsize=(15, 8))
    
    data = []
    labels = []
    
    for i, idx in enumerate(sample_indices):
        t_idx = timesteps[idx]
        noise = all_timestep_noises[t_idx]  # [K, batch_size, C, H, W]
        
        # Reshape to [K * batch_size, C*H*W]
        K = noise.shape[0]
        batch_size = noise.shape[1]
        noise_reshaped = noise.view(K * batch_size, -1)
        
        # Get number of dimensions
        n_dims = noise_reshaped.shape[1]
        
        # Compute Mahalanobis distance from zero vector with identity covariance
        # With identity covariance, this is equivalent to the L2 norm (Euclidean distance)
        # Normalize by sqrt(dimensions) to account for dimensionality
        mahalanobis_distances = torch.norm(noise_reshaped, dim=1).cpu().numpy() / np.sqrt(3*64*64.)
        
        data.append(mahalanobis_distances)
        labels.append(f'σ={t_values[idx]:.3f}')
    
    plt.violinplot(data, showmeans=True)
    plt.xticks(range(1, len(sample_indices)+1), labels)
    plt.title('Distribution of Mahalanobis Distances of Noise Vectors by $\sigma$')
    plt.ylabel('Mahalanobis Distance from N(0,I)')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Add a horizontal line at y=1 to show expected value for standard normal
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Expected value for N(0,I)')
    plt.legend()
    
    plt.savefig(f'{output_dir}/noise_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_diffusion_map_visualization(all_timestep_noises, timesteps, t_values, output_dir):
    """Create and save a diffusion map visualization based on noise data"""
    print("Generating diffusion map visualization...")
    
    # Sample a subset of timesteps for computational efficiency
    sample_timesteps = np.linspace(0, len(timesteps)-1, min(len(timesteps), 18), dtype=int)
    sample_timesteps = [timesteps[i] for i in sample_timesteps]
    
    # Collect data from sampled timesteps
    data_points = []
    timestep_indices = []
    
    for i, t_idx in enumerate(sample_timesteps):
        # Get first few noise samples from each timestep
        noise = all_timestep_noises[t_idx][:, 0]  # Use first 5 noise samples from first batch
        flat_noise = noise.reshape(noise.shape[0], -1).cpu().numpy()
        
        # Further reduce dimensionality for computational efficiency if needed
        # if flat_noise.shape[1] > 1000:
        #     # Sample random dimensions
        #     cols = np.random.choice(flat_noise.shape[1], 1000, replace=False)
        #     flat_noise = flat_noise[:, cols]
        
        data_points.append(flat_noise)
        timestep_indices.extend([i] * flat_noise.shape[0])
    
    # Stack all data points
    data = np.vstack(data_points)
    
    # 1. Compute pairwise distances
    dist_matrix = euclidean_distances(data)
    
    # 2. Apply Gaussian kernel to get similarity matrix
    epsilon = np.median(dist_matrix) ** 2  # Bandwidth parameter
    kernel_matrix = np.exp(-dist_matrix**2 / epsilon)
    
    # 3. Create normalized graph Laplacian (Diffusion operator)
    # Compute degree matrix (sum of each row)
    D = np.sum(kernel_matrix, axis=1)
    D_inv_sqrt = 1.0 / np.sqrt(D)
    
    # Symmetric normalized Laplacian
    P = D_inv_sqrt[:, np.newaxis] * kernel_matrix * D_inv_sqrt[np.newaxis, :]
    
    # 4. Compute eigenvectors of P
    eigenvals, eigenvecs = eigh(P)
    
    # Sort by eigenvalues in descending order
    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # 5. Create diffusion map embedding (using first few non-trivial eigenvectors)
    # Skip the first eigenvector (constant)
    embedding = eigenvecs[:, 1:3]  # Using 2nd and 3rd eigenvectors for 2D visualization
    
    print(embedding[:, 0].shape)

    # Plot the embedding
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
               c=timestep_indices, cmap='viridis', 
               alpha=0.8, s=50)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Timestep Index')
    
    # Add timestep labels for each cluster
    # for i, t_idx in enumerate(sample_timesteps):
    #     # Find center of each timestep cluster
    #     mask = np.array(timestep_indices) == i
    #     if np.any(mask):
    #         center_x = np.mean(embedding[mask, 0])
    #         center_y = np.mean(embedding[mask, 1])
    #         plt.annotate(f'σ={t_values[sample_timesteps.index(t_idx)]:.3f}', 
    #                      (center_x, center_y), 
    #                      fontsize=12, ha='center', va='center',
    #                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.title('Diffusion Map of Noise Data')
    plt.xlabel('Diffusion Coordinate 1')
    plt.ylabel('Diffusion Coordinate 2')
    plt.grid(alpha=0.3)
    
    plt.savefig(f'{output_dir}/diffusion_map.png')
    plt.close()

def create_tsne_visualization(all_timestep_noises, timesteps, t_values, output_dir='diffusion_maps'):
    """Create and save a t-SNE visualization based on noise data"""
    print("Generating t-SNE visualization...")
    
    # Sample a subset of timesteps for computational efficiency
    sample_timesteps = np.linspace(0, len(timesteps)-1, min(len(timesteps), 18), dtype=int)
    sample_timesteps = [timesteps[i] for i in sample_timesteps]
    
    # Collect data from sampled timesteps
    data_points = []
    timestep_indices = []
    
    for i, t_idx in enumerate(sample_timesteps):
        # Get noise samples from each timestep
        noise = all_timestep_noises[t_idx][:, 0]  # Use first batch
        flat_noise = noise.reshape(noise.shape[0], -1).cpu().numpy()
        
        data_points.append(flat_noise)
        timestep_indices.extend([i] * flat_noise.shape[0])
    
    # Stack all data points
    data = np.vstack(data_points)
    
    # Apply t-SNE for dimensionality reduction
    print(f"Running t-SNE on {data.shape[0]} samples of dimension {data.shape[1]}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data)-1))
    embedding = tsne.fit_transform(data)
    
    # Plot the embedding
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
               c=timestep_indices, cmap='viridis', 
               alpha=0.8, s=50)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Timestep Index')
    
    # Calculate centroids for each timestep group
    centroids = []
    for i in range(len(sample_timesteps)):
        mask = np.array(timestep_indices) == i
        if np.any(mask):
            centroid_x = np.mean(embedding[mask, 0])
            centroid_y = np.mean(embedding[mask, 1])
            centroids.append((centroid_x, centroid_y))
            
            # # Add timestep label at each centroid
            # plt.annotate(f'σ={t_values[sample_timesteps.index(sample_timesteps[i])]:.3f}', 
            #             (centroid_x, centroid_y), 
            #             fontsize=10, ha='center', va='center',
            #             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Draw arrows between consecutive centroids
    for i in range(len(centroids) - 1):
        plt.annotate("", 
                    xy=centroids[i+1],        # End point
                    xytext=centroids[i],      # Start point
                    arrowprops=dict(arrowstyle="->", color="black", linewidth=1.5),
                    )
    
    # Add title and labels
    plt.title('t-SNE Visualization of Noise Data with Timestep Progression')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(alpha=0.3)
    
    plt.savefig(f'{output_dir}/tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_tsne_animation(all_timestep_noises, timesteps, t_values, output_dir='diffusion_maps', save_frames=False):
    """Create an animation of noise candidates in t-SNE space across timesteps and iterations
    
    Args:
        all_timestep_noises: Dictionary mapping timestep indices to noise tensors
        timesteps: List of timestep indices
        t_values: Tensor of timestep values (sigma values)
        output_dir: Directory to save output files
        save_frames: Whether to save individual frames alongside the animation
    """
    print("Generating t-SNE animation...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample a subset of timesteps for computational efficiency
    sample_timesteps = np.linspace(0, len(timesteps)-1, min(len(timesteps), 10), dtype=int)
    sample_timesteps = [timesteps[i] for i in sample_timesteps]
    
    # Define the number of iterations to use per timestep
    max_iterations = 20
    
    # Prepare data for t-SNE
    all_data = []
    frame_indices = []  # To keep track of which frame each data point belongs to
    timestep_labels = []  # To store timestep value for each frame
    iteration_indices = []  # To track iteration number within each timestep
    
    for ts_idx, t_idx in enumerate(sample_timesteps):
        # Get noise samples from each timestep
        noise = all_timestep_noises[t_idx]  # Shape: [iterations, batch_size, C, H, W]
        
        # Get number of iterations for this timestep
        num_iterations = noise.shape[0]
        iterations_to_use = min(max_iterations, num_iterations)  # Use up to max_iterations
        
        # Sample iterations if needed
        if num_iterations > max_iterations:
            iter_indices = np.linspace(0, num_iterations-1, iterations_to_use, dtype=int)
        else:
            iter_indices = range(iterations_to_use)
        
        for iter_idx in iter_indices:
            # Use first sample from batch for this iteration
            iter_noise = noise[iter_idx, 0].reshape(1, -1).cpu().numpy()
            all_data.append(iter_noise)
            
            # Calculate frame index (each timestep gets multiple frames, one per iteration)
            frame_idx = ts_idx * max_iterations + list(iter_indices).index(iter_idx)
            frame_indices.append(frame_idx)
            
            # Store normalized iteration index (0-1 range for color coding)
            normalized_iter = list(iter_indices).index(iter_idx) / (iterations_to_use - 1)
            iteration_indices.append(normalized_iter)
            
            # Store timestep value for this frame
            try:
                # Get the appropriate sigma value
                sigma_val = t_values[t_idx] if t_idx < len(t_values) else t_values[-1]
                timestep_labels.append(f"σ={sigma_val:.4f}, Iteration {iter_idx+1}/{iterations_to_use}")
            except IndexError:
                # Handle case where t_idx might be out of bounds
                timestep_labels.append(f"Step {t_idx}, Iteration {iter_idx+1}/{iterations_to_use}")
    
    # Stack all data points
    all_data = np.vstack(all_data)
    
    # Apply t-SNE for dimensionality reduction
    print(f"Running t-SNE on {all_data.shape[0]} samples of dimension {all_data.shape[1]}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_data)-1))
    embedding = tsne.fit_transform(all_data)
    
    # Get unique frame indices and count
    unique_frames = sorted(set(frame_indices))
    num_frames = len(unique_frames)
    
    # Set up figure and colormap for iterations (plasma offers better color distinctions)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use viridis colormap for iterations (consistent with typical iteration progress coloring)
    cmap_iters = plt.cm.get_cmap('viridis')
    
    # More distinctive colors for timesteps
    timestep_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    # Determine global min/max for consistent axes
    x_min, x_max = np.min(embedding[:, 0]), np.max(embedding[:, 0])
    y_min, y_max = np.min(embedding[:, 1]), np.max(embedding[:, 1])
    
    # Add padding to axes limits
    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax.set_xlim([x_min - padding * x_range, x_max + padding * x_range])
    ax.set_ylim([y_min - padding * y_range, y_max + padding * y_range])
    
    # Set up scatter plot with empty data initially
    scatter = ax.scatter([], [], s=80, c=[], cmap='viridis', alpha=0.8)
    
    # Add colorbar showing iteration progression (0-1 range for consistency)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Iteration Progress (0=start, 1=end)')
    
    # Title will be updated in animation
    title = ax.set_title('', fontsize=14)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.grid(alpha=0.3)
    
    # Create text annotation for timestep label
    text_label = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=14,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Track trajectories: maps timestep -> list of points for that timestep
    trajectories = {}
    for i, fi in enumerate(frame_indices):
        timestep = fi // max_iterations
        if timestep not in trajectories:
            trajectories[timestep] = []
        trajectories[timestep].append((embedding[i], iteration_indices[i]))
    
    # Lines for each timestep trajectory (but we won't connect between different timesteps)
    trajectory_lines = []
    for ts in range(len(sample_timesteps)):
        if ts in trajectories:
            # Sort points by iteration within this timestep
            points = sorted(trajectories[ts], key=lambda x: x[1])
            trajectories[ts] = [p[0] for p in points]  # Store just the points, sorted by iteration
            
            # Create line with improved style and arrow markers
            line, = ax.plot([], [], '-', alpha=0.7, color=timestep_colors[ts % len(timestep_colors)], 
                          linewidth=2.5, solid_capstyle='round', 
                          marker='>', markersize=8, markevery=3)  # Add arrow markers
            trajectory_lines.append(line)
        else:
            trajectory_lines.append(None)
    
    # Animation update function
    def update(frame):
        # For smooth transitions between frames, we can include points from adjacent frames with reduced opacity
        transition_width = 8  # Higher value for smoother transitions
        
        # Clear previous points
        if len(ax.collections) > 0:
            ax.collections[0].remove()
        
        points = []
        colors = []
        sizes = []
        alpha_values = []
        
        for i, fi in enumerate(frame_indices):
            # Calculate distance from current frame
            frame_distance = abs(fi - frame)
            
            # Include points if they're within the transition window
            if frame_distance <= transition_width:
                points.append(embedding[i])
                
                # Use iteration number for color (normalized 0-1)
                colors.append(iteration_indices[i])
                
                # Size decreases as distance from current frame increases
                opacity = max(0, 1 - (frame_distance / transition_width))
                sizes.append(120 * opacity)  # Larger points
                alpha_values.append(0.9 * opacity)
        
        if points:
            points = np.array(points)
            scatter = ax.scatter(points[:, 0], points[:, 1], s=sizes, c=colors, 
                               cmap='viridis', alpha=alpha_values, vmin=0, vmax=1,
                               edgecolors='black', linewidths=0.5)  # Add outlines to points
        
        # Update trajectory lines - only show trajectory for current timestep
        current_timestep = int(frame // max_iterations)
        for ts, line in enumerate(trajectory_lines):
            if line is not None:
                if ts == current_timestep:
                    # For current timestep, show trajectory up to current iteration
                    current_iter = int(frame % max_iterations)
                    points = trajectories[ts][:current_iter+1]
                    if len(points) > 1:  # Need at least 2 points for a line
                        xs, ys = zip(*points)
                        line.set_data(xs, ys)
                        
                        # Adjust marker frequency based on number of points
                        if len(points) > 5:
                            # For longer lines, show markers every few points
                            marker_every = max(1, len(points) // 5)
                            line.set_markevery(marker_every)
                        else:
                            # For shorter lines, just show the last point
                            line.set_markevery([-1])
                else:
                    # For other timesteps, show no trajectory
                    line.set_data([], [])
        
        # Update timestep label - find the closest matching frame
        closest_idx = min(range(len(frame_indices)), key=lambda i: abs(frame_indices[i] - frame))
        text_label.set_text(timestep_labels[closest_idx])
        
        title.set_text('t-SNE Visualization of Noise Candidates Across Denoising Steps')
        
        return [scatter, text_label, title] + trajectory_lines
    
    # Calculate the total number of frames for the animation - increase for slower animation
    total_frames = num_frames * 3  # Triple the frames for even slower playback
    
    # Create animation with more frames for smoother, slower effect
    print("Creating animation...")
    ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, num_frames-1, total_frames, dtype=int), 
                                 interval=250, blit=True)  # Increased interval for slower playback
    
    # Save animation with higher quality
    video_path = f'{output_dir}/tsne_animation.mp4'
    print(f"Saving animation to {video_path}...")
    
    # If requested, save individual frames
    if save_frames:
        frames_dir = f"{output_dir}/animation_frames"
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Saving individual frames to {frames_dir}...")
        
        # Generate frames at appropriate intervals
        frame_count = min(60, total_frames)  # Cap at 60 frames to avoid too many files
        for i, frame_idx in enumerate(np.linspace(0, num_frames-1, frame_count, dtype=int)):
            # Update the figure for this frame
            update(frame_idx)
            plt.savefig(f"{frames_dir}/frame_{i:03d}.png", dpi=300, bbox_inches='tight')
            print(f"Saved frame {i+1}/{frame_count}")
    
    # Check available writers and choose the best available one
    try:
        available_writers = animation.writers.list()
        print(f"Available animation writers: {available_writers}")
        
        if animation.writers.is_available('ffmpeg'):
            writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='TSNE Animation'), bitrate=5000)
            ani.save(video_path, writer=writer, dpi=300)
        elif animation.writers.is_available('imagemagick'):
            writer = animation.ImageMagickWriter(fps=10, metadata=dict(artist='TSNE Animation'))
            ani.save(video_path, writer=writer, dpi=300)
        elif 'pillow' in available_writers:
            ani.save(f'{output_dir}/tsne_animation.gif', writer='pillow', fps=6, dpi=150)
            print(f"Saved as GIF instead of MP4 at {output_dir}/tsne_animation.gif")
        else:
            print("No suitable animation writers found. Saving individual frames instead.")
            # Save individual frames
            frames_dir = f"{output_dir}/animation_frames"
            os.makedirs(frames_dir, exist_ok=True)
            
            # Sample frames to save
            for i, frame_idx in enumerate(np.linspace(0, num_frames-1, 60, dtype=int)):  # More frames
                # Update the figure for this frame
                update(frame_idx)
                plt.savefig(f"{frames_dir}/frame_{i:03d}.png", dpi=300, bbox_inches='tight')
                print(f"Saved frame {i+1}/60")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Saving a static visualization of key frames instead.")
        
        # Create a grid of snapshots at different timesteps
        plt.figure(figsize=(20, 16))
        
        # Sample 16 frames
        sample_frames = np.linspace(0, num_frames-1, 16, dtype=int)
        
        for i, frame_idx in enumerate(sample_frames):
            plt.subplot(4, 4, i+1)
            
            # Get points for this frame
            points = []
            colors = []
            for j, fi in enumerate(frame_indices):
                if fi == frame_idx:
                    points.append(embedding[j])
                    colors.append(iteration_indices[j])  # Color by iteration
            
            if points:
                points = np.array(points)
                plt.scatter(points[:, 0], points[:, 1], c=colors, s=80, alpha=0.8, cmap='viridis', vmin=0, vmax=1)
            
            # Find the label for this frame
            closest_idx = min(range(len(frame_indices)), key=lambda i: abs(frame_indices[i] - frame_idx))
            plt.title(timestep_labels[closest_idx], fontsize=10)
            plt.grid(alpha=0.3)
            
            if i % 4 == 0:
                plt.ylabel('t-SNE Dimension 2')
            if i >= 12:
                plt.xlabel('t-SNE Dimension 1')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/tsne_keyframes.png', dpi=300, bbox_inches='tight')
    
    plt.close()
    print("Animation or alternative visualizations created successfully!")

def main():
    print("Loading noise data...")
    all_timestep_noises, t_steps = load_data()
    
    print(f"Found noise data for {len(all_timestep_noises)} timesteps")
    print(f"Creating visualizations...")
    
    create_visualizations(all_timestep_noises, t_steps)
    
    print("Visualizations complete!")

if __name__ == "__main__":
    main()
