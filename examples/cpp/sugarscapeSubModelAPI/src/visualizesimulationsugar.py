import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os

def visualize():
    # Load data
    print("Loading logs...")
    try:
        if not os.path.exists('bugs_log.csv'):
            print("bugs_log.csv not found.")
            return
        bugs_data = np.genfromtxt('bugs_log.csv', delimiter=',', skip_header=1)
        
        if not os.path.exists('cells_log.csv'):
            print("cells_log.csv not found.")
            return
        cells_data = np.genfromtxt('cells_log.csv', delimiter=',', skip_header=1)
    except Exception as e:
        print(f"Error loading logs: {e}")
        return

    # Extract columns
    # bugs: step,x,y,sugar,metabolism
    bug_steps = bugs_data[:, 0]
    bug_x = bugs_data[:, 1]
    bug_y = bugs_data[:, 2]
    bug_sugar = bugs_data[:, 3]

    # cells: step,x,y,sugar,max_sugar
    cell_steps = cells_data[:, 0]
    cell_x = cells_data[:, 1]
    cell_y = cells_data[:, 2]
    cell_sugar = cells_data[:, 3]

    unique_bug_steps = np.sort(np.unique(bug_steps)).astype(int)
    unique_cell_steps = np.sort(np.unique(cell_steps)).astype(int)
    grid_dim = 256

    # 1. Statistics Plot
    print("Generating statistics plot...")
    bug_count = []
    avg_bug_sugar = []
    avg_cell_sugar = []
    
    for s in unique_bug_steps:
        mask = (bug_steps == s)
        bug_count.append(np.sum(mask))
        avg_bug_sugar.append(np.mean(bug_sugar[mask]))

    for s in unique_cell_steps:
        mask = (cell_steps == s)
        avg_cell_sugar.append(np.mean(cell_sugar[mask]))

    fig_stats, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Bug Count / Avg Sugar', color='tab:red')
    ax1.plot(unique_bug_steps, bug_count, color='tab:red', label='Bug Count')
    ax1.plot(unique_bug_steps, avg_bug_sugar, color='tab:orange', label='Avg Bug Sugar')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Avg Cell Sugar', color='tab:blue')
    # Resample unique_cell_steps to match unique_bug_steps for plotting if needed, 
    # but here we just plot them as they are.
    ax2.plot(unique_cell_steps, avg_cell_sugar, color='tab:blue', marker='o', label='Avg Cell Sugar')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.legend(loc='upper right')

    plt.title('Sugarscape Simulation Statistics')
    fig_stats.tight_layout()
    plt.savefig('sugarscape_stats.png')
    print("Saved sugarscape_stats.png")

    # 2. Animation
    print("Creating animation (this might take a moment)...")
    fig_anim, ax_anim = plt.subplots(figsize=(8, 8))
    
    # Helper to get grid at a specific step
    def get_grid(step):
        # find closest available cell step (rounding down)
        available_steps = unique_cell_steps[unique_cell_steps <= step]
        if len(available_steps) == 0:
            s = unique_cell_steps[0]
        else:
            s = available_steps[-1]
            
        mask = (cell_steps == s)
        grid = np.zeros((grid_dim, grid_dim))
        grid[cell_x[mask].astype(int), cell_y[mask].astype(int)] = cell_sugar[mask]
        return grid

    grid_img = ax_anim.imshow(get_grid(unique_bug_steps[0]).T, origin='lower', cmap='YlOrBr', 
                             extent=[0, grid_dim, 0, grid_dim], vmin=0, vmax=7)
    
    # Plot bugs
    mask0 = (bug_steps == unique_bug_steps[0])
    scat = ax_anim.scatter(bug_x[mask0], bug_y[mask0], c='red', s=2, alpha=0.6, label='Bugs')
    
    plt.colorbar(grid_img, ax=ax_anim, label='Sugar Level')
    ax_anim.set_title(f'Sugarscape - Step {unique_bug_steps[0]}')
    ax_anim.set_xlim(0, grid_dim)
    ax_anim.set_ylim(0, grid_dim)

    def update(frame):
        s = unique_bug_steps[frame]
        # Update grid if this step has cell data
        if s in unique_cell_steps:
            grid_img.set_data(get_grid(s).T)
        
        mask = (bug_steps == s)
        scat.set_offsets(np.c_[bug_x[mask], bug_y[mask]])
        ax_anim.set_title(f'Sugarscape - Step {s} (Bugs: {int(np.sum(mask))})')
        return grid_img, scat

    # Reduce number of frames for GIF if too many
    frames = unique_bug_steps
    if len(frames) > 100:
        frames = frames[::len(frames)//100]

    ani = animation.FuncAnimation(fig_anim, update, frames=len(frames), interval=100, blit=True)
    
    try:
        ani.save('sugarscape_simulation.gif', writer='pillow', fps=10)
        print("Saved sugarscape_simulation.gif")
    except Exception as e:
        print(f"Could not save animation: {e}")

    # 3. Static snapshots (Start, Mid, End)
    print("Generating snapshots...")
    steps_to_plot = [unique_bug_steps[0], unique_bug_steps[len(unique_bug_steps)//2], unique_bug_steps[-1]]
    fig_snap, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, s in enumerate(steps_to_plot):
        ax = axes[i]
        grid = get_grid(s)
        ax.imshow(grid.T, origin='lower', cmap='YlOrBr', extent=[0, grid_dim, 0, grid_dim], vmin=0, vmax=7)
        mask = (bug_steps == s)
        ax.scatter(bug_x[mask], bug_y[mask], c='red', s=1, alpha=0.5)
        ax.set_title(f'Step {s} (Bugs: {int(np.sum(mask))})')
        ax.set_xlim(0, grid_dim)
        ax.set_ylim(0, grid_dim)
    
    plt.tight_layout()
    plt.savefig('sugarscape_snapshots.png')
    print("Saved sugarscape_snapshots.png")

    print("Visualization complete.")

if __name__ == "__main__":
    # Change to the directory of the script if logs aren't in CWD
    if not os.path.exists('bugs_log.csv'):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir:
            os.chdir(script_dir)
    visualize()
