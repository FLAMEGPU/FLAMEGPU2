import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os

def visualize():
    # Load data
    try:
        # Load bees_log.csv using numpy
        # Header: step,id,x,y,hunger_level,wait
        if not os.path.exists('bees_log.csv'):
            print("bees_log.csv not found.")
            return
        bees_data = np.genfromtxt('bees_log.csv', delimiter=',', skip_header=1)
        
        # Load flowers_log.csv using numpy
        # Header: id,x,y,nectar
        if not os.path.exists('flowers_log.csv'):
            print("flowers_log.csv not found.")
            return
        flowers_data = np.genfromtxt('flowers_log.csv', delimiter=',', skip_header=1)
    except Exception as e:
        print(f"Error loading logs: {e}")
        return

    # Extract columns
    step = bees_data[:, 0]
    bee_id = bees_data[:, 1]
    bee_x = bees_data[:, 2]
    bee_y = bees_data[:, 3]
    hunger_level = bees_data[:, 4]
    wait = bees_data[:, 5]

    flower_x = flowers_data[:, 1]
    flower_y = flowers_data[:, 2]
    # flower_nectar = flowers_data[:, 3] # unused for now

    unique_steps = np.sort(np.unique(step)).astype(int)
    grid_dim = 100

    # 1. Plot Average Hunger Level and Wait over time
    print("Generating statistics plot...")
    avg_hunger = []
    avg_wait = []
    for s in unique_steps:
        mask = (step == s)
        avg_hunger.append(np.mean(hunger_level[mask]))
        avg_wait.append(np.mean(wait[mask]))

    fig_stats, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Avg Hunger Level', color='tab:red')
    ax1.plot(unique_steps, avg_hunger, color='tab:red', linewidth=2, label='Avg Hunger Level')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Avg Wait', color='tab:blue')
    ax2.plot(unique_steps, avg_wait, color='tab:blue', linewidth=2, label='Avg Wait')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title('Average Bee Hunger Level and Wait over Time')
    fig_stats.tight_layout()
    plt.savefig('hunger_wait_plot.png')
    print("Saved hunger_wait_plot.png")

    # 2. Animation
    print("Creating animation (this might take a moment)...")
    fig_anim, ax_anim = plt.subplots(figsize=(8, 8))
    
    # Plot static flowers
    ax_anim.scatter(flower_x, flower_y, c='green', marker='*', s=60, alpha=0.4, label='Flowers')
    
    # Initial bee plot
    mask0 = (step == unique_steps[0])
    # Color bees by hunger level: yellow (full) to red (starving)
    scat = ax_anim.scatter(bee_x[mask0], bee_y[mask0], c=hunger_level[mask0], 
                           cmap='YlOrRd', vmin=0, vmax=100,
                           marker='o', s=30, edgecolors='k', linewidths=0.5, label='Bees')
    
    cbar = plt.colorbar(scat, ax=ax_anim)
    cbar.set_label('Hunger Level')

    ax_anim.set_xlim(-1, grid_dim)
    ax_anim.set_ylim(-1, grid_dim)
    ax_anim.set_title(f'Bee Simulation - Step {unique_steps[0]}')
    ax_anim.legend(loc='upper right')

    def update(frame):
        s = unique_steps[frame]
        mask = (step == s)
        # Update bee positions
        scat.set_offsets(np.c_[bee_x[mask], bee_y[mask]])
        # Update colors based on hunger level
        scat.set_array(hunger_level[mask])
        ax_anim.set_title(f'Bee Simulation - Step {s}')
        return scat,

    ani = animation.FuncAnimation(fig_anim, update, frames=len(unique_steps), interval=100, blit=True)
    
    # Save animation (requires ffmpeg or pillow)
    try:
        import PIL
        ani.save('bee_simulation.gif', writer='pillow', fps=10)
        print("Saved bee_simulation.gif")
    except ImportError:
        print("Pillow not found, skipping GIF save.")
    except Exception as e:
        print(f"Could not save animation: {e}")

    # 3. Static snapshots
    print("Generating movement snapshots...")
    steps_to_plot = [0, 25, 50, 75, 99]
    steps_to_plot = [s for s in steps_to_plot if s in unique_steps]

    fig_snap, axes = plt.subplots(1, len(steps_to_plot), figsize=(20, 4))
    if len(steps_to_plot) == 1:
        axes = [axes]

    for i, s in enumerate(steps_to_plot):
        ax = axes[i]
        mask = (step == s)

        ax.scatter(flower_x, flower_y, c='green', marker='*', s=30, alpha=0.3)
        ax.scatter(bee_x[mask], bee_y[mask], c=hunger_level[mask], 
                   cmap='YlOrRd', vmin=0, vmax=100, marker='o', s=10)

        ax.set_title(f'Step {s}')
        ax.set_xlim(0, grid_dim)
        ax.set_ylim(0, grid_dim)
        ax.set_aspect('equal')

    plt.suptitle('Bee and Flower Positions over Time')
    plt.tight_layout()
    plt.savefig('movement_snapshots.png')
    print("Saved movement_snapshots.png")

    # 4. Individual bee trajectories (sample 15 bees)
    print("Generating trajectories plot...")
    plt.figure(figsize=(10, 10))
    unique_ids = np.unique(bee_id)
    np.random.seed(42)
    sample_ids = np.random.choice(unique_ids, min(15, len(unique_ids)), replace=False)
    
    plt.scatter(flower_x, flower_y, c='green', marker='*', s=100, alpha=0.2, label='Flowers')

    for bid in sample_ids:
        mask = (bee_id == bid)
        idx = np.argsort(step[mask])
        plt.plot(bee_x[mask][idx], bee_y[mask][idx], marker='.', alpha=0.6, linewidth=1)
        # Mark start
        plt.scatter(bee_x[mask][idx][0], bee_y[mask][idx][0], marker='o', c='blue', s=30, zorder=5)
        # Mark end
        plt.scatter(bee_x[mask][idx][-1], bee_y[mask][idx][-1], marker='x', c='red', s=40, zorder=5)

    plt.title('Sample Bee Trajectories (15 Bees)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, grid_dim)
    plt.ylim(0, grid_dim)
    plt.gca().set_aspect('equal')
    # Custom legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='green', marker='*', linestyle='None', markersize=10, alpha=0.3),
                    Line2D([0], [0], color='gray', marker='.', linestyle='-', alpha=0.6),
                    Line2D([0], [0], color='blue', marker='o', linestyle='None', markersize=8),
                    Line2D([0], [0], color='red', marker='x', linestyle='None', markersize=8)]
    plt.legend(custom_lines, ['Flowers', 'Bee Paths', 'Start', 'End'], loc='upper right')
    
    plt.savefig('bee_trajectories.png')
    print("Saved bee_trajectories.png")

    print("Visualization complete.")
    # plt.show()

if __name__ == "__main__":
    # Change to the directory of the script to ensure logs are found
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    visualize()
