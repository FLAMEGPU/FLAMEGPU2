import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os
import json
import xml.etree.ElementTree as ET
import glob

def parse_xml_step(file_path):
    """Parses a FLAME GPU 2 exportData XML file to extract agent information."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        agents = []
        for xagent in root.findall('xagent'):
            if xagent.find('name') is not None and xagent.find('name').text == 'bee':
                _id_elem = xagent.find('_id')
                id_val = int(_id_elem.text) if _id_elem is not None else 0
                x_val = int(xagent.find('x').text)
                y_val = int(xagent.find('y').text)
                hunger_val = float(xagent.find('hunger_level').text)
                wait_val = int(xagent.find('wait').text)
                
                agents.append({
                    'id': id_val,
                    'x': x_val,
                    'y': y_val,
                    'hunger_level': hunger_val,
                    'wait': wait_val
                })
        return agents
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []

def parse_flowers(file_path):
    """Parses the initial_state XML to find flower positions (flower_cell with nectar > 0)."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        flowers = []
        for xagent in root.findall('xagent'):
            if xagent.find('name') is not None and xagent.find('name').text == 'flower_cell':
                nectar = float(xagent.find('nectar').text)
                if nectar > 0:
                    flowers.append({
                        'x': int(xagent.find('x').text),
                        'y': int(xagent.find('y').text)
                    })
        return flowers
    except Exception as e:
        print(f"Error parsing flowers from {file_path}: {e}")
        return []

def visualize():
    # Load Summary Statistics from simulation_log.json
    try:
        if not os.path.exists('simulation_log.json'):
            print("simulation_log.json not found. Run the simulation first.")
            return
            
        with open('simulation_log.json', 'r') as f:
            log_data = json.load(f)
            
        steps_info = log_data.get('steps', [])
        unique_steps = [s['step_index'] for s in steps_info]
        avg_hunger = [s['agents']['bee']['default'][0]['variables']['hunger_level']['mean'] for s in steps_info]
        avg_wait = [s['agents']['bee']['default'][0]['variables']['wait']['mean'] for s in steps_info]
    except Exception as e:
        print(f"Error loading summary statistics: {e}")
        # If simulation hasn't run or log is incomplete, we might get NaNs here
        return

    grid_dim = 100

    # 1. Plot Average Hunger Level and Wait over time
    print("Generating statistics plot...")
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

    # Load Spatial Data from XML files
    print("Loading spatial data from XML files...")
    flower_list = parse_flowers('step_0.xml')
    flower_x = [f['x'] for f in flower_list]
    flower_y = [f['y'] for f in flower_list]
    
    # Collect all step files
    step_files = sorted(glob.glob('step_*.xml'), key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # We'll build a data structure similar to what the CSV provided to minimize changes to the plotting logic
    # but only for the steps we need (to save memory)
    all_bees_by_step = {}
    for i, f_path in enumerate(step_files):
        # Only parse steps we actually use for plots/animation to save time
        # unique_steps includes 0 (initial), but step files start from 1
        step_idx = i + 1 
        all_bees_by_step[step_idx] = parse_xml_step(f_path)

    # 2. Animation
    print("Creating animation (this might take a moment)...")
    fig_anim, ax_anim = plt.subplots(figsize=(8, 8))
    
    # Plot static flowers
    ax_anim.scatter(flower_x, flower_y, c='green', marker='*', s=60, alpha=0.4, label='Flowers')
    
    # Initial bee plot (from step 1)
    first_step_bees = all_bees_by_step.get(1, [])
    if not first_step_bees:
        print("No bee data found for step 1.")
        return

    bx = [b['x'] for b in first_step_bees]
    by = [b['y'] for b in first_step_bees]
    bh = [b['hunger_level'] for b in first_step_bees]

    scat = ax_anim.scatter(bx, by, c=bh, 
                           cmap='YlOrRd', vmin=0, vmax=100,
                           marker='o', s=30, edgecolors='k', linewidths=0.5, label='Bees')
    
    cbar = plt.colorbar(scat, ax=ax_anim)
    cbar.set_label('Hunger Level')

    ax_anim.set_xlim(-1, grid_dim)
    ax_anim.set_ylim(-1, grid_dim)
    ax_anim.set_title(f'Bee Simulation - Step 1')
    ax_anim.legend(loc='upper right')

    def update(frame):
        s = frame + 1
        bees = all_bees_by_step.get(s, [])
        if bees:
            bx_u = [b['x'] for b in bees]
            by_u = [b['y'] for b in bees]
            bh_u = [b['hunger_level'] for b in bees]
            scat.set_offsets(np.c_[bx_u, by_u])
            scat.set_array(np.array(bh_u))
            ax_anim.set_title(f'Bee Simulation - Step {s}')
        return scat,

    ani = animation.FuncAnimation(fig_anim, update, frames=len(step_files), interval=100, blit=True)
    
    try:
        ani.save('bee_simulation.gif', writer='pillow', fps=10)
        print("Saved bee_simulation.gif")
    except Exception as e:
        print(f"Could not save animation: {e}")

    # 3. Static snapshots
    print("Generating movement snapshots...")
    steps_to_plot = [1, 25, 50, 75, 100]
    steps_to_plot = [s for s in steps_to_plot if s in all_bees_by_step]

    fig_snap, axes = plt.subplots(1, len(steps_to_plot), figsize=(20, 4))
    if len(steps_to_plot) == 1:
        axes = [axes]

    for i, s in enumerate(steps_to_plot):
        ax = axes[i]
        bees = all_bees_by_step[s]
        bx_s = [b['x'] for b in bees]
        by_s = [b['y'] for b in bees]
        bh_s = [b['hunger_level'] for b in bees]

        ax.scatter(flower_x, flower_y, c='green', marker='*', s=30, alpha=0.3)
        ax.scatter(bx_s, by_s, c=bh_s, 
                   cmap='YlOrRd', vmin=0, vmax=100, marker='o', s=30, edgecolors='k', linewidths=0.5)

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
    
    # Get all unique IDs from the first step
    unique_ids = [b['id'] for b in all_bees_by_step[1]]
    np.random.seed(42)
    sample_ids = np.random.choice(unique_ids, min(15, len(unique_ids)), replace=False)
    
    plt.scatter(flower_x, flower_y, c='green', marker='*', s=100, alpha=0.2, label='Flowers')

    for bid in sample_ids:
        traj_x = []
        traj_y = []
        for s in sorted(all_bees_by_step.keys()):
            # Find the bee with this ID in this step
            bee = next((b for b in all_bees_by_step[s] if b['id'] == bid), None)
            if bee:
                traj_x.append(bee['x'])
                traj_y.append(bee['y'])
        
        if traj_x:
            plt.plot(traj_x, traj_y, marker='.', alpha=0.6, linewidth=1)
            # Mark start
            plt.scatter(traj_x[0], traj_y[0], marker='o', c='blue', s=30, zorder=5)
            # Mark end
            plt.scatter(traj_x[-1], traj_y[-1], marker='x', c='red', s=40, zorder=5)

    plt.title('Sample Bee Trajectories (15 Bees)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, grid_dim)
    plt.ylim(0, grid_dim)
    plt.gca().set_aspect('equal')
    
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='green', marker='*', linestyle='None', markersize=10, alpha=0.3),
                    Line2D([0], [0], color='gray', marker='.', linestyle='-', alpha=0.6),
                    Line2D([0], [0], color='blue', marker='o', linestyle='None', markersize=8),
                    Line2D([0], [0], color='red', marker='x', linestyle='None', markersize=8)]
    plt.legend(custom_lines, ['Flowers', 'Bee Paths', 'Start', 'End'], loc='upper right')
    
    plt.savefig('bee_trajectories.png')
    print("Saved bee_trajectories.png")

    print("Visualization complete.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    if not os.path.exists(output_dir):
        print(f"Directory {output_dir} not found. Run the simulation first.")
    else:
        os.chdir(output_dir)
        visualize()

