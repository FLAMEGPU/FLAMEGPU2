import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os
import json
import xml.etree.ElementTree as ET
import glob

def parse_xml_step_bugs(file_path):
    """Parses XML step file to extract bug positions and sugar levels."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        bugs = []
        for xagent in root.findall('xagent'):
            if xagent.find('name') is not None and xagent.find('name').text == 'bug':
                _id_elem = xagent.find('_id')
                id_val = int(_id_elem.text) if _id_elem is not None else 0
                
                # Check for both 'x' and 'y' (old) and 'pos' (new array)
                pos_elem = xagent.find('pos')
                if pos_elem is not None:
                    # pos is an array, often exported as space separated or multiple tags?
                    # In XML it's usually <pos>0 0</pos> or similar depending on FLAMEGPU version
                    pos_text = pos_elem.text.strip().split()
                    x_val = int(pos_text[0])
                    y_val = int(pos_text[1])
                else:
                    x_val = int(xagent.find('x').text)
                    y_val = int(xagent.find('y').text)
                
                sugar_val = float(xagent.find('sugar_level').text)
                metabolism_val = float(xagent.find('metabolism').text)
                
                bugs.append({
                    'id': id_val,
                    'x': x_val,
                    'y': y_val,
                    'sugar_level': sugar_val,
                    'metabolism': metabolism_val
                })
        return bugs
    except Exception as e:
        print(f"Error parsing bugs from {file_path}: {e}")
        return []

def parse_xml_step_cells(file_path):
    """Parses XML step file to extract sugar_cell environmental sugar levels."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        cells = []
        for xagent in root.findall('xagent'):
            if xagent.find('name') is not None and xagent.find('name').text == 'sugar_cell':
                # Check for both 'x' and 'y' (old) and 'pos' (new array)
                pos_elem = xagent.find('pos')
                if pos_elem is not None:
                    pos_text = pos_elem.text.strip().split()
                    x_val = int(pos_text[0])
                    y_val = int(pos_text[1])
                else:
                    x_val = int(xagent.find('x').text)
                    y_val = int(xagent.find('y').text)
                
                env_sugar = float(xagent.find('env_sugar_level').text)
                env_max = float(xagent.find('env_max_sugar_level').text)
                
                cells.append({
                    'x': x_val,
                    'y': y_val,
                    'env_sugar_level': env_sugar,
                    'env_max_sugar_level': env_max
                })
        return cells
    except Exception as e:
        print(f"Error parsing cells from {file_path}: {e}")
        return []

def visualize():
    # Load Spatial Data from XML files first so we can count bugs
    print("Loading spatial data from XML files (this may take a while for cells)...")
    step_files = sorted(glob.glob('step_*.xml'), key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    if not step_files:
        print("No step_*.xml files found in output directory.")
        return

    all_bugs_by_step = {}
    all_cells_by_step = {}
    
    cell_load_interval = 10
    
    for f_path in step_files:
        step_idx = int(f_path.split('_')[1].split('.')[0])
        # Always load bugs
        all_bugs_by_step[step_idx] = parse_xml_step_bugs(f_path)
        # Load cells periodically (it's heavy, ~65,536 lines each step)
        if step_idx == 0 or step_idx % cell_load_interval == 0 or step_idx == int(step_files[-1].split('_')[1].split('.')[0]):
            all_cells_by_step[step_idx] = parse_xml_step_cells(f_path)

    # Load Summary Statistics from simulation_log.json
    try:
        if not os.path.exists('simulation_log.json'):
            print("simulation_log.json not found. Run the simulation first.")
            return
            
        with open('simulation_log.json', 'r') as f:
            log_data = json.load(f)
            
        steps_info = log_data.get('steps', [])
        unique_bug_steps = [s['step_index'] for s in steps_info]
        # In python step index starts at 0 for step log? 
        # Actually it's 1, 2... for steps.
        
        # We use XML counts for consistency with spatial visualization
        bug_count = [len(all_bugs_by_step.get(s, [])) for s in unique_bug_steps]
        avg_bug_sugar = [s['agents']['bug']['default'][0]['variables']['sugar_level']['mean'] for s in steps_info]
        avg_cell_sugar = [s['agents']['sugar_cell']['default'][0]['variables']['env_sugar_level']['mean'] for s in steps_info]
    except Exception as e:
        print(f"Error loading summary statistics: {e}")
        # Fallback to just using what we have in all_bugs_by_step if JSON fails
        unique_bug_steps = sorted(all_bugs_by_step.keys())
        bug_count = [len(all_bugs_by_step[s]) for s in unique_bug_steps]
        avg_bug_sugar = [0] * len(unique_bug_steps)
        avg_cell_sugar = [0] * len(unique_bug_steps)

    grid_dim = 256

    # 1. Statistics Plot
    print("Generating statistics plot...")
    fig_stats, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Bug Count / Avg Sugar', color='tab:red')
    ax1.plot(unique_bug_steps, bug_count, color='tab:red', label='Bug Count')
    ax1.plot(unique_bug_steps, avg_bug_sugar, color='tab:orange', label='Avg Bug Sugar')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Avg Cell Sugar', color='tab:blue')
    ax2.plot(unique_bug_steps, avg_cell_sugar, color='tab:blue', marker='o', markersize=3, label='Avg Cell Sugar')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.legend(loc='upper right')

    plt.title('Sugarscape Simulation Statistics')
    fig_stats.tight_layout()
    plt.savefig('../sugarscape_stats.png')
    print("Saved sugarscape_stats.png")

    # Helper to get grid at a specific step
    def get_grid(step):
        # find closest available cell step (rounding down)
        available_steps = sorted(all_cells_by_step.keys())
        if not available_steps:
            return np.zeros((grid_dim, grid_dim))
        s = available_steps[0]
        for a_s in available_steps:
            if a_s <= step:
                s = a_s
            else:
                break
                
        cells = all_cells_by_step[s]
        grid = np.zeros((grid_dim, grid_dim))
        for c in cells:
            val = c['env_sugar_level']
            if val < 0: val = 0
            grid[c['x'], c['y']] = val
        return grid

    # 2. Animation
    print("Creating animation (this might take a moment)...")
    fig_anim, ax_anim = plt.subplots(figsize=(8, 8))
    
    first_step_idx = sorted(all_bugs_by_step.keys())[0]
    grid_img = ax_anim.imshow(get_grid(first_step_idx).T, origin='lower', cmap='YlOrBr', 
                             extent=[0, grid_dim, 0, grid_dim], vmin=0, vmax=7)
    
    first_step_bugs = all_bugs_by_step.get(first_step_idx, [])
    bx = [b['x'] for b in first_step_bugs]
    by = [b['y'] for b in first_step_bugs]
    scat = ax_anim.scatter(bx, by, c='red', s=2, alpha=0.6, label='Bugs')
    
    plt.colorbar(grid_img, ax=ax_anim, label='Sugar Level')
    ax_anim.set_title(f'Sugarscape - Step {first_step_idx}')
    ax_anim.set_xlim(0, grid_dim)
    ax_anim.set_ylim(0, grid_dim)

    def update(frame):
        s = frames[frame]
        if s in all_cells_by_step:
            grid_img.set_data(get_grid(s).T)
        
        bugs = all_bugs_by_step.get(s, [])
        if bugs:
            bx_u = [b['x'] for b in bugs]
            by_u = [b['y'] for b in bugs]
            scat.set_offsets(np.c_[bx_u, by_u])
        ax_anim.set_title(f'Sugarscape - Step {s} (Bugs: {len(bugs)})')
        return grid_img, scat

    frames = sorted(all_bugs_by_step.keys())
    # Limit frames if too many
    if len(frames) > 100:
        frames = frames[::len(frames)//100]

    ani = animation.FuncAnimation(fig_anim, update, frames=len(frames), interval=100, blit=True)
    
    try:
        ani.save('../sugarscape_simulation.gif', writer='pillow', fps=10)
        print("Saved sugarscape_simulation.gif")
    except Exception as e:
        print(f"Could not save animation: {e}")

    # 3. Static snapshots (Start, Mid, End)
    print("Generating snapshots...")
    sorted_steps = sorted(all_bugs_by_step.keys())
    if len(sorted_steps) >= 3:
        steps_to_plot = [sorted_steps[0], sorted_steps[len(sorted_steps)//2], sorted_steps[-1]]
    else:
        steps_to_plot = sorted_steps
        
    fig_snap, axes = plt.subplots(1, len(steps_to_plot), figsize=(18, 6))
    if len(steps_to_plot) == 1:
        axes = [axes]
        
    for i, s in enumerate(steps_to_plot):
        ax = axes[i]
        grid = get_grid(s)
        ax.imshow(grid.T, origin='lower', cmap='YlOrBr', extent=[0, grid_dim, 0, grid_dim], vmin=0, vmax=7)
        
        bugs = all_bugs_by_step.get(s, [])
        bx_s = [b['x'] for b in bugs]
        by_s = [b['y'] for b in bugs]
        
        ax.scatter(bx_s, by_s, c='red', s=1, alpha=0.5)
        ax.set_title(f'Step {s} (Bugs: {len(bugs)})')
        ax.set_xlim(0, grid_dim)
        ax.set_ylim(0, grid_dim)
    
    plt.tight_layout()
    plt.savefig('../sugarscape_snapshots.png')
    print("Saved sugarscape_snapshots.png")

    print("Visualization complete.")

if __name__ == "__main__":
    # Look for output directory relative to script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    if not os.path.exists(output_dir):
        # Check current working directory
        if os.path.exists("output"):
            output_dir = os.path.abspath("output")
        else:
            print(f"Directory {output_dir} not found. Run the simulation first.")
            exit(1)
            
    os.chdir(output_dir)
    visualize()
