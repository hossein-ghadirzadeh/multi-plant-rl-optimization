import re
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
LOG_FILE_PATH = 'training_logs.txt'
PLOTS_DIR = 'plots'

def parse_logs(log_file):
    """
    Parses the training log file to extract win rates and metrics for each combination.
    """
    if not os.path.exists(log_file):
        print(f"Error: Log file not found at '{log_file}'")
        return None

    with open(log_file, 'r') as f:
        content = f.read()

    # Regex to find the *final* evaluation block for each combination.
    # This is more robust as it specifically looks for the eval that follows the 10-match demo.
    final_eval_pattern = re.compile(
        r"Right\(DQN\) win rate over 10 matches: [\d.]+\s+Final eval \(50 eps\) Right\(DQN\) win rate: ([\d.]+)",
        re.DOTALL
    )
    win_rates = [float(rate) for rate in final_eval_pattern.findall(content)]
    
    # Regex to find the summary of the best combination at the very end
    best_summary_match = re.search(r"=== Grid Search Complete ===\s+Best Win Rate: ([\d.]+)", content, re.DOTALL)

    if len(win_rates) != 32 or not best_summary_match:
        print(f"Error: Could not parse all 32 combinations. Found {len(win_rates)}. The log file might be incomplete or malformed.")
        return None
        
    # --- Find the best combination's data ---
    best_win_rate_summary = float(best_summary_match.group(1))
    
    # Find the index of the best combination by matching the summary win rate
    best_combo_index = -1
    for i, rate in enumerate(win_rates):
        if round(rate, 3) == round(best_win_rate_summary, 3):
            best_combo_index = i
            break
            
    if best_combo_index == -1:
        print("Warning: Could not definitively match the best summary win rate to a specific combination. Using max value as fallback.")
        best_combo_index = np.argmax(win_rates)

    best_combo_number = best_combo_index + 1
    
    # To find the metrics of the best combo, we first split the log by "=== Combination"
    # to isolate the correct block of text.
    all_combo_sections = re.split(r"=== Combination \d+/32 ===", content)
    # The first split is empty, so we look at index `best_combo_number`
    best_combo_log_section = all_combo_sections[best_combo_number]

    # Now find the specific metrics within that specific section
    best_combo_metrics_pattern = re.compile(
        r"Final eval .*? metrics: \{'avg_rally_left': ([\d.]+), 'avg_rally_right': ([\d.]+)",
        re.DOTALL
    )
    best_combo_metrics_match = best_combo_metrics_pattern.search(best_combo_log_section)
    
    if not best_combo_metrics_match:
        print(f"Error: Could not find detailed metrics for the best combination (#{best_combo_number}).")
        return {"all_win_rates": win_rates, "best_metrics": None}

    best_metrics = {
        "combo_number": best_combo_number,
        "win_rate": win_rates[best_combo_index],
        "avg_rally_left": float(best_combo_metrics_match.group(1)),
        "avg_rally_right": float(best_combo_metrics_match.group(2))
    }
    
    return {
        "all_win_rates": win_rates,
        "best_metrics": best_metrics
    }

def plot_win_rate_trend(win_rates, best_combo_number):
    """
    Plots a bar chart of the win rate for each combination.
    """
    if not win_rates:
        return
        
    combo_numbers = range(1, len(win_rates) + 1)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 7))
    
    colors = ['#4c72b0' for _ in win_rates]
    best_idx = best_combo_number - 1
    if 0 <= best_idx < len(colors):
        colors[best_idx] = '#c44e52' # Highlight the best one
    
    plt.bar(combo_numbers, win_rates, color=colors)
    
    plt.title('DQN Win Rate Across 32 Hyperparameter Combinations', fontsize=16, pad=20)
    plt.xlabel('Hyperparameter Combination Number', fontsize=12)
    plt.ylabel('Final Evaluation Win Rate (%)', fontsize=12)
    plt.xticks(np.arange(1, 33, 2)) # Show ticks for every other combo
    plt.yticks([i/10 for i in range(11)])
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.legend([plt.Rectangle((0,0),1,1, color='#c44e52')], ['Best Performing Model'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'win_rate_trend.png'), dpi=300)
    print(f"Saved win rate trend plot to '{os.path.join(PLOTS_DIR, 'win_rate_trend.png')}'")
    plt.show()

def plot_rally_length_comparison(best_metrics):
    """
    Plots a bar chart comparing the rally length of Q-Learning vs DQN for the best model.
    """
    if not best_metrics:
        return
        
    labels = ['Q-Learning Agent', 'DQN Agent (Best Model)']
    values = [best_metrics['avg_rally_left'], best_metrics['avg_rally_right']]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))
    
    bars = plt.bar(labels, values, color=['#4c72b0', '#c44e52'])
    
    plt.title(f'Average Rally Length Comparison (Best Model - Combo #{best_metrics["combo_number"]})', fontsize=14, pad=20)
    plt.ylabel('Average Rally Length (Hits per Rally)', fontsize=12)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'rally_length_comparison.png'), dpi=300)
    print(f"Saved rally length comparison plot to '{os.path.join(PLOTS_DIR, 'rally_length_comparison.png')}'")
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    parsed_data = parse_logs(LOG_FILE_PATH)
    
    if parsed_data and parsed_data["best_metrics"]:
        print(f"Successfully parsed logs. Best combination found: #{parsed_data['best_metrics']['combo_number']}")
        plot_win_rate_trend(parsed_data['all_win_rates'], parsed_data['best_metrics']['combo_number'])
        plot_rally_length_comparison(parsed_data['best_metrics'])
        print("\nAnalysis complete.")
    else:
        print("\nAnalysis failed. Please check the log file.")