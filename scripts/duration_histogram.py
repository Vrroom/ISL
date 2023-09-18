import os
import json
import matplotlib.pyplot as plt
import argparse

def get_durations_from_jsons(directory):
    durations = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                if 'duration' in data:
                    durations.append(data['duration'])
    return durations

def plot_histogram(durations):
    plt.hist(durations, bins=1000, alpha=0.7, color='blue')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Durations')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot histogram of durations from JSON files in a directory.')
    parser.add_argument('--directory', type=str, help='Directory containing JSON files')
    args = parser.parse_args()

    durations = get_durations_from_jsons(args.directory)
    plot_histogram(durations)
