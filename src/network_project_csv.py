import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import laplace
from scipy.stats import laplace
from scipy.stats import probplot
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

csv_files = ["group1.csv", "group2.csv", "group3.csv", "group4.csv", "group5.csv", "mix.csv",
             "record_with_background_noise_text.csv",
             "record_with_background_noise_voice.csv", "record_with_background_noise_pic.csv",
             "record_with_background_noise_video.csv", "record_with_background_noise_files.csv"]
group_names = ["Group 1", "Group 2", "Group 3", "Group 4", "Group 5", "mix", "record_with_background_noise_text",
               "record_with_background_noise_voice", "record_with_background_noise_pic",
               "record_with_background_noise_video", "record_with_background_noise_files"]


def extract_data_from_csv(csv_file):
    df = pd.read_csv(csv_file)

    # Convert 'frame.time' column to datetime
    df['Time'] = pd.to_datetime(df['Time'])

    # Convert datetime to seconds since the first timestamp
    ref_time = df['Time'].iloc[0]
    df['seconds'] = (df['Time'] - ref_time).dt.total_seconds()

    message_timestamps = df['seconds'].tolist()
    message_sizes = df['Length'].tolist()

    return message_timestamps, message_sizes


def compute_inter_message_delays(message_timestamps):
    delays = np.diff(message_timestamps)
    return delays


def plot_data_for_group(group_name, delays, sizes):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.hist(delays, bins=50)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title(f"Inter-message Delays for {group_name}")
    plt.xlabel("Delay (s)")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(sizes, bins=50)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title(f"Message Sizes for {group_name}")
    plt.xlabel("Size (Bytes)")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def analyze_characteristics(all_delays, all_sizes):
    # Here we use k-means clustering to find unique characteristics
    # Combine delays and sizes for clustering
    X = np.column_stack((all_delays, all_sizes))
    kmeans = KMeans(n_clusters=len(group_names))
    kmeans.fit(X)

    # Check cluster centers
    cluster_centers = kmeans.cluster_centers_
    for idx, center in enumerate(cluster_centers):
        print(f"Group {idx + 1} center: Delay {center[0]}, Size {center[1]}")

    return kmeans.labels_


def deduce_group_membership(all_delays, all_sizes, labels):
    # This is a basic clustering approach, one can use advanced techniques and ML models for better accuracy
    X = np.column_stack((all_delays, all_sizes))
    predictions = KMeans(n_clusters=len(group_names)).fit_predict(X)
    for idx, group in enumerate(group_names):
        if predictions[idx] == labels[idx]:
            print(f"User might belong to {group}")
        else:
            print(f"User might not belong to {group}")


def analyze_unique_characteristics(all_delays, all_sizes):
    # Statistical Analysis for each group
    for idx, group in enumerate(group_names):
        mean_delay = np.mean(all_delays[idx])
        std_delay = np.std(all_delays[idx])
        mean_size = np.mean(all_sizes[idx])
        std_size = np.std(all_sizes[idx])

        print(f"Statistics for {group}:")
        print(f"Mean Inter-message Delay: {mean_delay}")
        print(f"Standard Deviation of Inter-message Delay: {std_delay}")
        print(f"Mean Message Size: {mean_size}")
        print(f"Standard Deviation of Message Size: {std_size}")
        print("\n")


def deduce_group_membership_ML(all_delays, all_sizes, labels):
    # This uses a Random Forest classifier for deducing group membership
    X = np.column_stack((all_delays, all_sizes))
    y = labels

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Classifier Accuracy: {accuracy * 100:.2f}%")

    for idx, group in enumerate(group_names):
        if y_pred[idx] == labels[idx]:
            print(f"User might belong to {group}")
        else:
            print(f"User might not belong to {group}")


def plot_packet_length_vs_time(timestamps, sizes, group_name):
    plt.figure(figsize=(10, 5))
    # Plots a vertical line from 0 to the packet size for each timestamp
    for t, s in zip(timestamps, sizes):
        plt.plot([t, t], [0, s], 'b-')
    # plots background lines
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title(f"Packet Length vs. Time for {group_name}")
    plt.xlabel("Time")
    plt.ylabel("Packet Length (Bytes)")
    plt.xlim(min(timestamps), max(timestamps))
    plt.show()


def plot_pdf_of_delays(delays, group_name):
    plt.figure(figsize=(10, 5))
    sns.kdeplot(delays, shade=True, color='r')
    plt.title(f"PDF of Inter-message Delays for {group_name}")
    plt.xlabel("Inter-message Delays (Seconds)")
    plt.ylabel("Probability Density")
    plt.grid(True, which="both", ls="--", c='0.65')
    plt.show()


def categorize_by_size(size):
    if size <= 100:
        return "text"
    elif size <= 500:
        return "photo"
    elif size <= 1000:
        return "video"
    elif size <= 1500:
        return "file"
    else:
        return "audio"


def plot_ccdf_from_data(data, label, max_val):
    """
    Plot CCDF of data on normalized message sizes to their maximum.
    """
    # Normalize data
    normalized_data = data / max_val
    sorted_data = np.sort(normalized_data)
    ccdf = np.arange(len(sorted_data), 0, -1) / len(sorted_data)

    plt.plot(sorted_data, ccdf, label=label)


def plot_ccdf_from_csv(file_path):

    df = pd.read_csv(file_path)

    # Extract and categorize Length column
    df['Category'] = df['Length'].apply(categorize_by_size)

    plt.figure(figsize=(10, 6))

    categories = ["text", "photo", "video", "file", "audio"]
    max_val = df['Length'].max()
    for category in categories:
        lengths = df[df['Category'] == category]['Length'].values
        # Check the number of data points for each category
        print(f"Number of data points for {category}: {len(lengths)}")
        plot_ccdf_from_data(lengths, category, max_val)

    plt.ylabel("CCDF")
    plt.xlabel("Normalized Packet Size")
    plt.title("CCDF of Normalized Packet Sizes")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()


from scipy.stats import laplace, probplot


def plot_qq_laplace(delays):
    # Calculate quantiles
    sorted_delays = np.sort(delays)
    probs = np.arange(1, len(sorted_delays) + 1) / (len(sorted_delays) + 1)

    laplace_quantiles = laplace.ppf(probs)

    plt.figure(figsize=(8, 8))
    plt.plot(laplace_quantiles, sorted_delays, "o", label="Data Quantiles")
    plt.plot([min(laplace_quantiles), max(laplace_quantiles)],
             [min(laplace_quantiles), max(laplace_quantiles)], 'r-', label="y=x")
    plt.xlabel("Theoretical Laplace Quantiles")
    plt.ylabel("Data Quantiles")
    plt.title("Quantile-Quantile plot of Delays against Laplace Distribution")
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.65')
    plt.show()


def main():
    all_delays = []
    all_sizes = []
    labels = []
    plot_ccdf_from_csv('mix.csv')

    for csv_file, group_name in zip(csv_files, group_names):
        message_timestamps, message_sizes = extract_data_from_csv(csv_file)
        delays = compute_inter_message_delays(message_timestamps)

        all_delays.extend(delays)
        all_sizes.extend(message_sizes)

        plot_packet_length_vs_time(message_timestamps, message_sizes, group_name)
        plot_data_for_group(group_name, delays, message_sizes)
        plot_pdf_of_delays(delays, group_name)
        plot_qq_laplace(delays)
        labels.extend([group_name] * len(delays))

    analyze_unique_characteristics(all_delays, all_sizes)
    deduce_group_membership_ML(all_delays, all_sizes, labels)


main()
