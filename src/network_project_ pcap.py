import matplotlib.pyplot as plt
import numpy as np
from scapy.all import rdpcap
from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

pcap_files = ["group1.pcap", "group2.pcap", "group3.pcap", "group4.pcap", "group5.pcap", "mix.pcap",
              "record_with_background_noise_text.pcap", "record_with_background_noise_voice.pcap",
              "record_with_background_noise_pic.pcap", "record_with_background_noise_video.pcap",
              "record_with_background_noise_files.pcap"]
group_names = ["Group 1", "Group 2", "Group 3", "Group 4", "Group 5", "mix", "record_with_background_noise_text",
               "record_with_background_noise_voice", "record_with_background_noise_pic",
               "record_with_background_noise_video", "record_with_background_noise_files"]


def extract_data_from_pcap(pcap_file):
    packets = rdpcap(pcap_file)

    # Filter for IM traffic (this filter is hypothetical and might vary based on the actual protocol)
    # For example, WhatsApp uses port 5222 for traffic (this might not be the case, you'll need to check)
    im_packets = [p for p in packets if p.haslayer('TCP') and ((p['TCP'].sport == 45806 or p['TCP'].sport == 45948 or
                                                                p['TCP'].sport == 59888 or p['TCP'].sport == 43620)
                                                               or p['TCP'].dport == 443)]

    message_timestamps = [p.time for p in im_packets]
    message_sizes = [len(p) for p in im_packets]

    return message_timestamps, message_sizes


def compute_inter_message_delays(message_timestamps):
    delays = np.diff(message_timestamps)
    return delays


def plot_data_for_group(group_name, delays, sizes):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.hist(delays, bins=50)
    plt.title(f"Inter-message Delays for {group_name}")
    plt.xlabel("Delay (s)")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(sizes, bins=50)
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


# def plot_packet_length_vs_time(timestamps, sizes, group_name):
#     plt.figure(figsize=(10, 5))
#     plt.plot(timestamps, sizes, '-')
#     plt.scatter(timestamps, sizes, alpha=0.5)
#     plt.title(f"Packet Length vs. Time for {group_name}")
#     plt.xlabel("Time")
#     plt.ylabel("Packet Length (Bytes)")
#     plt.xlim(min(timestamps), max(timestamps))
#     plt.show()

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



def main():
    all_delays = []
    all_sizes = []
    labels = []
    for pcap_file, group_name in zip(pcap_files, group_names):
        message_timestamps, message_sizes = extract_data_from_pcap(pcap_file)
        delays = compute_inter_message_delays(message_timestamps)

        # print(f"For {group_name}: Timestamps = {message_timestamps}, Sizes = {message_sizes}")

        all_delays.extend(delays)
        all_sizes.extend(message_sizes)

        plot_packet_length_vs_time(message_timestamps, message_sizes, group_name)
        plot_data_for_group(group_name, delays, message_sizes)
        labels.extend([group_name] * len(delays))

    analyze_unique_characteristics(all_delays, all_sizes)
    deduce_group_membership_ML(all_delays, all_sizes, labels)


main()
