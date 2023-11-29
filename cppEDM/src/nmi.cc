#include <iostream>
#include <cmath>
#include <valarray>
#include <map>

using namespace std;

double calculate_entropy(const valarray<double>& values) {
    map<double, int> value_counts;

    // Count occurrences of each value
    for (double value : values) {
        value_counts[value]++;
    }

    double entropy = 0.0;
    int total_values = values.size();

    for (const auto& pair : value_counts) {
        double probability = static_cast<double>(pair.second) / total_values;
        entropy -= probability * log2(probability);
    }

    return entropy;
}

double calculate_mutual_information(const valarray<double>& x, const valarray<double>& y) {
    if (x.size() != y.size()) {
        cerr << "Error: Input vectors must have the same size." << endl;
        return 0.0;
    }

    int total_values = x.size();

    // Create a map to count joint occurrences of x and y values
    map<pair<double, double>, int> joint_counts;

    for (int i = 0; i < total_values; ++i) {
        joint_counts[{x[i], y[i]}]++;
    }

    double mutual_information = 0.0;

    for (const auto& pair : joint_counts) {
        double p_xy = static_cast<double>(pair.second) / total_values;
        double p_x = count(begin(x), end(x), pair.first.first) / static_cast<double>(total_values);
        double p_y = count(begin(y), end(y), pair.first.second) / static_cast<double>(total_values);

        mutual_information += p_xy * log2(p_xy / (p_x * p_y));
    }

    return mutual_information;
}

double calculate_normalized_mutual_information(const valarray<double>& x, const valarray<double>& y) {
    double mutual_information = calculate_mutual_information(x, y);
    double entropy_x = calculate_entropy(x);
    double entropy_y = calculate_entropy(y);

    // Normalize mutual information
    double normalized_mutual_information = 2 * mutual_information / (entropy_x + entropy_y);

    return normalized_mutual_information;
}

/*int main() {
    // Example usage
    valarray<double> x = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    valarray<double> y = {1.0, 2.0, 2.0, 1.0, 2.0, 1.0};

    double nmi = calculate_normalized_mutual_information(x, y);

    cout << "Normalized Mutual Information: " << nmi << endl;

    return 0;
}*/
