import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    evidence = []
    labels = []

    month_mapping = {
        "Jan": 0,
        "Feb": 1,
        "Mar": 2,
        "Apr": 3,
        "May": 4,
        "June": 5,
        "Jul": 6,
        "Aug": 7,
        "Sep": 8,
        "Oct": 9,
        "Nov": 10,
        "Dec": 11,
    }
    visitor_mapping = {"Returning_Visitor": 1, "New_Visitor": 0, "Other": 0}

    with open(filename, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            (
                administrative,
                administrative_duration,
                informational,
                informational_duration,
                product_related,
                product_related_duration,
                bounce_rates,
                exit_rates,
                page_values,
                special_day,
                month,
                operating_systems,
                browser,
                region,
                traffic_type,
                visitor_type,
                weekend,
                label,
            ) = row

            evidence_row = [
                int(administrative),
                float(administrative_duration),
                int(informational),
                float(informational_duration),
                int(product_related),
                float(product_related_duration),
                float(bounce_rates),
                float(exit_rates),
                float(page_values),
                float(special_day),
                month_mapping[month],
                int(operating_systems),
                int(browser),
                int(region),
                int(traffic_type),
                visitor_mapping[visitor_type],
                1 if weekend == "TRUE" else 0,
            ]

            label = 1 if label == "TRUE" else 0

            evidence.append(evidence_row)
            labels.append(label)

    return evidence, labels


def train_model(evidence, labels):
    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(evidence, labels)
    return classifier


def evaluate(true_labels, predicted_labels):
    true_positive = true_negative = false_positive = false_negative = 0

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label == 1 and predicted_label == 1:
            true_positive += 1
        elif true_label == 0 and predicted_label == 0:
            true_negative += 1
        elif true_label == 0 and predicted_label == 1:
            false_positive += 1
        else:
            false_negative += 1

    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)

    return sensitivity, specificity


if __name__ == "__main__":
    main()
