def my_normalize(data):
    num_rows, num_cols = len(data), len(data[0])
    normalized_data = []

    for col in range(num_cols):
        column = [row[col] for row in data]
        min_val = min(column)
        max_val = max(column)
        normalized_column = [(x - min_val) / (max_val - min_val) for x in column]
        normalized_data.append(normalized_column)

    normalized_data = list(map(list, zip(*normalized_data)))  # Transpose the result
    return normalized_data

def my_standardize(data):
    num_rows, num_cols = len(data), len(data[0])
    standardized_data = []

    for col in range(num_cols):
        column = [row[col] for row in data]
        mean_val = sum(column) / num_rows
        std_dev = (sum((x - mean_val) ** 2 for x in column) / num_rows) ** 0.5
        standardized_column = [(x - mean_val) / std_dev for x in column]
        standardized_data.append(standardized_column)

    standardized_data = list(map(list, zip(*standardized_data)))  # Transpose the result
    return standardized_data

# Ma'lumotlar tuzilishi
data = [[1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]]

# Normalizatsiya
normalized_data = my_normalize(data)

print("Normalizatsiya natijasi:")
for row in normalized_data:
    formatted_row = " ".join("{0:.3f}".format(val) for val in row)
    print(formatted_row)

# Standartlashtirish
standardized_data = my_standardize(data)

print("\nStandartlashtirish natijasi:")
for row in standardized_data:
    formatted_row = " ".join("{0:.3f}".format(val) for val in row)
    print(formatted_row)









