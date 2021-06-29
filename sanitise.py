import csv

new_row_list = []

with open("data/^SP500.txt", 'r') as f:
    reader = csv.reader(f, delimiter=';')
    for row in reader:
        new_row_list.append(row)

with open("data/sp500_cleaned.txt", 'w') as f:
    writer = csv.writer(f)

    writer.writerows(new_row_list)



