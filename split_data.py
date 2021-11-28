import csv

train_data_path = 'train_dataset.tsv'
memory_data = []

with open(train_data_path, "r", encoding="utf-8") as file:
    lines = file.readlines()
    train_file_len = int(len(lines) * 0.9)
    valid_file_len = int(len(lines) * 0.05)

    for line in lines:
        memory_data.append(line)

train_set = memory_data[:train_file_len]
valid_set = memory_data[train_file_len:train_file_len+valid_file_len]
test_set = memory_data[train_file_len+valid_file_len:]
print(f"Train: {len(train_set)} | Validation: {len(valid_set)} | Test: {len(test_set)}")

with open('train.tsv', "w", encoding="utf-8") as file:
    tsv_writer = csv.writer(file, delimiter='\t')
    for line in train_set:
        query, response, search_idx, label = line.split('\t')
        tsv_writer.writerow([query.strip(), response.strip(), search_idx.strip(), label.strip()])

with open('valid.tsv', "w", encoding="utf-8") as file:
    tsv_writer = csv.writer(file, delimiter='\t')
    for line in valid_set:
        query, response, search_idx, label = line.split('\t')
        tsv_writer.writerow([query.strip(), response.strip(), search_idx.strip(), label.strip()])

with open('test.tsv', "w", encoding="utf-8") as file:
    tsv_writer = csv.writer(file, delimiter='\t')
    for line in test_set:
        query, response, search_idx, label = line.split('\t')
        tsv_writer.writerow([query.strip(), response.strip(), search_idx.strip(), label.strip()])