with open('all_test_sequences.json', 'r') as in_file:
    with open('all_test_sequences_formatted_correctly.json', 'w') as out_file:
        for line in in_file.readlines():
            out_file.write(f'{line.strip()},\n')