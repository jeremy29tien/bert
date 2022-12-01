import json

# Opening JSON file
with open('tmp/output.jsonl') as embedding_file:
    data_list = list(embedding_file)
    # data = json.load(embedding_file)

    # Print the data of dictionary
    print(data_list[0])
