import json

# Opening JSON file
with open('tmp/output.jsonl') as embedding_file:
    data_list = list(embedding_file)
    # data = json.load(embedding_file)

    data = []
    for data_string in data_list:
        result = json.loads(data_string)
        data.append(result)
        print(f"result: {result}")
        print(isinstance(result, dict))

    print(data[0]['features'])

