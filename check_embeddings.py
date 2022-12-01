import json
import numpy as np

# Opening JSON file
with open('tmp/output.jsonl') as embedding_file:
    data_list = list(embedding_file)
    # data = json.load(embedding_file)

    data = []
    for data_string in data_list:
        result = json.loads(data_string)
        data.append(result)
        # print(f"result: {result}")
        # print(isinstance(result, dict))

    # The beginning of each string has a '[CLS]' token appended. There may also be '[SEP]' tokens.
    # print(data[0]['features'][1]['layers'][0]['values'])
    first_line = data[0]['features']
    king_tok_layers = first_line[1]['layers']
    king_tok_embedding = king_tok_layers[0]['values']  # We are taking the embeddings from the last Transformer layer.

    queen_tok_embedding = data[1]['features'][1]['layers'][0]['values']
    prince_tok_embedding = data[2]['features'][1]['layers'][0]['values']
    chair_tok_embedding = data[3]['features'][1]['layers'][0]['values']
    guitar_tok_embedding = data[4]['features'][1]['layers'][0]['values']

    king_tok_embedding = np.asarray(king_tok_embedding)
    queen_tok_embedding = np.asarray(queen_tok_embedding)
    prince_tok_embedding = np.asarray(prince_tok_embedding)
    chair_tok_embedding = np.asarray(chair_tok_embedding)
    guitar_tok_embedding = np.asarray(guitar_tok_embedding)

    print("king_tok_embedding:", king_tok_embedding)
    print("queen_tok_embedding:", queen_tok_embedding)
    print("prince_tok_embedding:", prince_tok_embedding)
    print("chair_tok_embedding:", chair_tok_embedding)
    print("guitar_tok_embedding:", guitar_tok_embedding)

    print("king - queen = ", np.linalg.norm(king_tok_embedding - queen_tok_embedding))
    print("king - prince = ", np.linalg.norm(king_tok_embedding - prince_tok_embedding))
    print("queen - prince = ", np.linalg.norm(queen_tok_embedding - prince_tok_embedding))
    print("king - chair = ", np.linalg.norm(king_tok_embedding - chair_tok_embedding))
    print("queen - chair = ", np.linalg.norm(queen_tok_embedding - chair_tok_embedding))
    print("prince - chair = ", np.linalg.norm(prince_tok_embedding - chair_tok_embedding))
    print("king - guitar = ", np.linalg.norm(king_tok_embedding - guitar_tok_embedding))
    print("queen - guitar = ", np.linalg.norm(queen_tok_embedding - guitar_tok_embedding))
    print("prince - guitar = ", np.linalg.norm(prince_tok_embedding - guitar_tok_embedding))
    print("chair - guitar = ", np.linalg.norm(chair_tok_embedding - guitar_tok_embedding))


