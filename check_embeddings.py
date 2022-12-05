import json
import numpy as np
from scipy import spatial

print("---SINGLE WORD EMBEDDINGS---")
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

    # print("king_tok_embedding:", king_tok_embedding)
    # print("queen_tok_embedding:", queen_tok_embedding)
    # print("prince_tok_embedding:", prince_tok_embedding)
    # print("chair_tok_embedding:", chair_tok_embedding)
    # print("guitar_tok_embedding:", guitar_tok_embedding)

    print("EUCLIDEAN DISTANCE (L2 norm):")
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

    print("COSINE DISTANCE (1 - cosine_similarity):")
    print("king - queen = ", spatial.distance.cosine(king_tok_embedding, queen_tok_embedding))
    print("king - prince = ", spatial.distance.cosine(king_tok_embedding, prince_tok_embedding))
    print("queen - prince = ", spatial.distance.cosine(queen_tok_embedding, prince_tok_embedding))
    print("king - chair = ", spatial.distance.cosine(king_tok_embedding, chair_tok_embedding))
    print("queen - chair = ", spatial.distance.cosine(queen_tok_embedding, chair_tok_embedding))
    print("prince - chair = ", spatial.distance.cosine(prince_tok_embedding, chair_tok_embedding))
    print("king - guitar = ", spatial.distance.cosine(king_tok_embedding, guitar_tok_embedding))
    print("queen - guitar = ", spatial.distance.cosine(queen_tok_embedding, guitar_tok_embedding))
    print("prince - guitar = ", spatial.distance.cosine(prince_tok_embedding, guitar_tok_embedding))
    print("chair - guitar = ", spatial.distance.cosine(chair_tok_embedding, guitar_tok_embedding))

print("\n---SENTENCE EMBEDDINGS---")
with open('tmp/sentence_output.jsonl') as embedding_file:
    data_list = list(embedding_file)
    data = []
    for data_string in data_list:
        result = json.loads(data_string)
        data.append(result)

    # data has embeddings of sentences
    # RABBIT VS. TURTLE
    slightly_faster_words = data[0]['features']
    slightly_faster_embedding = []
    for word_embedding in slightly_faster_words:
        slightly_faster_embedding.append(word_embedding['layers'][0]['values'])
    slightly_faster_embedding = np.mean(np.asarray(slightly_faster_embedding), axis=0)

    faster_words = data[1]['features']
    faster_embedding = []
    for word_embedding in faster_words:
        faster_embedding.append(word_embedding['layers'][0]['values'])
    faster_embedding = np.mean(np.asarray(faster_embedding), axis=0)

    much_faster_words = data[2]['features']
    much_faster_embedding = []
    for word_embedding in much_faster_words:
        much_faster_embedding.append(word_embedding['layers'][0]['values'])
    much_faster_embedding = np.mean(np.asarray(much_faster_embedding), axis=0)

    alittle_faster_words = data[3]['features']
    alittle_faster_embedding = []
    for word_embedding in alittle_faster_words:
        alittle_faster_embedding.append(word_embedding['layers'][0]['values'])
    alittle_faster_embedding = np.mean(np.asarray(alittle_faster_embedding), axis=0)

    # PLANE VS. CAR
    slightly_larger_words = data[4]['features']
    slightly_larger_embedding = []
    for word_embedding in slightly_larger_words:
        slightly_larger_embedding.append(word_embedding['layers'][0]['values'])
    slightly_larger_embedding = np.mean(np.asarray(slightly_larger_embedding), axis=0)

    larger_words = data[5]['features']
    larger_embedding = []
    for word_embedding in larger_words:
        larger_embedding.append(word_embedding['layers'][0]['values'])
    larger_embedding = np.mean(np.asarray(larger_embedding), axis=0)

    much_larger_words = data[6]['features']
    much_larger_embedding = []
    for word_embedding in much_larger_words:
        much_larger_embedding.append(word_embedding['layers'][0]['values'])
    much_larger_embedding = np.mean(np.asarray(much_larger_embedding), axis=0)

    alittle_larger_words = data[7]['features']
    alittle_larger_embedding = []
    for word_embedding in alittle_larger_words:
        alittle_larger_embedding.append(word_embedding['layers'][0]['values'])
    alittle_larger_embedding = np.mean(np.asarray(alittle_larger_embedding), axis=0)

    print("EUCLIDEAN DISTANCE (L2 norm):")
    print("RABBIT VS. TURTLE")
    print("(slightly faster) - (faster) = ", np.linalg.norm(slightly_faster_embedding - faster_embedding))
    print("(much faster) - (faster) = ", np.linalg.norm(much_faster_embedding - faster_embedding))
    print("(slightly faster) - (much faster) = ", np.linalg.norm(slightly_faster_embedding - much_faster_embedding))
    print("(a little faster) - (faster) = ", np.linalg.norm(alittle_faster_embedding - faster_embedding))
    print("(slightly faster) - (a little faster) = ", np.linalg.norm(slightly_faster_embedding - alittle_faster_embedding))
    print("PLANE VS. CAR")
    print("(slightly larger) - (larger) = ", np.linalg.norm(slightly_larger_embedding - larger_embedding))
    print("(much larger) - (larger) = ", np.linalg.norm(much_larger_embedding - larger_embedding))
    print("(slightly larger) - (much larger) = ", np.linalg.norm(slightly_larger_embedding - much_larger_embedding))
    print("(a little larger) - (larger) = ", np.linalg.norm(alittle_larger_embedding - larger_embedding))
    print("(slightly larger) - (a little larger) = ", np.linalg.norm(slightly_larger_embedding - alittle_larger_embedding))

    print("COSINE DISTANCE (1 - cosine_similarity):")
    print("RABBIT VS. TURTLE")
    print("(slightly faster) - (faster) = ", spatial.distance.cosine(slightly_faster_embedding, faster_embedding))
    print("(much faster) - (faster) = ", spatial.distance.cosine(much_faster_embedding, faster_embedding))
    print("(slightly faster) - (much faster) = ", spatial.distance.cosine(slightly_faster_embedding, much_faster_embedding))
    print("(a little faster) - (faster) = ", spatial.distance.cosine(alittle_faster_embedding, faster_embedding))
    print("(slightly faster) - (a little faster) = ", spatial.distance.cosine(slightly_faster_embedding, alittle_faster_embedding))
    print("PLANE VS. CAR")
    print("(slightly larger) - (larger) = ", spatial.distance.cosine(slightly_larger_embedding, larger_embedding))
    print("(much larger) - (larger) = ", spatial.distance.cosine(much_larger_embedding, larger_embedding))
    print("(slightly larger) - (much larger) = ", spatial.distance.cosine(slightly_larger_embedding, much_larger_embedding))
    print("(a little larger) - (larger) = ", spatial.distance.cosine(alittle_larger_embedding, larger_embedding))
    print("(slightly larger) - (a little larger) = ", spatial.distance.cosine(slightly_larger_embedding, alittle_larger_embedding))

    print("\n---COMPARING DIFFERENCE VECTORS---")
    print("(much faster) - (faster) vs. (much larger) - (larger)")
    # print("(much faster) - (faster) =", much_faster_embedding - faster_embedding)
    print("||(much faster) - (faster)||_2 =", np.linalg.norm(much_faster_embedding - faster_embedding))
    # print("(much larger) - (larger) =", much_larger_embedding - larger_embedding)
    print("||(much larger) - (larger)||_2 =", np.linalg.norm(much_larger_embedding - larger_embedding))
    print("Cosine similarity:", 1-spatial.distance.cosine(much_faster_embedding-faster_embedding, much_larger_embedding-larger_embedding))
    print("---")
    print("(slightly faster) - (faster) vs. (slightly larger) - (larger)")
    print("||(slightly faster) - (faster)||_2 =", np.linalg.norm(slightly_faster_embedding - faster_embedding))
    print("||(slightly larger) - (larger)||_2 =", np.linalg.norm(slightly_larger_embedding - larger_embedding))
    print("Cosine similarity:", 1-spatial.distance.cosine(slightly_faster_embedding-faster_embedding, slightly_larger_embedding-larger_embedding))
    print("---")
    print("(slightly faster) - (much faster) vs. (slightly larger) - (much larger)")
    print("||(slightly faster) - (much faster)||_2 =", np.linalg.norm(slightly_faster_embedding - much_faster_embedding))
    print("||(slightly larger) - (much larger)||_2 =", np.linalg.norm(slightly_larger_embedding - much_larger_embedding))
    print("Cosine similarity:", 1-spatial.distance.cosine(slightly_faster_embedding-much_faster_embedding, slightly_larger_embedding-much_larger_embedding))


