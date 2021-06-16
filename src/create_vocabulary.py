import argparse
import pickle

import text_utils

def main(args):
    import coco_captions_dataset
    corpus = coco_captions_dataset.captions(args.data_file)

    corpus = map(lambda x: x.lower(), corpus) 
    corpus = map(text_utils.tokenize, corpus)

    vocabulary = text_utils.fit_vocabulary(corpus, 
        min_count=args.min_count)
    print(len(vocabulary), "tokens in the vocabulary.")

    test_doc = [
            '<pad>', '<unk>', '<start>', '<end>', 'it', 'is', 'a',
            'non-existent-word-in-the-vocabulary']
    print()
    print("Test document:", test_doc)
    print("Corresponding indices:", vocabulary.doc2idx(test_doc))

    with open(args.save_file, 'wb') as f:
        pickle.dump(vocabulary, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('data_file', type=str)
    parser.add_argument('save_file', type=str)
    parser.add_argument('--min-count', type=int, default=4)
    args = parser.parse_args()

    main(args)
