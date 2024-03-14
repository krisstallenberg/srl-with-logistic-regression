def extract_predicate(sentence):

    for row in sentence:
        if row['predicate'] != '_':
            return [row['predicate'] for word in sentence]
