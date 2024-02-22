import spacy
nlp = spacy.load("en_core_web_sm")

def extract_ner(sentence):
    """
    Extract the NER tag of each token in the sentence.

    sentence (list): The sentence list of objects.
    """
    # Initialize an empty list for feature values.
    features = []

    # Recreate the full sentence and create a spacy doc from it.
    lemmas_in_sentence = ' '.join([token['form'] for token in sentence])
    doc = nlp(lemmas_in_sentence)

    # Only tokens that start a named entity receive a NER tag.
    for token in doc:
        if token.ent_type_ and token.ent_iob_ == 'B':
            features.append(token.ent_type_)
        else:
            features.append('_')

    return features
