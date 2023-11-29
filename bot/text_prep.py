import nltk
from natasha import Doc, Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger
from nltk.corpus import stopwords
nltk.download("stopwords")

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

stop_words = stopwords.words('russian')
stop_words.extend(['что', 'это', 'так',
                    'вот', 'быть', 'как',
                    'в', '—', 'к', 'за', 'из', 'из-за',
                    'на', 'ок', 'кстати',
                    'который', 'мочь', 'весь',
                    'еще', 'также', 'свой',
                    'ещё', 'самый', 'ул', 'комментарий',
                    'английский', 'язык', 'Наш', 'наш',
                    'наш проект', 'которых', 'которые',
                    'проект', 'которым', 'Наш проект',
                    'Санкт-Петербург', 'Санкт-Петербурга',
                    'Санкт-Петербургу', 'Санкт-Петербургом',
                    'Санкт-Петербурге', 'Петербург',
                    'Петербурга', 'Петербургу', 'Петербургом',
                    'Петербурге', 'Питер', 'Питера', 'Питеру',
                    'Питером', 'Питере'])


def text_prep(text) -> str:
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    for token in doc.tokens:
        token.lemmatize(morph_vocab)

    lemmas = [_.lemma for _ in doc.tokens]
    words = [lemma for lemma in lemmas if lemma.isalpha() and len(lemma) > 2]
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)
