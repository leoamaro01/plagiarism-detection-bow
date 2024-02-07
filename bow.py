from math import log, sqrt

def bow(text1: str, text2: str) -> list[float]:
    pgphs1 = get_paragraphs(text1)
    pgphs2 = get_paragraphs(text2)
    
    # gets the unique terms in the whole "corpus"
    terms = get_terms(text1 + ' ' + text2)
    
    # now we create a frequency matrix, where each row is 
    # a frequency vector for a paragraph
    t1feqs = list(map(get_terms_frequencies, pgphs1))
    t2feqs = list(map(get_terms_frequencies, pgphs2))

    bow1 = list(map(get_doc_tf_idf, t1feqs))
    bow2 = list(map(get_doc_tf_idf, t2feqs))
    
    def get_best_similarity(v: list[float]) -> float:
        return max(map(lambda p: cosine_similarity(v, p)
                       , bow2))
    
    return list(map(get_best_similarity, bow1))
    

# returns how similar two vectors are in terms of the angle between them
def cosine_similarity(v1: list[float], v2: list[float]):
    dot_product = sum(map(lambda x, y: x * y, v1, v2))
    norm1 = sqrt(sum(map(lambda x: x*x, v1)))
    norm2 = sqrt(sum(map(lambda x: x*x, v2)))
    
    return dot_product / (norm1 * norm2)


# splits a text by line endings, returning a list of paragraphs
def get_paragraphs(text: str) -> list[str]:
    raw_pgphs = text.splitlines()
    
    pgphs = []
    
    for p in raw_pgphs:
        if p != '' and not p.isspace():
            pgphs.append()
            
    return pgphs


# returns a list of cleaned up words in a text, with repetition
# e.g. "love is love, always love yourself" -> ["love", "is", 
# "love", "always", "love", "yourself"]
def get_clean_words(text: str) -> list[str]:
    return list(map(cleanup_word, text.split()))    


# retrieves a list of the unique terms in a text.
# e.g. "love is love, always love yourself" -> ["love", "is", "always", "yourself"]
def get_terms(text: str) -> list[str]:
    words = get_clean_words()
    
    unique_words = []
    
    for w in words:
        if w not in unique_words:
            unique_words.append(w)
            
    return unique_words


# removes whitespaces and special characters from a word
def cleanup_word(word: str) -> str:
    clean_w = ''
    
    for c in word:
        if c.isalpha():
            clean_w += c
            
    return clean_w


# calculates the tf_idf vector for a document frequencies vector
def get_doc_tf_idf(doc_freqs: list[int], corpus: list[list[int]]) -> list[float]:
    def get_tf(t_idx: int) -> float:
        return (0.5 + 0.5 * (doc_freqs[t_idx] / max(1, max(doc_freqs))))
    
    def get_idf(t_idx: int) -> float:
        docs_with_t: float = 0 # type float to avoid integer division
        for d in corpus:
            if d[t_idx] > 0:
                docs_with_t += 1
        docs_with_t = max(docs_with_t, 1)
        return log(len(corpus) / docs_with_t)
    
    def get_tfidf(t_idx: int) -> float:
        return get_tf(t_idx) * get_idf(t_idx)
    
    return list(map(get_tfidf, range(len(doc_freqs))))
    

# returns a list with the frequency of each term in `terms` found in `text`
def get_terms_frequencies(text: str, terms: list[str]) -> list[int]:
    freqs = []
    
    words = get_clean_words(text)
    
    for t in terms:
        count = 0
        for w in words:
            if w == t:
                count += 1
        freqs.append(count)
    
    return freqs