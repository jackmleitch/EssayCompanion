import nltk
from keybert import KeyBERT
from nltk.corpus import wordnet 
from typing import List, Dict

# download required nltk packages
for download in ['wordnet', 'omw-1.4']:
    try:
        nltk_path = nltk.find(f"corpora/{download}")
    except Exception:
        nltk.download(f"{download}")

class GetKeywords():
    """Extract keywords and find synonyms for them"""

    def __init__(self) -> None:
        self.model = KeyBERT()

    def get_keywords(self, query: str) -> List[str]:
        """Get keywords from query text using KeyBERT"""
        keywords = self.model.extract_keywords(query)
        return [word[0] for word in keywords]

    def get_synonyms_for_keyword(self, keyword: str, max_synonyms=5) -> List[str]:
        """Find synonyms for a given word"""
        synonyms = []
        for synonym in wordnet.synsets(keyword):
            for lemma in synonym.lemmas():
                synonyms.append(lemma.name().replace("_", " "))
        return [synonym for synonym in list(set(synonyms)) 
                    if synonym.lower() != keyword.lower()][:max_synonyms] 

    def get_synonyms_for_keywords(self, query: str, max_synonyms=5) -> Dict[str, List]:
        """Find synonyms for all keywords and return them"""
        keywords = self.get_keywords(query)
        keyword_synonyms = {}
        for keyword in keywords:
            synonyms = self.get_synonyms_for_keyword(keyword, max_synonyms)
            if len(synonyms) > 0: 
                keyword_synonyms[keyword] = synonyms
        return keyword_synonyms

if __name__ == '__main__':
    print("Loading KeyBERT model...")
    pipeline = GetKeywords()
    print('Model loaded!')
    query = """
        The ultimate test of your knowledge is your capacity to convey it to another. 
        The best way of learning is by teaching & practicing. D. Mermin once described a
        person who has a good understanding of physics but can't convey it, as being 
        a natural phenomenon, not a physicist. 
        """
    print(pipeline.get_synonyms_for_keywords(query))


