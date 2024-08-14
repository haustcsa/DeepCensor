class SentenceSim:
    def __init__(self, **kwargs):
        self.doc1 = kwargs.get('doc1')
        self.doc2 = kwargs.get('doc2')
    def Sentence_Sim(self):
        documents = [self.doc1, self.doc2]
        from sklearn.feature_extraction.text import CountVectorizer
        import pandas as pd

        # 文本向量化
        count_vectorizer = CountVectorizer(stop_words='english')
        sparse_matrix = count_vectorizer.fit_transform(documents)

        # 文本向量化的可视化表格
        doc_term_matrix = sparse_matrix.todense()
        df = pd.DataFrame(doc_term_matrix,
                          columns=count_vectorizer.get_feature_names_out(),
                          index=['doc1', 'doc2'])
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(df)[1][0]





