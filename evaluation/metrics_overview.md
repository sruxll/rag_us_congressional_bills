Process for Evaluation

---

### **Evaluating Semantic Relationship**

**[Bertscorer](https://pypi.org/project/bert-score/)** 

*  Compares word embeddings using BERT,
*  Measures similarity by aligning words in the generated summary to the source material, takes context into account.
* Rewards semantically similar words (synonyms) and penalizes mismatches.
* More sensitive to word meaning and placement.


**Score**:
*   range: 0 to 1
         0.85 or higher ~ strong similarity
         0.75 or higher  ~ significant
         lower than 0.70, ~less meaningfull/ weak similarity



**[Cosign Similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html#cosine-similarity)**

*   Compares sentence level embeddings, vector representatio of the whole sentence.
*   Checks the overall semantic meaning, not just for word for word matching like BertScorer
* More forgiving to text variations as long as the core meaning is the same.

Formula:
  Precision = LCS length / number of words in the generated summary

  Recall = LCS length / number of words in the reference summary

  F1 Score combines both to balance precision and recall.

**Score**:
* range: -1 to 1
         > 0.8 ~ very similar
         0.6 < ~ significant
      
---


### **Evaluating Lexical Overlap**
**[Rouge_L](https://pypi.org/project/bert-score/)**


*   Measures longest common subsequence of words (LCS) between the generated summary and source documents.
*   Focuses on word order, the longer the sequence of words in the correct order the higher the score.
* 


*   Precision: Number of words in LCS / Number of words in the generated output
*   Recall: Number of words in LCS / Number of words in the reference summary

**Score**:

* range: 0 to 1
           > 0.4 ~ strong word and sequence overlap
              0.2 < ~ significant
---


### **Evaluating Relavance**

**[Named based Entities(Spacy)](https://spacy.io/api/entityrecognizer/)**


*  Extracts named entities (e.g., people, organizations, locations, dates, bill names).

* Measures if key factual elements from source document are in the generated summary, ensuring factual accuracy.

**Score**:
* range: 0 to 1
       > 0.5 ~ strong factual consistency
       < 0.3 ~ possibly hallucinated or incomplete summary

