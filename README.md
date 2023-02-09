# IndQNER
IndQNER is a Named Entity Recognition (NER) benchmark dataset that was created by manually annotating 8 chapters in the Indonesian translation of the Quran. 
The annotation was performed using a web-based text annotation tool, [Tagtog](https://www.tagtog.com/), and the BIO (Beginning-Inside-Outside) tagging format. 
The dataset contains:
* 3114 sentences
* 48689 tokens
* 2476 named entities
* 18 named entity categories

## Named Entity Classes
The named entity classes were initially defined by analyzing [the existing Quran concepts ontology](https://corpus.quran.com/concept.jsp).
The initial classes were updated based on the information acquired during the annotation process. Finally, there are 20 classes as follows:
1. Allah
2. Allah's Throne
3. Artifact
4. Astronomical body
5. Event
6. False deity
7. Holy book
8. Language
9. Angel
10. Person
11. Messenger
12. Prophet
13. Sentient
14. Afterlife location
15. Geographical location
16. Color
17. Religion
18. Food
19. Fruit
20. The book of Allah

## Annotation Stage
There were eight annotators who contributed to the annotation process. They are Informatics Engineering students at the State Islamic University Syarif Hidayatullah Jakarta. 
1. Anggita Maharani Gumay Putri
2. Muhammad Destamal Junas
3. Naufaldi Hafidhigbal
4. Nur Kholis Azzam Ubaidillah
5. Puspitasari
6. Septiany Nur Anggita
7. Wilda Nurjannah
8. William Santoso

## Verification Stage
We found many named entity and class candidates during the annotation stage. To verify the candidates, we consulted Quran and Tafseer (content) experts who are lecturers at Quran and Tafseer Department, the State Islamic University Syarif Hidayatullah Jakarta.
1. Dr. Lilik Ummi Kultsum, MA
2. Dr. Jauhar Azizy, MA
3. Dr. Eva Nugraha, M.Ag.

## Evaluation
We evaluated the annotation quality of IndQNER by performing experiments in two settings: supervised learning (Bi-LSTM+CRF) and transfer learning ([IndoBERT](https://huggingface.co/indobenchmark/indobert-base-p1) fine-tuning). 

### Supervised Learning Setting
The implementation of Bi-LSTM and CRF utilized [IndoBERT](https://huggingface.co/indobenchmark/indobert-base-p1) to provide word embeddings. All experiments used batch size of 8 (the first experiment) and 16 (the remaining experiments). These are the results:
|Maximum sequence length|Number of e-poch|Precision|Recall|F1 score|
|-----------------------|----------------|---------|------|--------|
|25|10|0.96|0.93|0.94|
|256|10|0.94|0.92|0.93|
|256|20|0.99|0.97|0.98|
|256|40|0.96|0.96|0.96|
|512|10|0.92|0.92|0.92|
|512|20|0.96|0.95|0.96|
|512|40|0.97|0.95|0.96|

### Transfer Learning Setting
We performed several experiments with different parameters in IndoBERT fine-tuning. All experiments used learning rate of 2e-5 and batch size of 16. These are the results:
|Maximum sequence length|Number of e-poch|Precision|Recall|F1 score|
|-----------------------|----------------|---------|------|--------|
|256|10|0.67|0.65|0.65|
|256|20|0.60|0.59|0.59|
|256|40|0.75|0.72|0.71|
|256|100|0.73|0.68|0.68|
|512|10|0.72|0.62|0.64|
|512|20|0.62|0.57|0.58|
|512|40|0.72|0.66|0.67|
|512|100|0.68|0.68|0.67|

This dataset is also a part of [NusaCrowd project](https://github.com/IndoNLP/nusa-crowd) that aims to collect Natural Language Processing (NLP) datasets for the Indonesian languages.

## Contact
If you have any questions or feedbacks, feel free to contact us at ria.hari.gusmita@uni-paderborn.de or ria.gusmita@uinjkt.ac.id
