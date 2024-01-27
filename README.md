# IndQNER
IndQNER is a Named Entity Recognition (NER) benchmark dataset that was created by manually annotating 8 chapters in the Indonesian translation of the Quran. The annotation was performed using a web-based text annotation tool, [Tagtog](https://www.tagtog.com/), and the BIO (Beginning-Inside-Outside) tagging format. The dataset contains:
* 3117 sentences
* 62027 tokens
* 2475 named entities
* 18 named entity categories

## Named Entity Classes
The named entity classes were initially defined by analyzing the existing Quran concepts ontology. The initial classes were updated based on the information acquired during the annotation process. Finally, there are 20 classes, as follows:
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
There were eight annotators who contributed to the annotation process. They were informatics engineering students at the State Islamic University Syarif Hidayatullah Jakarta. 
1. Anggita Maharani Gumay Putri
2. Muhammad Destamal Junas
3. Naufaldi Hafidhigbal
4. Nur Kholis Azzam Ubaidillah
5. Puspitasari
6. Septiany Nur Anggita
7. Wilda Nurjannah
8. William Santoso

## Verification Stage
We found many named entity and class candidates during the annotation stage. To verify the candidates, we consulted Quran and Tafseer (content) experts who were lecturers at the Quran and Tafseer Department, the State Islamic University Syarif Hidayatullah Jakarta.
1. Dr. Lilik Ummi Kultsum, MA
2. Dr. Jauhar Azizy, MA
3. Dr. Eva Nugraha, M.Ag.

## Evaluation
We evaluated the annotation quality of IndQNER by performing experiments in two settings: supervised learning (BiLSTM+CRF) and transfer learning ([IndoBERT](https://huggingface.co/indobenchmark/indobert-base-p1) fine-tuning). 

### Supervised Learning Setting
The implementation of BiLSTM and CRF utilized [IndoBERT](https://huggingface.co/indobenchmark/indobert-base-p1) to provide word embeddings. All experiments used a batch size of 16. These are the results:

|Maximum sequence length|Number of e-poch|Precision|Recall|F1 score|
|-----------------------|----------------|---------|------|--------|
|         256		|       10	 |   0.94  | 0.92 |  0.93  |
|         256		|       20	 |   0.99  | 0.97 |  0.98  |
|         256		|       40	 |   0.96  | 0.96 |  0.96  |
|         256		|       100	 |   0.97  | 0.96 |  0.96  |
|         512		|	10	 |   0.92  | 0.92 |  0.92  |
|	  512		| 	20	 |   0.96  | 0.95 |  0.96  |
|	  512 		|       40       |   0.97  | 0.95 |  0.96  |
|	  512 		|       100      |   0.97  | 0.95 |  0.96  |

### Transfer Learning Setting
We performed several experiments with different parameters in IndoBERT fine-tuning. All experiments used a learning rate of 2e-5 and a batch size of 16. These are the results:

|Maximum sequence length|Number of e-poch|Precision|Recall|F1 score|
|-----------------------|----------------|---------|------|--------|
|	  256		|	10	 |   0.67  | 0.65 |  0.65  |
|	  256 		|	20	 |   0.60  | 0.59 |  0.59  |
|	  256		|	40	 |   0.75  | 0.72 |  0.71  |
|	  256		|	100	 |   0.73  | 0.68 |  0.68  |
|	  512		|	10	 |   0.72  | 0.62 |  0.64  |
|	  512		|	20	 |   0.62  | 0.57 |  0.58  |
|	  512		|	40	 |   0.72  | 0.66 |  0.67  |
|	  512		|	100	 |   0.68  | 0.68 |  0.67  |

This dataset is also a part of [NusaCrowd project](https://github.com/IndoNLP/nusa-crowd) that aims to collect Natural Language Processing (NLP) datasets for Indonesian and its local languages.

## How to Cite
```bibtex
@InProceedings{10.1007/978-3-031-35320-8_12,
author="Gusmita, Ria Hari
and Firmansyah, Asep Fajar
and Moussallem, Diego
and Ngonga Ngomo, Axel-Cyrille",
editor="M{\'e}tais, Elisabeth
and Meziane, Farid
and Sugumaran, Vijayan
and Manning, Warren
and Reiff-Marganiec, Stephan",
title="IndQNER: Named Entity Recognition Benchmark Dataset from the Indonesian Translation of the Quran",
booktitle="Natural Language Processing and Information Systems",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="170--185",
abstract="Indonesian is classified as underrepresented in the Natural Language Processing (NLP) field, despite being the tenth most spoken language in the world with 198 million speakers. The paucity of datasets is recognized as the main reason for the slow advancements in NLP research for underrepresented languages. Significant attempts were made in 2020 to address this drawback for Indonesian. The Indonesian Natural Language Understanding (IndoNLU) benchmark was introduced alongside IndoBERT pre-trained language model. The second benchmark, Indonesian Language Evaluation Montage (IndoLEM), was presented in the same year. These benchmarks support several tasks, including Named Entity Recognition (NER). However, all NER datasets are in the public domain and do not contain domain-specific datasets. To alleviate this drawback, we introduce IndQNER, a manually annotated NER benchmark dataset in the religious domain that adheres to a meticulously designed annotation guideline. Since Indonesia has the world's largest Muslim population, we build the dataset from the Indonesian translation of the Quran. The dataset includes 2475 named entities representing 18 different classes. To assess the annotation quality of IndQNER, we perform experiments with BiLSTM and CRF-based NER, as well as IndoBERT fine-tuning. The results reveal that the first model outperforms the second model achieving 0.98 F1 points. This outcome indicates that IndQNER may be an acceptable evaluation metric for Indonesian NER tasks in the aforementioned domain, widening the research's domain range.",
isbn="978-3-031-35320-8"
}
```

## Contact
If you have any questions or feedbacks, feel free to contact us at ria.hari.gusmita@uni-paderborn.de or ria.gusmita@uinjkt.ac.id
