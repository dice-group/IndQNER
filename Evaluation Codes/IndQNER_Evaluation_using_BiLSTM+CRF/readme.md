# Indonesian NER using BiLSTM+CRF+IndoBERT
This model is used to predict named entities in Indonesian text utilizing the BiLSTM+CRF techniques, and IndoBERT. The model is built by modifying an existing NER: https://github.com/PeijiYang/BERT-BiLSTM-CRF-NER-pytorch. IndoBERT is used to provide word embeddings. It is downloaded from https://huggingface.co/indobenchmark/indobert-base-p1. 
  
## Datasets
The model uses datasets splitted into training, validation, and test sets in 8:1:1 ratio. The datasets use BIO (Beginning-Inside-Outside) tagging format.
The first column refers to words and the second column represents named entity classes. These are some examples:

Jika	O
kamu	O
(	O
tetap	O
)	O
dalam	O
keraguan	O
tentang	O
apa	O
(	O
Al-Qur’an	B-HolyBook
)	O
yang	O
Kami	O
turunkan	O
kepada	O
hamba	O
Kami	O
(	O
Nabi	O
Muhammad	B-Messenger
)	O
,	O
buatlah	O
satu	O
surah	O
yang	O
semisal	O
dengannya	O
dan	O
ajaklah	O
penolong-penolongmu	O
selain	O
Allah	B-Allah
,	O
jika	O
kamu	O
orang-orang	O
yang	O
benar	O
.	O

## How to run
Training and test steps can be performed at the same time using this syntax:
python ner.py --do_train True     --do_eval True     --do_test True     --max_seq_length 25  --train_batch_size 8     --eval_batch_size 8     --num_train_epochs 10     --do_lower_case     --logging_steps 200     --need_birnn True     --rnn_dim 256     --clean True
