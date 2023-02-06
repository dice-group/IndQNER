# IndoBERT Fine-tuning
This is the second setting of IndQNER evaluation that is fine-tuning an existing Indonesian pre-trained language model, IndoBERT, using the IndQNER dataset.
Among two version of IndoBERTs, we utilize this [IndoBERT](https://huggingface.co/indobenchmark/indobert-base-p1).

## How to do the fine-tuning
IndoBERT fine-tuning can be performed by executing the scripts in **_indqner_evaluation_using_indobert_finetuning.py_** or **_IndQNER_Evaluation_using_IndoBERT_FineTuning.ipynb_**. 
The latter is executed using a web-based interactive computing platform provided by Google Research i.e. Google Colab. 
If you want to use a Jupyter Notebook, you can simply remove the first two block sections in the file.
The scripts used for IndoBERT fine-tuning are from https://github.com/IndoNLP/indonlu.
