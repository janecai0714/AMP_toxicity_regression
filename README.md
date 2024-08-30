# quantitative prediction of peptide toxicity

## Getting Started

### Python packages

* torch==2.0.1+cu118
* biopython==1.81
* transformers==4.28.1
* tokenizers==0.13.3
### Variable explanation 
* HD = 50% hemolysis value (μM)
* pHD = -log HD, higher pHD means strong toxicity
### Executing program (take EC as an example)
* run /finetune/ProtBERT_finetune.py to build the regression model, result in result.csv
* run /finetune/predict/predict.py to predict 50% hemolysis value of input sequences, input a fasta file and output a csv file.
