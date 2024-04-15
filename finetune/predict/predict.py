from model_def import REG
from seq_dataloader import *
from seq_dataloader import _get_train_data_loader, _get_test_data_loader, freeze
import collections
from onedrivedownloader import download
import os
import warnings
warnings.filterwarnings('ignore')

def read_fasta_file(fasta_path, csv_path):
    f = open(fasta_path, "r")
    seq = collections.OrderedDict()
    for line in f:
        if line.startswith(">"):
            name = line.split()[0]
            seq[name] = ''
        else:
            seq[name] += line.replace("\n", '').strip()
    f.close()
    seq_df = pd.DataFrame(seq.items(), columns=['ID', 'SEQUENCE'])
    seq_df["SEQUENCE_space"] = [" ".join(ele) for ele in seq_df["SEQUENCE"]]
    seq_df.to_csv(csv_path)
    return seq_df
def predict(model, fasta_path, csv_path):
    batch_size = 500
    seq = read_fasta_file(fasta_path, csv_path)
    test_loader = _get_test_data_loader(batch_size, csv_path)
    predict_list= []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch['input_ids']
            b_input_mask = batch['attention_mask']
            predict_pHD50, _ = model(b_input_ids, attention_mask=b_input_mask)
            predict_list.extend(predict_pHD50.data.numpy())

    predict_list = [item for sublist in predict_list for item in sublist]
    seq["predicted_pHD50"] = predict_list
    seq["predicted_HD50_μM"] = [10**(-item) for item in predict_list]
    seq = seq.drop(columns = ["SEQUENCE_space"])
    seq.to_csv(csv_path, index=False)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_url = "https://ipmedumo-my.sharepoint.com/:u:/g/personal/p2214906_mpu_edu_mo/EfNaaXTnusBFtvwJlnHbdZUBQRVyvZ0XeTjRVn1269DHMA?e=PYkSTd"
    model_path = "prot_bert_finetune_toxicity.pkl"
    if not os.path.exists(model_path):
        download(model_url, model_path)
    model = REG()
    model.load_state_dict(torch.load(model_path))

    fasta_path = "test.fasta"
    csv_path = "test_toxicity.csv"
    predict(model, fasta_path, csv_path)
    print('smart')
