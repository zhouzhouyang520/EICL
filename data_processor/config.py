import torch
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/ED")
parser.add_argument("--data_name", type=str, default="dataset_preproc.p")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--example_num", type=int, default=5)
parser.add_argument("--embed_name", type=str, default="dataset_emb.p")
parser.add_argument("--example_name", type=str, default="example_index.p")
parser.add_argument("--ed_dir", type=str, default="ed_json_data")
parser.add_argument("--ed_tst", type=str, default="ed_tst.json")
parser.add_argument("--emb_model", type=str, default="semantic")
parser.add_argument("--emo_prob_name", type=str, default="emo_prob.json")


parser.add_argument("--cuda", default=False, action="store_true")
parser.add_argument('--device_id', dest='device_id', type=str, default="0", help='gpu device id')

def print_opts(opts):
    """Prints the values of all command-line arguments."""
    print("=" * 80) 
    print("Opts".center(80))
    print("-" * 80) 
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print("{:>30}: {:<30}".format(key, opts.__dict__[key]).center(80))
    print("=" * 80) 


args = parser.parse_args()
print_opts(args)

use_gpu = args.cuda
device = torch.device("cuda" if args.cuda else "cpu")
device_id = args.device_id
data_dir = args.data_dir #+ "/" + args.data_name
batch_size = args.batch_size
embed_name = args.embed_name
example_name = args.example_name 
ed_dir = args.ed_dir
ed_tst = args.ed_tst
example_num = args.example_num 
emb_model = args.emb_model
emo_prob_name = args.emo_prob_name
embed_name = embed_name if emb_model == "semantic" else "dataset_emo_emb.p"
example_name = example_name if emb_model == "semantic" else "example_emo_index.p"
print(f"emb_model: {emb_model}, embed_name: {embed_name}, example_name: {example_name}")
