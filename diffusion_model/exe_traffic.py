import argparse
import torch
import datetime
import json
import yaml
import os
import time
from main_model import OpenDiff
from dataset_physio_4traffic_new import get_dataloader
from utils import train, evaluate


parser = argparse.ArgumentParser(description="Opendiff")
parser.add_argument("--traindata", type=str, default='shanghai')
parser.add_argument("--condition_set", type=str, default='csdi')
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=1)

args = parser.parse_args()
print(args)



start_time =  time.time()

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/"+ args.traindata +'_' +args.condition_set + '_' + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
)


data_process_time =  time.time()

data_process_used = data_process_time-start_time

model = OpenDiff(config, args.device).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))


train_time = time.time()
train_used = train_time - data_process_time

evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)

infer_time = time.time()

infer_used = infer_time - train_time
overall_used = infer_time - start_time

