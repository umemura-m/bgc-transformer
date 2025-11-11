#!/usr/bin/env python3

import argparse,shutil,logging,csv,pickle,subprocess,shutil
from pathlib import Path
from collections import defaultdict
from pretoken_memmap_dataset import PretokMemmapDataset
from tokenizers import Tokenizer
from transformers import RobertaConfig,RobertaForMaskedLM,DataCollatorForLanguageModeling,Trainer,TrainingArguments,PreTrainedTokenizerFast,TrainerCallback

# argument setting
ap = argparse.ArgumentParser()
ap.add_argument("--target",type=str,required=True,help="Target name")
ap.add_argument("--vocab-json",type=str,required=True,help="Path to your tokenizer JSON")
ap.add_argument("--epochs",type=int,required=True,help="Epoch number for training")
ap.add_argument("--save-every",type=int,default=50,help="Save model every N epochs")
args = ap.parse_args()

# tokenizer setting for special-token awareness
tok_obj = Tokenizer.from_file(args.vocab_json)
tok = PreTrainedTokenizerFast(tokenizer_object=tok_obj)
tok.add_special_tokens({"mask_token":"[MASK]","pad_token":"[PAD]","sep_token":"[SEP]"})

# datasets from pretokenized shards
train_meta = './shards.'+args.target+"/train.meta.json"
eval_meta  = './shards.'+args.target+"/eval.meta.json"
train_dataset = PretokMemmapDataset(train_meta)
eval_dataset  = PretokMemmapDataset(eval_meta)

# model setting
config = RobertaConfig(
    vocab_size=19723,
    max_position_embeddings=514,
    hidden_size=1024,
    num_attention_heads=16,
    num_hidden_layers=8,
    type_vocab_size=1
)
model = RobertaForMaskedLM(config)

# MLM collator (random masking every epoch)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tok, mlm=True, mlm_probability=0.15
)

#--- trainer (tune workers for speed) ---
odir = f"./out.{args.target}"
training_args = TrainingArguments(
    num_train_epochs=args.epochs,
    do_train=True,
    do_eval =True,
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    output_dir =odir,
    logging_dir=odir,
    overwrite_output_dir=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=1e-4,              # default:5e-5,tested 3e-4
    weight_decay=0.02,               # default:0.0
    warmup_ratio=0.04,               # default:0.0
    lr_scheduler_type="linear",      # default:"linear"
#   max_grad_norm=1.0,               # default:1.0
    fp16=True,                       # default:False
    dataloader_num_workers=0,
    dataloader_persistent_workers=False,
    report_to=[],
)

# callback instances
logger = logging.getLogger(__name__)

class SaveEveryXEpochs(TrainerCallback):
    def __init__(self,every=50,prefix="epoch_"):
        self.every = int(every)
        self.prefix = prefix
        self._tag_epoch = None

    def on_epoch_end(self,args,state,control,**kwargs):
        e = int(state.epoch or 0)
        if e and e % self.every == 0:
            control.should_save = True
            self._tag_epoch = e
        return control

    def on_save(self,args,state,control,**kwargs):
        if getattr(state,"is_world_process_zero",True) and self._tag_epoch:
            e = self._tag_epoch
            self._tag_epoch = None

            chk_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            tag_dir = Path(args.output_dir) / f"{self.prefix}{e}"

            if chk_dir.exists():
                if tag_dir.exists():
                    shutil.rmtree(tag_dir) if tag_dir.is_dir() else tag_dir.unlink()
                chk_dir.rename(tag_dir)
                logger.info(f"Renamed {chk_dir.name} -> {tag_dir.name}")
            else:
                logger.warning(f"Expected {chk_dir} not found; skipping rename")
        return control

class EpochLossesRecorder(TrainerCallback):
    def __init__(self,pkl_name="epoch_losses.pkl",csv_name="epoch_losses.csv",write_each_eval=True,float_fmt=".6f"):
        self.pkl_name,self.csv_name = pkl_name,csv_name
        self.write_each_eval = write_each_eval
        self.float_fmt = float_fmt
        self._train_losses = defaultdict(list)   # epoch -> [losses]
        self.records = []

    @staticmethod
    def _is_main(state):  # world 0
        return getattr(state, "is_world_process_zero",True)

    def on_log(self,args,state,control,logs=None,**kwargs):
        if self._is_main(state) and logs and state.epoch is not None and "loss" in logs:
            self._train_losses[int(state.epoch)].append(float(logs["loss"]))
        return control

    def on_evaluate(self,args,state,control,metrics=None,**kwargs):
        if not (self._is_main(state) and state.epoch is not None):
            return control
        e = int(state.epoch)
        tl = self._train_losses.get(e, [])
        self.records.append({
            "epoch": e,
            "global_step": state.global_step,
            "train_loss": float(tl[-1]) if tl else None,
            "eval_loss": float(metrics["eval_loss"]) if metrics and "eval_loss" in metrics else None,
        })
        if self.write_each_eval:
            self._dump(args)
        return control

    def on_train_end(self,args,state,control,**kwargs):
        if self._is_main(state):
            self._dump(args)
        return control

    def _dump(self,args):
        out = Path(args.output_dir)
        out.mkdir(parents=True,exist_ok=True)
        with open(out / self.pkl_name,"wb") as f:
            pickle.dump(self.records,f)
        with open(out / self.csv_name,"w",newline="",encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch","global_step","train_loss","eval_loss"])
            fmt = lambda x: "" if x is None else f"{x:{self.float_fmt}}"
            for r in self.records:
                w.writerow([r["epoch"], r["global_step"], fmt(r["train_loss"]), fmt(r["eval_loss"])])

save_model_callback = SaveEveryXEpochs(every=args.save_every)
epoch_losses_callback  = EpochLossesRecorder()

# trainer setting
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[save_model_callback,epoch_losses_callback]
)

trainer.train()

#--- end ---
