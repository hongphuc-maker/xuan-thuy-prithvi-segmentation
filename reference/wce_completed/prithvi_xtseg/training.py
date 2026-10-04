from __future__ import annotations
import importlib.metadata
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from xuanthuy_seg.contracts import atomic_json, stable_hash, sha256_file
from xuanthuy_seg.losses.cross_entropy import deterministic_masked_cross_entropy
from .data import Scene, load_config, prepare, stack_crops, source_hash
from .model import build_model, predict_logits
from .inference import validation, create_map


def atomic_torch(payload,path):
    path=Path(path);temp=path.with_name(path.name+".tmp")
    torch.save(payload,temp);os.replace(temp,path)


def seed_everything(config):
    s=config["training"]["seed"]
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(config["training"]["deterministic_algorithms"])


def open_experiment(config_path,data_root,run_root):
    config=load_config(config_path);run=Path(run_root);run.mkdir(parents=True,exist_ok=True)
    versions={k:importlib.metadata.version(k) for k in ["torch","torchvision","terratorch","torchgeo","timm","huggingface-hub","rasterio","numpy","pandas"]}
    method={"config":config,"extension_source_sha256":source_hash(),"versions":versions,
            "baseline_commit":config["baseline_commit"],"baseline_package_sources":{}}
    import xuanthuy_seg.data.preparation as preparation
    import xuanthuy_seg.data.splits as splits
    import xuanthuy_seg.losses.cross_entropy as ce
    for mod in [preparation,splits,ce]:method["baseline_package_sources"][mod.__name__]=sha256_file(mod.__file__)
    digest=stable_hash(method);lock_path=run/"RUN_LOCK.json"
    if lock_path.exists():
        lock=json.loads(lock_path.read_text())
        if lock["method_hash"]!=digest:
            raise ValueError("Existing run has different data/model/code/runtime settings. Use a NEW run folder.")
    else:
        if any(run.iterdir()):raise ValueError("Nonempty run directory without lock; choose a new directory")
        atomic_json({"method_hash":digest,"method":method},lock_path)
        atomic_json(config,run/"experiment.json")
    scene=Scene(config,data_root)
    output=run/"prepared"
    audit_path=output/"DATA_AUDIT.json"
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        for name,sha in audit["artifact_sha256"].items():
            if sha256_file(output/name)!=sha:raise ValueError("Prepared data artifact changed")
        manifest=pd.read_csv(output/"train_manifest.csv")
        schedule=pd.read_csv(output/"batch_schedule.csv")
        val=pd.read_csv(output/"validation_manifest.csv")
        batches=[g.sort_values("slot").manifest_index.to_numpy(np.int64)
                 for _,g in schedule.groupby("batch_index",sort=True)]
    else:
        manifest,batches,val=prepare(scene,output)
    return Experiment(scene,run,manifest,batches,val,digest)


class Experiment:
    def __init__(self,scene,run,manifest,batches,val,method_hash):
        self.scene=scene;self.config=scene.config;self.run=run
        self.manifest=manifest;self.batches=batches;self.val=val;self.method_hash=method_hash
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model=None
    def load_model(self,pretrained=True,test_override=None):
        seed_everything(self.config)
        self.model=build_model(self.config,pretrained=pretrained,test_override=test_override).to(self.device)
        return self.model
    def _crop(self,index):
        row=self.manifest.iloc[int(index)]
        return self.scene.crop(int(row.row),int(row.col),domain=self.scene.domains["train_input"])
    def smoke_test(self):
        if self.model is None:self.load_model()
        self.model.train();crop=self._crop(self.batches[0][0]);batch=stack_crops([crop],self.device)
        self.model.zero_grad(set_to_none=True)
        if self.device.type=="cuda":torch.cuda.reset_peak_memory_stats()
        with torch.autocast(device_type=self.device.type,dtype=torch.float16,enabled=self.device.type=="cuda"):
            logits=predict_logits(self.model,batch)
        loss=deterministic_masked_cross_entropy(logits.float(),batch["target"],class_weights=self.scene.weights.to(self.device))
        if not torch.isfinite(loss):raise ValueError("Non-finite smoke loss")
        loss.backward()
        grads=[p.grad for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):raise ValueError("Invalid gradient")
        report={"input_shape":list(batch["image"].shape),"output_shape":list(logits.shape),
                "loss":float(loss.detach()),"gradient_tensors":len(grads),"device":str(self.device),
                "peak_cuda_gb":torch.cuda.max_memory_allocated()/1e9 if self.device.type=="cuda" else None,
                "pretrained":self.model.pretrained_provenance,"method_hash":self.method_hash}
        self.model.zero_grad(set_to_none=True)
        atomic_json(report,self.run/"SMOKE_TEST.json")
        # Smoke backward does not update weights; restore seed before the real experiment.
        seed_everything(self.config)
        return report
    def _batch_order(self,step):
        pass_index,within=divmod(step,len(self.batches))
        order=np.random.default_rng(self.config["training"]["seed"]+pass_index).permutation(len(self.batches))
        return self.batches[int(order[within])]
    def train(self,stop_after_steps=None):
        if self.device.type!="cuda" and stop_after_steps is None:
            raise RuntimeError("Full Prithvi training requires a Colab GPU; select GPU runtime")
        if self.model is None:self.load_model(pretrained=not (self.run/"latest_resume.pt").exists())
        t=self.config["training"];weights=self.scene.weights.to(self.device)
        encoder_ids={id(p) for p in self.model.encoder.parameters()}
        enc=[p for p in self.model.parameters() if p.requires_grad and id(p) in encoder_ids]
        dec=[p for p in self.model.parameters() if p.requires_grad and id(p) not in encoder_ids]
        optimizer=torch.optim.AdamW([{"params":enc,"lr":t["encoder_lr"]},
                                    {"params":dec,"lr":t["decoder_lr"]}],weight_decay=t["weight_decay"])
        scaler=torch.amp.GradScaler("cuda",enabled=self.device.type=="cuda")
        state={"step":0,"best_any":-1.,"best_pass":-1.,"stale":0,"patience_best":-1.,"history":[],"validation":[],"stopped":False}
        latest=self.run/"latest_resume.pt"
        if latest.exists():
            saved=torch.load(latest,map_location=self.device,weights_only=False)
            if saved["method_hash"]!=self.method_hash:raise ValueError("Resume contract differs")
            self.model.load_state_dict(saved["model"],strict=True)
            optimizer.load_state_dict(saved["optimizer"]);scaler.load_state_dict(saved["scaler"])
            state=saved["state"]
            self.model.pretrained_provenance=saved["pretrained_provenance"]
            torch.set_rng_state(saved["rng_cpu"].cpu())
            if self.device.type=="cuda":torch.cuda.set_rng_state_all([x.cpu() for x in saved["rng_cuda"]])
            del saved
        end=min(t["maximum_steps"],state["step"]+stop_after_steps) if stop_after_steps else t["maximum_steps"]
        def save_resume():
            atomic_torch({"method_hash":self.method_hash,"model":self.model.state_dict(),
                           "optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),"state":state,
                           "pretrained_provenance":self.model.pretrained_provenance,
                           "rng_cpu":torch.get_rng_state(),
                           "rng_cuda":torch.cuda.get_rng_state_all() if self.device.type=="cuda" else []},latest)
            pd.DataFrame(state["history"]).to_csv(self.run/"training_history.csv",index=False)
            pd.DataFrame(state["validation"]).to_csv(self.run/"validation_history.csv",index=False)
        for step in tqdm(range(state["step"],end),desc="Optimizer steps"):
            if state["stopped"]:break
            self.model.train();optimizer.zero_grad(set_to_none=True);start=time.monotonic()
            crops=[self._crop(i) for i in self._batch_order(step)]
            micro=t["micro_batch_size"]
            groups=[crops[i:i+micro] for i in range(0,len(crops),micro)]
            denominators=[sum(float(self.scene.weights[b["target"][b["target"]!=255]].sum()) for b in group)
                          for group in groups]
            denominator=sum(denominators)
            if denominator<=0:raise ValueError("Batch has no valid supervised pixels")
            total=0.
            for group,den in zip(groups,denominators):
                if den==0:continue
                batch=stack_crops(group,self.device)
                with torch.autocast(device_type=self.device.type,dtype=torch.float16,enabled=self.device.type=="cuda"):
                    logits=predict_logits(self.model,batch)
                # Exact whole-batch WCE weighting across microbatches, not mean of microbatch means.
                contribution=deterministic_masked_cross_entropy(logits.float(),batch["target"],class_weights=weights)*(den/denominator)
                if not torch.isfinite(contribution):raise ValueError("Non-finite WCE; aborting before optimizer update")
                scaler.scale(contribution).backward();total+=float(contribution.detach())
                del logits,batch,contribution
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),t["clip_grad_norm"],error_if_nonfinite=True)
            scaler.step(optimizer);scaler.update()
            state["step"]=step+1
            state["history"].append({"step":step+1,"train_WCE":total,"grad_norm":float(norm),
                                      "seconds":time.monotonic()-start})
            is_val=(step+1)%t["validation_interval_steps"]==0 or step+1==t["maximum_steps"]
            if is_val:
                summary,per=validation(self.model,self.scene,self.val,self.device)
                summary["step"]=step+1;state["validation"].append(summary)
                score=summary["macro_F1_11"]
                selected_payload={"model":self.model.state_dict(),"method_hash":self.method_hash,
                                  "summary":summary,"pretrained_provenance":self.model.pretrained_provenance}
                if score>state["best_any"]:
                    state["best_any"]=score;atomic_torch(selected_payload,self.run/"best_any.pt")
                if summary["acceptance_passed"] and score>state["best_pass"]:
                    state["best_pass"]=score;atomic_torch(selected_payload,self.run/"best_accepted.pt")
                if score>state["patience_best"]+t["min_delta"]:
                    state["patience_best"]=score;state["stale"]=0
                else:state["stale"]+=1
                state["stopped"]=state["stale"]>=t["patience_evaluations"]
                per.to_csv(self.run/"latest_validation_per_class.csv",index=False)
                print(f"Step {step+1}: val F1={score:.4f}, IoU={summary['macro_IoU_11']:.4f}, gate={summary['acceptance_passed']}")
            if is_val or (step+1)%t["save_interval_steps"]==0 or step+1==end:save_resume()
        # A short diagnostic stop is resumable and is not a completed scientific experiment.
        complete=state["stopped"] or state["step"]>=t["maximum_steps"]
        result={"step":state["step"],"completed":complete,"early_stopped":state["stopped"],
                "best_F1":state["best_any"],"best_accepted_F1":state["best_pass"],"method_hash":self.method_hash}
        atomic_json(result,self.run/("TRAINING_COMPLETE.json" if complete else "TRAINING_PROGRESS.json"))
        return result
    def map(self):
        if not (self.run/"TRAINING_COMPLETE.json").exists():
            raise ValueError("Complete training/model selection before producing the final map")
        accepted=self.run/"best_accepted.pt"
        chosen=accepted if accepted.exists() else self.run/"best_any.pt"
        if not chosen.exists():raise ValueError("No selected checkpoint")
        if self.model is None:self.load_model(pretrained=False)
        payload=torch.load(chosen,map_location=self.device,weights_only=False)
        if payload["method_hash"]!=self.method_hash:raise ValueError("Selected checkpoint has different contract")
        self.model.load_state_dict(payload["model"],strict=True)
        self.model.pretrained_provenance=payload["pretrained_provenance"]
        selection={"file":chosen.name,"acceptance_passed":bool(accepted.exists()),"summary":payload["summary"]}
        atomic_json(selection,self.run/"MODEL_SELECTION.json")
        if not accepted.exists():print("Acceptance gate failed: output is an exploratory map, not an accepted model.")
        return create_map(self.model,self.scene,self.run,chosen,self.device)
