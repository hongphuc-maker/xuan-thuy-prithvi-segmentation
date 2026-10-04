from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import torch
from tqdm.auto import tqdm
from xuanthuy_seg.contracts import atomic_json, sha256_file, stable_hash
from .data import stack_crops
from .model import predict_logits


def metrics(confusion, classes):
    tp = np.diag(confusion).astype(float)
    support = confusion.sum(1).astype(float); predicted = confusion.sum(0).astype(float)
    precision = np.divide(tp,predicted,out=np.zeros(11),where=predicted>0)
    recall = np.divide(tp,support,out=np.zeros(11),where=support>0)
    f1 = np.divide(2*tp,support+predicted,out=np.zeros(11),where=support+predicted>0)
    union = support+predicted-tp
    iou = np.divide(tp,union,out=np.zeros(11),where=union>0)
    frame = pd.DataFrame({"code":range(1,12),"class":[classes[str(i)] for i in range(1,12)],
                           "support":support.astype(int),"predicted":predicted.astype(int),
                           "precision":precision,"recall":recall,"F1":f1,"IoU":iou})
    summary = {"OA":float(tp.sum()/max(support.sum(),1)),"macro_F1_11":float(f1.mean()),
               "macro_IoU_11":float(iou.mean()),"predicted_classes":int((predicted>0).sum()),
               "n_pixels":int(support.sum())}
    return summary, frame


def starts(length, size, stride):
    if length <= size: return [0]
    out = list(range(0,length-size+1,stride))
    if out[-1] != length-size: out.append(length-size)
    return out


@torch.inference_mode()
def mosaic(model, scene, windows, device, domain=None, progress=True):
    was_training = model.training; model.eval()
    h,w = scene.label.shape
    acc = np.zeros((11,h,w),np.float32); counts = np.zeros((h,w),np.uint16)
    try:
        for win in tqdm(windows,desc="Probability mosaic",disable=not progress):
            r,c,hh,ww = [int(win[k]) for k in ("row","col","height","width")]
            crop = scene.crop(r,c,hh,ww,domain=domain)
            batch = stack_crops([crop],device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                logits = predict_logits(model,batch)
            prob = logits.float().softmax(1)[0,:,:hh,:ww].cpu().numpy()
            mask = scene.image_valid[r:r+hh,c:c+ww].copy()
            if domain is not None: mask &= domain[r:r+hh,c:c+ww]
            acc[:,r:r+hh,c:c+ww] += prob * mask[None]
            counts[r:r+hh,c:c+ww] += mask.astype(np.uint16)
    finally:
        model.train(was_training)
    np.divide(acc,counts[None],out=acc,where=counts[None]>0)
    return acc, counts


def validation(model,scene,val_manifest,device):
    prob,counts = mosaic(model,scene,val_manifest.to_dict("records"),device,
                          domain=scene.domains["validation_input"],progress=False)
    mask = scene.val_mask
    if not np.all(counts[mask]>0): raise ValueError("Validation core has uncovered pixels")
    truth = scene.label[mask].astype(np.int64)-1
    pred = prob[:,mask].argmax(0)
    confusion = np.bincount(truth*11+pred,minlength=121).reshape(11,11)
    summary,frame = metrics(confusion,scene.config["class_map"])
    weights = scene.weights.numpy()[truth]
    chosen = prob[:,mask][truth,np.arange(len(truth))]
    summary["WCE"] = float((weights * -np.log(np.maximum(chosen,1e-12))).sum()/weights.sum())
    gate = scene.config["acceptance"]
    passed = summary["predicted_classes"] == gate["required_predicted_classes"]
    passed &= all(frame.loc[frame.code==int(k),"recall"].iloc[0] >= v
                  for k,v in gate["minimum_recall_by_code"].items())
    summary["acceptance_passed"] = bool(passed)
    return summary,frame


def create_map(model,scene,run_root,checkpoint_path,device):
    run_root = Path(run_root); out = run_root / "map"; out.mkdir(exist_ok=True)
    lock = json.loads((run_root/"RUN_LOCK.json").read_text())
    ckpt_hash = sha256_file(checkpoint_path)
    marker = out/"MAP_COMPLETE.json"
    if marker.exists():
        old=json.loads(marker.read_text())
        if old["method_hash"]!=lock["method_hash"] or old["checkpoint_sha256"]!=ckpt_hash:
            raise ValueError("Map already sealed for a different checkpoint; use a new run")
        for name,digest in old["artifacts"].items():
            if sha256_file(out/name)!=digest: raise ValueError("Sealed map artifact changed")
        return old
    h,w=scene.label.shape; size=scene.config["split"]["patch_policy"]["size"]
    stride=scene.config["split"]["patch_policy"]["inference_stride"]
    windows=[{"row":r,"col":c,"height":min(size,h-r),"width":min(size,w-c)}
              for r in starts(h,size,stride) for c in starts(w,size,stride)
              if scene.image_valid[r:r+size,c:c+size].any()]
    prob,counts=mosaic(model,scene,windows,device)
    valid=scene.image_valid & (counts>0)
    if not np.all(counts[scene.image_valid]>0): raise ValueError("Map has uncovered valid pixels")
    code=np.zeros((h,w),np.uint8);code[valid]=prob[:,valid].argmax(0).astype(np.uint8)+1
    confidence=np.full((h,w),np.nan,np.float32);confidence[valid]=prob[:,valid].max(0)
    entropy=np.full((h,w),np.nan,np.float32)
    entropy[valid]=-(prob[:,valid]*np.log(np.maximum(prob[:,valid],1e-12))).sum(0)/np.log(11)
    arrays={"classes_sep21_2026.tif":code,"max_softmax_uncalibrated.tif":confidence,
            "entropy_uncalibrated.tif":entropy,"common_validity.tif":valid.astype(np.uint8)}
    for name,array in arrays.items():
        p=out/name;temp=p.with_name(p.stem+".tmp.tif")
        nodata=np.nan if np.issubdtype(array.dtype,np.floating) else 0
        profile=scene.profile.copy();profile.update(count=1,dtype=str(array.dtype),nodata=nodata,compress="deflate")
        with rasterio.open(temp,"w",**profile) as dst:
            dst.write(array,1)
            dst.update_tags(target_date=scene.config["target_date"],method_hash=lock["method_hash"],
                            input_dates=",".join(scene.config["active_dates"]),
                            label_status=scene.config["label"]["status"],
                            quality_status=scene.config["quality_status"])
        temp.replace(p)
    marker_data={"method_hash":lock["method_hash"],"checkpoint_sha256":ckpt_hash,
                 "target_date":scene.config["target_date"],"valid_pixels":int(valid.sum()),
                 "artifacts":{n:sha256_file(out/n) for n in arrays},
                 "note":"Target September map; not validated against date-matched September ground truth"}
    atomic_json(marker_data,marker)
    return marker_data


def evaluate_reference_points(scene,run_root,points_root,confirm=False):
    if not confirm:
        raise ValueError("Enable this cell only after checkpoint/map selection is final")
    import geopandas as gpd
    import tempfile,zipfile
    run_root=Path(run_root);out=run_root/"reference_point_agreement";out.mkdir(exist_ok=True)
    marker=run_root/"map"/"MAP_COMPLETE.json"
    seal=json.loads(marker.read_text())
    mp=run_root/"map"/"classes_sep21_2026.tif"
    if sha256_file(mp)!=seal["artifacts"][mp.name]:raise ValueError("Map checksum changed")
    spec=scene.config["independent_points"];zp=Path(points_root)/spec["path"]
    if sha256_file(zp)!=spec["sha256"]:raise ValueError("Point ZIP checksum differs")
    completion=out/"AGREEMENT_COMPLETE.json"
    if completion.exists():
        old=json.loads(completion.read_text())
        if old["map_sha256"]!=sha256_file(mp):raise ValueError("Evaluation map changed")
        for name,digest in old["artifacts"].items():
            if sha256_file(out/name)!=digest:raise ValueError("Evaluation artifact changed")
        return old
    by_name={v.casefold():int(k) for k,v in scene.config["class_map"].items()}
    records=[]
    with rasterio.open(mp) as src,tempfile.TemporaryDirectory() as temp:
        with zipfile.ZipFile(zp) as z:
            for entry in z.infolist():
                dest=(Path(temp)/entry.filename).resolve()
                if Path(temp).resolve() not in dest.parents:raise ValueError("Unsafe ZIP path")
            z.extractall(temp)
        files=sorted(Path(temp).rglob("*.shp"))
        if not files:raise ValueError("No point shapefiles")
        for file in files:
            if file.stem.casefold() not in by_name:raise ValueError(f"Unknown class layer {file.stem}")
            frame=gpd.read_file(file)
            if frame.crs is None:raise ValueError("Point layer missing CRS")
            frame=frame.to_crs(src.crs)
            for idx,p in frame.geometry.items():
                if p is None or p.geom_type!="Point":raise ValueError("Point geometry required")
                r,c=src.index(p.x,p.y);inside=0<=r<src.height and 0<=c<src.width
                pred=int(next(src.sample([(p.x,p.y)]))[0]) if inside else 0
                records.append({"layer":file.stem,"id":str(idx),"truth":by_name[file.stem.casefold()],
                                "pred":pred,"row":r,"col":c,"usable":inside and 1<=pred<=11})
    frame=pd.DataFrame(records);good=frame[frame.usable]
    if good.empty:raise ValueError("No usable reference points")
    cm=np.bincount((good.truth.to_numpy()-1)*11+good.pred.to_numpy()-1,minlength=121).reshape(11,11)
    summary,per=metrics(cm,scene.config["class_map"])
    summary["macro_F1_represented_classes"]=float(per.loc[per.support>0,"F1"].mean())
    summary.update(input_points=len(frame),excluded_points=len(frame)-len(good),
                   interpretation="Agreement with previously used reference points; September accuracy not established",
                   map_sha256=sha256_file(mp),method_hash=seal["method_hash"])
    frame.to_csv(out/"points.csv",index=False);per.to_csv(out/"per_class.csv",index=False)
    np.savetxt(out/"confusion.csv",cm,fmt="%d",delimiter=",")
    summary["artifacts"]={n:sha256_file(out/n) for n in ["points.csv","per_class.csv","confusion.csv"]}
    atomic_json(summary,completion);return summary
