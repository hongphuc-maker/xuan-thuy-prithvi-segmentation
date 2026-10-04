from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import rasterio

COLORS=["#d7e3e8","#245c83","#84b6cf","#dc967a","#dcb45a","#ab65a0","#ded7be",
        "#6e6e6e","#1d6d3e","#7ab753","#8fd3d7","#bbd4ad"]


def show_inputs(experiment):
    scene=experiment.scene;fig,axes=plt.subplots(2,3,figsize=(15,9))
    offset=scene.config["radiometry"]["dn_offset"]
    scale=scene.config["radiometry"]["scale"]
    for i,(ax,spec) in enumerate(zip(axes.flat,scene.config["images"])):
        a=(scene.raw[i,[2,1,0]].astype(np.float32)+offset)/scale
        rgb=np.clip((a-.02)/.23,0,1).transpose(1,2,0)
        rgb[~scene.valid_by_time[i]]=1
        ax.imshow(rgb);ax.set_title(spec["date"]);ax.axis("off")
    fig.suptitle("RGB cùng thang hiển thị — cần kiểm tra mây/bóng mây bằng SCL")
    fig.tight_layout();return fig


def show_split(experiment):
    s=experiment.scene;fig,axes=plt.subplots(1,2,figsize=(14,6))
    axes[0].imshow(s.domains["split_mask"],cmap=ListedColormap(["white","#4daf4a","#377eb8","#ffb347"]),vmin=0,vmax=3)
    axes[0].set_title("Train xanh lá · validation xanh dương · guard cam")
    axes[1].imshow(s.domains["validation_input"],cmap="Greys");axes[1].set_title("Envelope đầu vào validation; chấm điểm chỉ core")
    for ax in axes:ax.axis("off")
    fig.tight_layout();return fig


def show_history(run_root):
    run=Path(run_root);fig,axes=plt.subplots(1,3,figsize=(16,4))
    tr=run/"training_history.csv";vp=run/"validation_history.csv"
    if tr.exists():
        x=pd.read_csv(tr)
        if not x.empty:axes[0].plot(x.step,x.train_WCE,label="Train WCE")
    if vp.exists():
        try:v=pd.read_csv(vp)
        except pd.errors.EmptyDataError:v=pd.DataFrame()
        if not v.empty:
            axes[0].plot(v.step,v.WCE,label="Validation mosaic WCE")
            axes[1].plot(v.step,v.macro_F1_11);axes[2].plot(v.step,v.macro_IoU_11)
    for ax,title in zip(axes,["WCE","Validation macro-F1 (11 lớp)","Validation macro-IoU (11 lớp)"]):
        ax.set_title(title);ax.set_xlabel("Optimizer step");ax.grid(alpha=.2)
    if axes[0].lines:axes[0].legend()
    fig.tight_layout();return fig


def show_map(experiment):
    s=experiment.scene;p=experiment.run/"map"/"classes_sep21_2026.tif"
    with rasterio.open(p) as src:a=src.read(1)
    fig,ax=plt.subplots(figsize=(13,8));im=ax.imshow(a,cmap=ListedColormap(COLORS),norm=BoundaryNorm(np.arange(-.5,12.5),12))
    cb=fig.colorbar(im,ax=ax,ticks=range(12),shrink=.8)
    cb.ax.set_yticklabels(["NoData"]+[s.config["class_map"][str(i)] for i in range(1,12)])
    ax.set_title("Xuân Thủy — bản đồ mục tiêu 21/09/2026 (nhãn lịch sử)");ax.axis("off")
    fig.tight_layout();return fig


def compare_validation(run_roots):
    rows=[]
    for root in run_roots:
        p=Path(root)/"MODEL_SELECTION.json"
        if p.exists():
            x=json.loads(p.read_text());c=json.loads((Path(root)/"experiment.json").read_text())
            rows.append({"run":Path(root).name,"T":len(c["active_dates"]),
                         "target_date":c["target_date"],"acceptance_passed":x["acceptance_passed"],**x["summary"]})
    return pd.DataFrame(rows)
