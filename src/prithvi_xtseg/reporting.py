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
    run=Path(run_root);fig,axes=plt.subplots(1,4,figsize=(20,4))
    tr=run/"dmi_training_history.csv";vp=run/"dmi_validation_history.csv"
    if tr.exists():
        x=pd.read_csv(tr)
        if not x.empty:
            axes[0].plot(x.step,x.train_DMI_loss,label="Train DMI")
            axes[3].plot(x.step,x.train_DMI_condition_number,label="Train condition")
    if vp.exists():
        try:v=pd.read_csv(vp)
        except pd.errors.EmptyDataError:v=pd.DataFrame()
        if not v.empty:
            axes[0].plot(v.step,v.validation_DMI_loss,label="Validation DMI")
            axes[1].plot(v.step,v.macro_F1_11);axes[2].plot(v.step,v.macro_IoU_11)
            axes[3].scatter(v.step,v.validation_DMI_condition_number,label="Validation condition",s=16)
    for ax,title in zip(axes,["Exact DMI","Mapped validation macro-F1 (11 lớp)",
                              "Mapped validation macro-IoU (11 lớp)","DMI condition number"]):
        ax.set_title(title);ax.set_xlabel("Optimizer step");ax.grid(alpha=.2)
    for ax in (axes[0],axes[3]):
        if ax.lines or ax.collections:ax.legend()
    axes[3].set_yscale("log")
    fig.tight_layout();return fig


def show_selected_checkpoints(experiment):
    frame=experiment.selected_checkpoint_table()
    if frame.empty:return frame
    columns=["role","file","dmi_step","macro_F1_11","macro_IoU_11","OA",
             "validation_DMI_loss","validation_DMI_rank",
             "validation_DMI_condition_number","acceptance_passed"]
    return frame[[column for column in columns if column in frame.columns]]


def show_map(experiment):
    s=experiment.scene;p=experiment.run/"map"/"classes_sep21_2026.tif"
    with rasterio.open(p) as src:a=src.read(1)
    fig,ax=plt.subplots(figsize=(13,8));im=ax.imshow(a,cmap=ListedColormap(COLORS),norm=BoundaryNorm(np.arange(-.5,12.5),12))
    cb=fig.colorbar(im,ax=ax,ticks=range(12),shrink=.8)
    cb.ax.set_yticklabels(["NoData"]+[s.config["class_map"][str(i)] for i in range(1,12)])
    ax.set_title("Xuân Thủy — bản đồ mục tiêu 21/09/2026 (nhãn lịch sử)");ax.axis("off")
    fig.tight_layout();return fig


def show_warmup_history(run_root):
    run=Path(run_root);fig,axes=plt.subplots(1,3,figsize=(15,4))
    tr=run/"warmup_training_history.csv";vp=run/"warmup_validation_history.csv"
    if tr.exists():
        x=pd.read_csv(tr)
        if not x.empty:axes[0].plot(x.step,x.train_WCE)
    if vp.exists():
        try:v=pd.read_csv(vp)
        except pd.errors.EmptyDataError:v=pd.DataFrame()
        if not v.empty:
            axes[1].plot(v.step,v.macro_F1_11,label="macro-F1")
            axes[1].plot(v.step,v.macro_IoU_11,label="macro-IoU")
            axes[2].plot(v.step,v.predicted_classes,label="predicted classes")
    for ax,title in zip(axes,["Warm-up WCE","Mapped validation","Class coverage"]):
        ax.set_title(title);ax.set_xlabel("Warm-up optimizer step");ax.grid(alpha=.2)
    for ax in axes:
        if ax.lines:ax.legend()
    fig.tight_layout();return fig


def compare_validation(run_roots):
    rows=[]
    for root in run_roots:
        p=Path(root)/"MODEL_SELECTION.json"
        if p.exists():
            x=json.loads(p.read_text());c=json.loads((Path(root)/"experiment.json").read_text())
            primary=x.get("checkpoints",{}).get("primary_macro_F1",{})
            if primary.get("available"):
                rows.append({"run":Path(root).name,"T":len(c["active_dates"]),
                             "target_date":c["target_date"],**primary["summary"]})
    return pd.DataFrame(rows)
