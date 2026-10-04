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

from xuanthuy_seg.contracts import atomic_json, sha256_file, stable_hash
from xuanthuy_seg.losses.cross_entropy import deterministic_masked_cross_entropy

from .data import Scene, load_config, prepare, source_hash, stack_crops
from .dmi import (
    diagnostics_from_joint,
    exact_joint_upstream_gradient,
    joint_numerator_from_logits,
    validate_joint_for_gradient,
)
from .inference import create_map, validation
from .model import build_model, predict_logits, predict_raw_logits


def atomic_torch(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def seed_everything(config):
    seed = int(config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(config["training"]["deterministic_algorithms"])


def open_experiment(config_path, data_root, run_root):
    config = load_config(config_path)
    run = Path(run_root)
    run.mkdir(parents=True, exist_ok=True)
    versions = {
        key: importlib.metadata.version(key)
        for key in [
            "torch",
            "torchvision",
            "terratorch",
            "torchgeo",
            "timm",
            "huggingface-hub",
            "rasterio",
            "numpy",
            "pandas",
        ]
    }
    method = {
        "config": config,
        "extension_source_sha256": source_hash(),
        "versions": versions,
        "baseline_commit": config["baseline_commit"],
        "baseline_package_sources": {},
    }
    import xuanthuy_seg.data.preparation as preparation
    import xuanthuy_seg.data.splits as splits
    import xuanthuy_seg.losses.cross_entropy as cross_entropy

    for module in [preparation, splits, cross_entropy]:
        method["baseline_package_sources"][module.__name__] = sha256_file(module.__file__)
    digest = stable_hash(method)
    lock_path = run / "RUN_LOCK.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if lock["method_hash"] != digest:
            raise ValueError(
                "Existing run has different data/model/code/runtime settings. Use a NEW run folder."
            )
    else:
        if any(run.iterdir()):
            raise ValueError("Nonempty run directory without lock; choose a new directory")
        atomic_json({"method_hash": digest, "method": method}, lock_path)
        atomic_json(config, run / "experiment.json")

    scene = Scene(config, data_root)
    output = run / "prepared"
    audit_path = output / "DATA_AUDIT.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        for name, sha in audit["artifact_sha256"].items():
            if sha256_file(output / name) != sha:
                raise ValueError("Prepared data artifact changed")
        manifest = pd.read_csv(output / "train_manifest.csv")
        schedule = pd.read_csv(output / "batch_schedule.csv")
        validation_manifest = pd.read_csv(output / "validation_manifest.csv")
        batches = [
            group.sort_values("slot").manifest_index.to_numpy(np.int64)
            for _, group in schedule.groupby("batch_index", sort=True)
        ]
    else:
        manifest, batches, validation_manifest = prepare(scene, output)

    panel_batches = int(config["training"]["permutation_panel_batches"])
    readiness_batches = int(config["warmup"]["readiness"]["fixed_train_batches"])
    fixed_batches = max(panel_batches, readiness_batches)
    if fixed_batches > len(batches):
        raise ValueError("Fixed panel requests more batches than the prepared schedule")
    panel_indices = [
        int(value)
        for batch in batches[:panel_batches]
        for value in batch.tolist()
    ]
    panel_contract = {
        "source": "first deterministic class-complete training batches",
        "n_batches": panel_batches,
        "n_patches": len(panel_indices),
        "manifest_indices": panel_indices,
        "patch_ids": [str(manifest.iloc[index].patch_id) for index in panel_indices],
        "used_for": (
            "channel-to-class permutation and warm-up DMI readiness only; "
            "never final accuracy"
        ),
    }
    panel_path = run / "FIXED_TRAIN_PANEL.json"
    if panel_path.exists():
        if json.loads(panel_path.read_text()) != panel_contract:
            raise ValueError("Existing fixed train panel differs from the locked schedule")
    else:
        atomic_json(panel_contract, panel_path)
    return Experiment(
        scene,
        run,
        manifest,
        batches,
        validation_manifest,
        panel_indices,
        digest,
    )


class Experiment:
    def __init__(self, scene, run, manifest, batches, val, permutation_indices, method_hash):
        self.scene = scene
        self.config = scene.config
        self.run = run
        self.manifest = manifest
        self.batches = batches
        self.val = val
        self.permutation_indices = permutation_indices
        self.method_hash = method_hash
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self, pretrained=True, test_override=None):
        seed_everything(self.config)
        self.model = build_model(
            self.config, pretrained=pretrained, test_override=test_override
        ).to(self.device)
        return self.model

    def _crop(self, index):
        row = self.manifest.iloc[int(index)]
        return self.scene.crop(
            int(row.row), int(row.col), domain=self.scene.domains["train_input"]
        )

    def _batch_order(self, step):
        pass_index, within = divmod(step, len(self.batches))
        order = np.random.default_rng(
            int(self.config["training"]["seed"]) + pass_index
        ).permutation(len(self.batches))
        return self.batches[int(order[within])]

    def _capture_rng(self):
        return {
            "cpu": torch.get_rng_state().clone(),
            "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
            if self.device.type == "cuda"
            else [],
        }

    def _restore_rng(self, state):
        torch.set_rng_state(state["cpu"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])

    def _set_stage(self, stage):
        if stage not in {"warmup", "dmi"}:
            raise ValueError(f"Unknown training stage: {stage}")
        for parameter in self.model.encoder.parameters():
            parameter.requires_grad_(False)
        morphology_trainable = stage == "dmi"
        for parameter in self.model.logit_closing.parameters():
            parameter.requires_grad_(morphology_trainable)
        self.model.train()
        self.model.encoder.eval()
        self.model.logit_closing.train(morphology_trainable)

    def _stage_parameters(self, stage):
        self._set_stage(stage)
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError(f"No trainable parameters for {stage}")
        encoder_ids = {id(parameter) for parameter in self.model.encoder.parameters()}
        if any(id(parameter) in encoder_ids for parameter in parameters):
            raise AssertionError("Frozen Prithvi encoder leaked into the optimizer")
        return parameters

    def parameter_report(self):
        if self.model is None:
            raise ValueError("Load the model first")
        encoder = list(self.model.encoder.parameters())
        morphology = list(self.model.logit_closing.parameters())
        encoder_ids = {id(parameter) for parameter in encoder}
        morphology_ids = {id(parameter) for parameter in morphology}
        task = [
            parameter
            for parameter in self.model.parameters()
            if id(parameter) not in encoder_ids and id(parameter) not in morphology_ids
        ]
        return {
            "encoder_parameters": int(sum(parameter.numel() for parameter in encoder)),
            "task_parameters": int(sum(parameter.numel() for parameter in task)),
            "morphology_parameters": int(sum(parameter.numel() for parameter in morphology)),
            "encoder_trainable": int(sum(parameter.numel() for parameter in encoder if parameter.requires_grad)),
            "task_trainable": int(sum(parameter.numel() for parameter in task if parameter.requires_grad)),
            "morphology_trainable": int(sum(parameter.numel() for parameter in morphology if parameter.requires_grad)),
            "encoder_training_mode": bool(self.model.encoder.training),
        }

    def _dmi_scaler(self):
        training = self.config["training"]
        return torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda",
            init_scale=float(training["dmi_amp_init_scale"]),
            growth_interval=int(training["dmi_amp_growth_interval"]),
        )

    def _exact_dmi_backward(self, batch_indices, scaler):
        training = self.config["training"]
        crops = [self._crop(index) for index in batch_indices]
        micro = int(training["micro_batch_size"])
        groups = [crops[index : index + micro] for index in range(0, len(crops), micro)]
        joint_numerator = torch.zeros((11, 11), dtype=torch.float64, device=self.device)
        valid_pixels = 0
        classes_present: set[int] = set()
        replay_states = []
        self._set_stage("dmi")
        with torch.no_grad():
            for group in groups:
                replay_states.append(self._capture_rng())
                batch = stack_crops(group, self.device)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.device.type == "cuda",
                ):
                    logits = predict_logits(self.model, batch)
                numerator, count = joint_numerator_from_logits(logits, batch["target"])
                joint_numerator += numerator
                valid_pixels += count
                labels = batch["target"]
                present = torch.unique(labels[labels != 255]).detach().cpu().tolist()
                classes_present.update(int(value) for value in present)
                del logits, batch, numerator
        rng_after_first_pass = self._capture_rng()
        if valid_pixels <= 0:
            raise ValueError("Effective DMI batch contains zero supervised pixels")
        joint = joint_numerator / float(valid_pixels)
        _, diagnostics = diagnostics_from_joint(
            joint,
            n_samples=valid_pixels,
            classes_present=len(classes_present),
        )
        if diagnostics.classes_present != 11:
            raise FloatingPointError(
                f"Effective DMI batch contains {diagnostics.classes_present}/11 classes"
            )
        validate_joint_for_gradient(joint, diagnostics)
        upstream = exact_joint_upstream_gradient(joint, valid_pixels).detach()
        try:
            for group, rng_state in zip(groups, replay_states, strict=True):
                self._restore_rng(rng_state)
                batch = stack_crops(group, self.device)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.device.type == "cuda",
                ):
                    logits = predict_logits(self.model, batch)
                local_numerator, _ = joint_numerator_from_logits(logits, batch["target"])
                surrogate = (local_numerator * upstream).sum()
                if not torch.isfinite(surrogate):
                    raise FloatingPointError("Non-finite exact-DMI gradient surrogate")
                scaler.scale(surrogate).backward()
                del logits, batch, local_numerator, surrogate
        finally:
            self._restore_rng(rng_after_first_pass)
        return diagnostics

    def _gradient_report(self):
        present = []
        missing = []
        bad = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                missing.append(name)
                continue
            present.append(name)
            if not bool(torch.isfinite(parameter.grad).all().detach().cpu()):
                bad.append(name)
        return {
            "finite": bool(present and not bad),
            "gradient_tensors": len(present),
            "missing_gradient_tensors": len(missing),
            "nonfinite_gradient_names": bad,
        }

    def _warmup_readiness(self, step):
        warmup = self.config["warmup"]
        original_rng = self._capture_rng()
        panel_reports = []
        try:
            self._set_stage("dmi")
            summary, per_class, permutation = validation(
                self.model,
                self.scene,
                self.val,
                self.manifest,
                self.permutation_indices,
                self.device,
            )
            n_fixed = int(warmup["readiness"]["fixed_train_batches"])
            for batch_index, indices in enumerate(self.batches[:n_fixed]):
                self._restore_rng(original_rng)
                self.model.zero_grad(set_to_none=True)
                record = {"batch_index": int(batch_index)}
                try:
                    diagnostics = self._exact_dmi_backward(indices, self._dmi_scaler())
                    gradients = self._gradient_report()
                    record.update(
                        DMI=diagnostics.to_dict(),
                        gradients=gradients,
                        passed=bool(
                            diagnostics.rank == 11
                            and diagnostics.finite_loss
                            and gradients["finite"]
                        ),
                    )
                except Exception as error:
                    record.update(
                        passed=False,
                        error_type=type(error).__name__,
                        error=str(error),
                    )
                finally:
                    self.model.zero_grad(set_to_none=True)
                panel_reports.append(record)
        finally:
            self._restore_rng(original_rng)
            self._set_stage("warmup")

        rare_thresholds = warmup["readiness"]["minimum_recall_by_code"]
        rare_recalls = {
            str(code): float(per_class.loc[per_class.code == int(code), "recall"].iloc[0])
            for code in rare_thresholds
        }
        flags = {
            "validation_DMI_rank_11": int(summary["validation_DMI_rank"]) == 11,
            "validation_DMI_finite": bool(summary["validation_DMI_finite"]),
            "four_fixed_batches_rank_11": bool(
                len(panel_reports) == 4
                and all(report.get("DMI", {}).get("rank") == 11 for report in panel_reports)
            ),
            "four_fixed_batches_finite_DMI": bool(
                len(panel_reports) == 4
                and all(report.get("DMI", {}).get("finite_loss", False) for report in panel_reports)
            ),
            "four_fixed_batches_finite_backward": bool(
                len(panel_reports) == 4 and all(report.get("passed", False) for report in panel_reports)
            ),
            "validation_predicts_all_11_classes": int(summary["predicted_classes"]) == 11,
            "rare_recall_gate": all(
                rare_recalls[str(code)] >= float(threshold)
                for code, threshold in rare_thresholds.items()
            ),
        }
        report = {
            "warmup_step": int(step),
            "ready": bool(all(flags.values())),
            "flags": flags,
            "rare_recalls": rare_recalls,
            "validation": summary,
            "fixed_train_batches": panel_reports,
            "permutation": permutation,
        }
        return report, per_class, permutation

    def warmup(self, target_step=None):
        if self.device.type != "cuda" and target_step is None:
            raise RuntimeError("Full Prithvi warm-up requires a Colab GPU")
        warmup = self.config["warmup"]
        training = self.config["training"]
        transition_path = self.run / "warmup_transition.pt"
        transition_json = transition_path.with_suffix(".json")
        if transition_path.exists():
            metadata = json.loads(transition_json.read_text())
            if metadata["method_hash"] != self.method_hash:
                raise ValueError("Warm-up transition belongs to another method")
            return metadata["result"]

        latest = self.run / "warmup_latest_resume.pt"
        if latest.exists():
            if self.model is None:
                self.load_model(pretrained=False)
        elif self.model is None:
            self.load_model(pretrained=True)
        parameters = self._stage_parameters("warmup")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(training["decoder_lr"]),
            weight_decay=float(training["weight_decay"]),
        )
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda",
            init_scale=float(training["warmup_amp_init_scale"]),
        )
        state = {
            "step": 0,
            "consecutive_ready": 0,
            "history": [],
            "validation": [],
            "readiness": [],
            "transition_ready": False,
        }
        if latest.exists():
            saved = torch.load(latest, map_location=self.device, weights_only=False)
            if saved["method_hash"] != self.method_hash:
                raise ValueError("Warm-up resume contract differs")
            self.model.load_state_dict(saved["model"], strict=True)
            self._set_stage("warmup")
            optimizer.load_state_dict(saved["optimizer"])
            scaler.load_state_dict(saved["scaler"])
            state = saved["state"]
            self.model.pretrained_provenance = saved["pretrained_provenance"]
            torch.set_rng_state(saved["rng_cpu"].cpu())
            if self.device.type == "cuda":
                torch.cuda.set_rng_state_all([value.cpu() for value in saved["rng_cuda"]])
            del saved

        maximum = int(warmup["maximum_steps"])
        end = maximum if target_step is None else min(maximum, int(target_step))
        interval = int(warmup["readiness_interval_steps"])
        minimum = int(warmup["minimum_steps"])
        required_consecutive = int(warmup["consecutive_ready_evaluations"])
        weights_cpu = self.scene.weights
        weights = weights_cpu.to(self.device)

        def save_resume():
            atomic_torch(
                {
                    "stage": "WCE_warmup",
                    "method_hash": self.method_hash,
                    "model": self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "state": state,
                    "pretrained_provenance": self.model.pretrained_provenance,
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all()
                    if self.device.type == "cuda"
                    else [],
                },
                latest,
            )
            pd.DataFrame(state["history"]).to_csv(
                self.run / "warmup_training_history.csv", index=False
            )
            pd.DataFrame(state["validation"]).to_csv(
                self.run / "warmup_validation_history.csv", index=False
            )
            atomic_json(state["readiness"], self.run / "warmup_readiness_history.json")

        for step in tqdm(range(int(state["step"]), end), desc="Frozen-encoder WCE warm-up"):
            self._set_stage("warmup")
            optimizer.zero_grad(set_to_none=True)
            started = time.monotonic()
            crops = [self._crop(index) for index in self._batch_order(step)]
            micro = int(training["micro_batch_size"])
            groups = [crops[index : index + micro] for index in range(0, len(crops), micro)]
            denominators = [
                sum(
                    float(weights_cpu[crop["target"][crop["target"] != 255]].sum())
                    for crop in group
                )
                for group in groups
            ]
            denominator = sum(denominators)
            if denominator <= 0:
                raise ValueError("Warm-up batch has no supervised pixels")
            total = 0.0
            for group, local_denominator in zip(groups, denominators, strict=True):
                if local_denominator == 0:
                    continue
                batch = stack_crops(group, self.device)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.device.type == "cuda",
                ):
                    logits = predict_raw_logits(self.model, batch)
                contribution = deterministic_masked_cross_entropy(
                    logits.float(),
                    batch["target"],
                    class_weights=weights,
                ) * (local_denominator / denominator)
                if not torch.isfinite(contribution):
                    raise FloatingPointError("Non-finite warm-up WCE before optimizer update")
                scaler.scale(contribution).backward()
                total += float(contribution.detach())
                del logits, batch, contribution
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                float(training["clip_grad_norm"]),
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()
            state["step"] = step + 1
            state["history"].append(
                {
                    "step": step + 1,
                    "train_WCE": total,
                    "grad_norm": float(gradient_norm),
                    "seconds": time.monotonic() - started,
                }
            )

            is_readiness = (step + 1) % interval == 0
            if is_readiness:
                report, per_class, permutation = self._warmup_readiness(step + 1)
                state["consecutive_ready"] = (
                    int(state["consecutive_ready"]) + 1 if report["ready"] else 0
                )
                report["consecutive_ready"] = int(state["consecutive_ready"])
                state["readiness"].append(report)
                state["validation"].append(
                    {"step": step + 1, **report["validation"], **report["flags"]}
                )
                per_class.to_csv(
                    self.run / "warmup_latest_validation_per_class.csv", index=False
                )
                atomic_json(
                    permutation, self.run / "warmup_latest_class_permutation.json"
                )
                print(
                    f"Warm-up step {step + 1}: mapped val F1="
                    f"{report['validation']['macro_F1_11']:.4f}, "
                    f"classes={report['validation']['predicted_classes']}/11, "
                    f"ready={report['ready']}, consecutive="
                    f"{state['consecutive_ready']}/{required_consecutive}"
                )
                if (
                    step + 1 >= minimum
                    and state["consecutive_ready"] >= required_consecutive
                ):
                    state["transition_ready"] = True
                    result = {
                        "stage": "WCE_warmup",
                        "step": step + 1,
                        "completed": True,
                        "transition_ready": True,
                        "selection_rule": warmup["selection"],
                        "consecutive_ready": int(state["consecutive_ready"]),
                        "method_hash": self.method_hash,
                    }
                    atomic_torch(
                        {
                            "role": "earliest_transition_ready",
                            "warmup_step": step + 1,
                            "model": self.model.state_dict(),
                            "method_hash": self.method_hash,
                            "readiness": report,
                            "permutation": permutation,
                            "pretrained_provenance": self.model.pretrained_provenance,
                        },
                        transition_path,
                    )
                    atomic_json(
                        {
                            "method_hash": self.method_hash,
                            "file": transition_path.name,
                            "sha256": sha256_file(transition_path),
                            "result": result,
                            "readiness": report,
                        },
                        transition_json,
                    )
            if (
                is_readiness
                or (step + 1) % int(training["save_interval_steps"]) == 0
                or step + 1 == end
                or state["transition_ready"]
            ):
                save_resume()
            if state["transition_ready"]:
                break

        blocked = bool(state["step"] >= maximum and not state["transition_ready"])
        result = {
            "stage": "WCE_warmup",
            "step": int(state["step"]),
            "target_step": int(end),
            "completed": bool(state["transition_ready"] or blocked),
            "transition_ready": bool(state["transition_ready"]),
            "blocked_at_maximum": blocked,
            "consecutive_ready": int(state["consecutive_ready"]),
            "method_hash": self.method_hash,
        }
        if state["transition_ready"]:
            atomic_json(result, self.run / "WARMUP_COMPLETE.json")
        elif blocked:
            atomic_json(result, self.run / "WARMUP_BLOCKED.json")
        else:
            atomic_json(result, self.run / "WARMUP_PROGRESS.json")
        return result

    def _load_transition_model(self):
        transition = self.run / "warmup_transition.pt"
        if not transition.exists():
            raise ValueError("Warm-up has not produced a transition-ready checkpoint")
        if self.model is None:
            self.load_model(pretrained=False)
        payload = torch.load(transition, map_location=self.device, weights_only=False)
        if payload["method_hash"] != self.method_hash:
            raise ValueError("Warm-up transition contract differs")
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.pretrained_provenance = payload["pretrained_provenance"]
        self._set_stage("dmi")
        return payload

    def smoke_test(self):
        report_path = self.run / "DMI_SMOKE_TEST.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report.get("method_hash") != self.method_hash:
                raise ValueError("Existing DMI smoke test belongs to another method")
            return report
        transition = self._load_transition_model()
        original_rng = self._capture_rng()
        self.model.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            diagnostics = self._exact_dmi_backward(self.batches[0], self._dmi_scaler())
            gradients = self._gradient_report()
            report = {
                "effective_batch_size": len(self.batches[0]),
                "two_pass_exact_DMI": True,
                "DMI": diagnostics.to_dict(),
                "gradients": gradients,
                "device": str(self.device),
                "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1e9
                if self.device.type == "cuda"
                else None,
                "warmup_transition_step": int(transition["warmup_step"]),
                "warmup_transition_sha256": sha256_file(self.run / "warmup_transition.pt"),
                "parameter_report": self.parameter_report(),
                "method_hash": self.method_hash,
            }
            if diagnostics.rank != 11 or not diagnostics.finite_loss or not gradients["finite"]:
                atomic_json(report, self.run / "DMI_SMOKE_TEST_FAILURE.json")
                raise ValueError("Invalid post-warm-up direct-DMI smoke test")
            atomic_json(report, report_path)
            return report
        finally:
            self.model.zero_grad(set_to_none=True)
            self._restore_rng(original_rng)

    def _checkpoint_payload(self, role, step, summary, permutation):
        return {
            "role": role,
            "dmi_step": int(step),
            "model": self.model.state_dict(),
            "method_hash": self.method_hash,
            "summary": summary,
            "permutation": permutation,
            "warmup_transition_sha256": sha256_file(self.run / "warmup_transition.pt"),
            "pretrained_provenance": self.model.pretrained_provenance,
        }

    def _save_selected_checkpoint(self, filename, role, step, summary, permutation):
        path = self.run / filename
        atomic_torch(self._checkpoint_payload(role, step, summary, permutation), path)
        atomic_json(
            {
                "role": role,
                "file": filename,
                "dmi_step": int(step),
                "summary": summary,
                "permutation": permutation,
            },
            path.with_suffix(".json"),
        )

    def _write_selection_manifest(self):
        roles = {
            "primary_macro_F1": self.run / "best_validation_macro_f1.pt",
            "secondary_DMI_loss": self.run / "best_validation_dmi_loss.pt",
        }
        payload = {
            "primary_role": "primary_macro_F1",
            "mapping_source": "fixed train-only permutation panel",
            "warmup_transition_sha256": sha256_file(self.run / "warmup_transition.pt"),
            "checkpoints": {},
        }
        for role, path in roles.items():
            if not path.exists():
                payload["checkpoints"][role] = {"available": False}
                continue
            metadata_path = path.with_suffix(".json")
            if not metadata_path.exists():
                raise ValueError(f"Checkpoint metadata is missing: {metadata_path.name}")
            metadata = json.loads(metadata_path.read_text())
            payload["checkpoints"][role] = {
                "available": True,
                "file": path.name,
                "sha256": sha256_file(path),
                "dmi_step": metadata["dmi_step"],
                "summary": metadata["summary"],
                "permutation": metadata["permutation"],
            }
        primary = payload["checkpoints"]["primary_macro_F1"]
        payload["primary_acceptance_passed"] = bool(
            primary.get("summary", {}).get("acceptance_passed", False)
        )
        atomic_json(payload, self.run / "MODEL_SELECTION.json")
        return payload

    def train(self, target_step=None):
        if self.device.type != "cuda" and target_step is None:
            raise RuntimeError("Full direct-DMI training requires a Colab GPU")
        if not (self.run / "warmup_transition.pt").exists():
            raise ValueError("Run warmup() until transition_ready=True before direct DMI")
        training = self.config["training"]
        latest = self.run / "dmi_latest_resume.pt"
        if latest.exists():
            if self.model is None:
                self.load_model(pretrained=False)
        else:
            self._load_transition_model()
        parameters = self._stage_parameters("dmi")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(training["decoder_lr"]),
            weight_decay=float(training["weight_decay"]),
        )
        scaler = self._dmi_scaler()
        state = {
            "step": 0,
            "best_macro_F1": -1.0,
            "best_validation_DMI": float("inf"),
            "patience_best": -1.0,
            "stale": 0,
            "history": [],
            "validation": [],
            "stopped": False,
        }
        if latest.exists():
            saved = torch.load(latest, map_location=self.device, weights_only=False)
            if saved["method_hash"] != self.method_hash:
                raise ValueError("Direct-DMI resume contract differs")
            self.model.load_state_dict(saved["model"], strict=True)
            self._set_stage("dmi")
            optimizer.load_state_dict(saved["optimizer"])
            scaler.load_state_dict(saved["scaler"])
            state = saved["state"]
            self.model.pretrained_provenance = saved["pretrained_provenance"]
            torch.set_rng_state(saved["rng_cpu"].cpu())
            if self.device.type == "cuda":
                torch.cuda.set_rng_state_all([value.cpu() for value in saved["rng_cuda"]])
            del saved

        maximum = int(training["maximum_steps"])
        end = maximum if target_step is None else min(maximum, int(target_step))

        def save_resume():
            atomic_torch(
                {
                    "stage": "direct_DMI",
                    "method_hash": self.method_hash,
                    "model": self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "state": state,
                    "warmup_transition_sha256": sha256_file(self.run / "warmup_transition.pt"),
                    "pretrained_provenance": self.model.pretrained_provenance,
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all()
                    if self.device.type == "cuda"
                    else [],
                },
                latest,
            )
            pd.DataFrame(state["history"]).to_csv(
                self.run / "dmi_training_history.csv", index=False
            )
            pd.DataFrame(state["validation"]).to_csv(
                self.run / "dmi_validation_history.csv", index=False
            )

        if state["step"] < end and not state["stopped"]:
            for step in tqdm(range(int(state["step"]), end), desc="Post-warm-up direct-DMI steps"):
                self._set_stage("dmi")
                optimizer.zero_grad(set_to_none=True)
                started = time.monotonic()
                diagnostics = self._exact_dmi_backward(self._batch_order(step), scaler)
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    parameters,
                    float(training["clip_grad_norm"]),
                    error_if_nonfinite=True,
                )
                scaler.step(optimizer)
                scaler.update()
                state["step"] = step + 1
                state["history"].append(
                    {
                        "step": step + 1,
                        "train_DMI_loss": diagnostics.loss,
                        "train_DMI_sign": diagnostics.sign,
                        "train_DMI_rank": diagnostics.rank,
                        "train_DMI_min_singular_value": diagnostics.min_singular_value,
                        "train_DMI_condition_number": diagnostics.condition_number,
                        "grad_norm": float(gradient_norm),
                        "seconds": time.monotonic() - started,
                    }
                )
                is_validation = (
                    (step + 1) % int(training["validation_interval_steps"]) == 0
                    or step + 1 == maximum
                    or step + 1 == end
                )
                if is_validation:
                    summary, per_class, permutation = validation(
                        self.model,
                        self.scene,
                        self.val,
                        self.manifest,
                        self.permutation_indices,
                        self.device,
                    )
                    summary["dmi_step"] = step + 1
                    state["validation"].append(summary)
                    score = float(summary["macro_F1_11"])
                    if score > state["best_macro_F1"]:
                        state["best_macro_F1"] = score
                        self._save_selected_checkpoint(
                            "best_validation_macro_f1.pt",
                            "best_validation_macro_f1",
                            step + 1,
                            summary,
                            permutation,
                        )
                    dmi_score = float(summary["validation_DMI_loss"])
                    if (
                        summary["validation_DMI_checkpoint_eligible"]
                        and dmi_score < state["best_validation_DMI"]
                    ):
                        state["best_validation_DMI"] = dmi_score
                        self._save_selected_checkpoint(
                            "best_validation_dmi_loss.pt",
                            "best_validation_dmi_loss",
                            step + 1,
                            summary,
                            permutation,
                        )
                    if score > state["patience_best"] + float(training["min_delta"]):
                        state["patience_best"] = score
                        state["stale"] = 0
                    else:
                        state["stale"] += 1
                    state["stopped"] = state["stale"] >= int(
                        training["patience_evaluations"]
                    )
                    per_class.to_csv(
                        self.run / "dmi_latest_validation_per_class.csv", index=False
                    )
                    atomic_json(
                        permutation, self.run / "dmi_latest_class_permutation.json"
                    )
                    print(
                        f"DMI step {step + 1}: mapped val F1={score:.4f}, "
                        f"IoU={summary['macro_IoU_11']:.4f}, "
                        f"DMI={dmi_score:.4f}, rank={summary['validation_DMI_rank']}, "
                        f"gate={summary['acceptance_passed']}"
                    )
                if (
                    is_validation
                    or (step + 1) % int(training["save_interval_steps"]) == 0
                    or step + 1 == end
                ):
                    save_resume()
                if state["stopped"]:
                    break

        complete = bool(state["stopped"] or state["step"] >= maximum)
        result = {
            "stage": "direct_DMI",
            "dmi_step": int(state["step"]),
            "target_step": int(end),
            "completed": complete,
            "early_stopped": bool(state["stopped"]),
            "best_macro_F1": float(state["best_macro_F1"]),
            "best_validation_DMI": (
                float(state["best_validation_DMI"])
                if np.isfinite(state["best_validation_DMI"])
                else None
            ),
            "method_hash": self.method_hash,
        }
        atomic_json(
            result,
            self.run / ("DMI_TRAINING_COMPLETE.json" if complete else "DMI_TRAINING_PROGRESS.json"),
        )
        if complete:
            result["model_selection"] = self._write_selection_manifest()
        return result

    def selected_checkpoint_table(self):
        rows = []
        for role, filename in [
            ("primary_macro_F1", "best_validation_macro_f1.pt"),
            ("secondary_DMI_loss", "best_validation_dmi_loss.pt"),
        ]:
            path = self.run / filename
            if not path.exists():
                continue
            metadata_path = path.with_suffix(".json")
            if not metadata_path.exists():
                raise ValueError(f"Checkpoint metadata is missing: {metadata_path.name}")
            payload = json.loads(metadata_path.read_text())
            rows.append(
                {
                    "role": role,
                    "file": filename,
                    "dmi_step": payload["dmi_step"],
                    **payload["summary"],
                }
            )
        return pd.DataFrame(rows)

    def map(self):
        if not (self.run / "DMI_TRAINING_COMPLETE.json").exists():
            raise ValueError("Complete direct-DMI training/model selection before the final map")
        chosen = self.run / "best_validation_macro_f1.pt"
        if not chosen.exists():
            raise ValueError("No macro-F1 checkpoint")
        if self.model is None:
            self.load_model(pretrained=False)
        payload = torch.load(chosen, map_location=self.device, weights_only=False)
        if payload["method_hash"] != self.method_hash:
            raise ValueError("Selected checkpoint has a different contract")
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.pretrained_provenance = payload["pretrained_provenance"]
        self._set_stage("dmi")
        if not payload["summary"].get("acceptance_passed", False):
            print("Acceptance gate failed: output is exploratory, not an accepted model.")
        return create_map(
            self.model,
            self.scene,
            self.run,
            chosen,
            payload,
            self.device,
        )
