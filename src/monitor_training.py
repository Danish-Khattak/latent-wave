from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


GIB = 1024**3
DASHBOARD_LINES = 6
ANSI_CLEAR_LINE = "\x1b[2K"


@dataclass
class RunState:
    model: str
    log_path: Path
    run_dir: Path
    offset: int = 0
    pending: bytes = b""
    progress: dict | None = None
    latest_epoch: dict | None = None
    last_validation: float | None = None
    seen_epochs: set[tuple[str, int, int]] = field(default_factory=set)
    new_summaries: list[dict] = field(default_factory=list)
    trainer_pid: int | None = None

    def poll_log(self) -> None:
        if not self.log_path.exists():
            return
        try:
            size = self.log_path.stat().st_size
            if size < self.offset:
                self.offset = 0
                self.pending = b""
                self.progress = None
                self.latest_epoch = None
                self.last_validation = None
            with self.log_path.open("rb") as stream:
                stream.seek(self.offset)
                data = stream.read()
                self.offset = stream.tell()
        except OSError:
            return
        if not data:
            return

        chunks = (self.pending + data).split(b"\n")
        self.pending = chunks.pop()
        for raw_line in chunks:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line.startswith("[progress] "):
                record = self._parse_record(line, "[progress] ")
                if record:
                    self.progress = record
                    validation = record.get("previous_validation_loss")
                    if validation is not None:
                        self.last_validation = float(validation)
            elif line.startswith("[epoch] "):
                record = self._parse_record(line, "[epoch] ")
                if not record:
                    continue
                self.latest_epoch = record
                validation = record.get("val_loss_mean")
                if validation is not None:
                    self.last_validation = float(validation)
                key = (str(record.get("architecture", "")), int(record.get("seed", 0)), int(record.get("epoch", 0)))
                if key not in self.seen_epochs:
                    self.seen_epochs.add(key)
                    self.new_summaries.append(record)

    @staticmethod
    def _parse_record(line: str, prefix: str) -> dict | None:
        try:
            result = json.loads(line[len(prefix):])
            return result if isinstance(result, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None

    def find_trainer_pid(self) -> int | None:
        """Find the root Python trainer for this output directory via /proc."""
        run_dir = os.fsencode(self.run_dir.name)
        matches: list[tuple[int, int, str]] = []
        try:
            entries = os.scandir("/proc")
        except OSError:
            self.trainer_pid = None
            return None
        with entries:
            for entry in entries:
                if not entry.name.isdigit():
                    continue
                try:
                    pid = int(entry.name)
                    cmdline = Path(entry.path, "cmdline").read_bytes()
                    if b"src/pretrain.py" not in cmdline or run_dir not in cmdline:
                        continue
                    executable = os.readlink(Path(entry.path, "exe"))
                    if "python" not in Path(executable).name.lower():
                        continue
                    stat = Path(entry.path, "stat").read_text(encoding="utf-8")
                    fields = stat[stat.rfind(")") + 2:].split()
                    process_state = fields[0]
                    parent_pid = int(fields[1])
                    if process_state not in {"Z", "X"}:
                        matches.append((pid, parent_pid, process_state))
                except (OSError, ValueError, IndexError):
                    continue
        if not matches:
            self.trainer_pid = None
            return None
        matching_pids = {pid for pid, _, _ in matches}
        roots = [pid for pid, parent_pid, _ in matches if parent_pid not in matching_pids]
        self.trainer_pid = min(roots or [pid for pid, _, _ in matches])
        return self.trainer_pid

    def status(self) -> str:
        if self.trainer_pid is not None:
            return f"RUNNING pid={self.trainer_pid}"
        if self.latest_epoch:
            epoch = int(self.latest_epoch.get("epoch", 0))
            total = int(self.latest_epoch.get("total_epochs", 0))
            if total and epoch >= total:
                return "COMPLETE"
            return f"STOPPED after epoch {epoch}"
        if self.progress:
            return "STOPPED (progress logged)"
        return "NOT STARTED / no log record"

    def current_record(self) -> tuple[dict | None, bool]:
        """Return the current progress record and whether it is an epoch summary."""
        progress = self.progress
        epoch_record = self.latest_epoch
        if not progress:
            return epoch_record, bool(epoch_record)
        if not epoch_record:
            return progress, False
        p_epoch, e_epoch = int(progress.get("epoch", 0)), int(epoch_record.get("epoch", 0))
        p_step, e_step = int(progress.get("step", 0)), int(epoch_record.get("step", 0))
        if p_epoch > e_epoch or (p_epoch == e_epoch and p_step > e_step):
            return progress, False
        return epoch_record, True


def duration(value: object) -> str:
    if value is None:
        return "not in log"
    try:
        seconds = max(0, int(float(value)))
    except (TypeError, ValueError):
        return "not in log"
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d{hours:02}h"
    if hours:
        return f"{hours}h{minutes:02}m"
    return f"{minutes}m{seconds:02}s"


def value(value_obj: object, fmt: str = ".4f") -> str:
    if value_obj is None:
        return "not in log"
    try:
        return format(float(value_obj), fmt)
    except (TypeError, ValueError):
        return "not in log"


def memory(record: dict | None) -> str:
    if not record:
        return "not in log"
    allocated = record.get("gpu_memory_allocated_bytes")
    reserved = record.get("gpu_memory_reserved_bytes")
    if allocated is None and reserved is None:
        return "not in log"
    a = f"{float(allocated) / GIB:.2f}" if allocated is not None else "not in log"
    r = f"{float(reserved) / GIB:.2f}" if reserved is not None else "not in log"
    return f"{a}/{r} GiB allocated/reserved"


def progress_details(state: RunState) -> tuple[int | None, int | None, int | None, dict | None, bool]:
    record, is_completed = state.current_record()
    if not record:
        return None, None, None, None, False
    epoch = int(record.get("epoch", 0))
    total_epochs = int(record.get("total_epochs", 0)) or None
    total_steps = int(record.get("total_steps", 0)) or None
    steps_per_epoch = (total_steps // total_epochs) if total_steps and total_epochs else None
    if is_completed:
        step_in_epoch = steps_per_epoch
    elif steps_per_epoch:
        step_in_epoch = int(record.get("step", 0)) - ((epoch - 1) * steps_per_epoch)
        step_in_epoch = max(0, min(step_in_epoch, steps_per_epoch))
    else:
        step_in_epoch = None
    return epoch, total_epochs, step_in_epoch, record, is_completed


def progress_bar(step: int | None, total: int | None, width: int = 20) -> str:
    if step is None or not total:
        return "[not in log]"
    fraction = max(0.0, min(1.0, step / total))
    filled = min(width, int(fraction * width))
    return f"[{'█' * filled}{'·' * (width - filled)}] {fraction * 100:5.1f}%"


def dashboard_for(state: RunState) -> list[str]:
    status = state.status()
    epoch, total_epochs, step, record, is_completed = progress_details(state)
    total_steps = None
    if record:
        total_steps = (int(record.get("total_steps", 0)) // int(record.get("total_epochs", 1))) or None
    epoch_text = f"{epoch}/{total_epochs}" if epoch is not None and total_epochs is not None else "not in log"
    step_text = f"{step}/{total_steps}" if step is not None and total_steps is not None else "not in log"
    bar = progress_bar(step, total_steps)
    if status.startswith("STOPPED") and epoch is not None and is_completed:
        status += " (last epoch complete)"
    if status.startswith("RUNNING") and record is None:
        status += " (waiting for log)"

    progress_line = (
        f"{state.model} seed {record.get('seed', 'not in log') if record else 'not in log'} | {status} | "
        f"epoch {epoch_text} | step {step_text} | {bar}"
    )
    if record:
        train_loss = value(record.get("train_loss_mean"))
        validation = state.last_validation
        lr = value(record.get("learning_rate"), ".3e")
        ema = value(record.get("ema_momentum"), ".6f")
        elapsed = duration(record.get("elapsed_seconds"))
        eta = duration(record.get("eta_seconds"))
        gpu = memory(record)
    else:
        train_loss = validation = lr = ema = elapsed = eta = gpu = "not in log"
    return [
        progress_line,
        f"    train {train_loss} | val {value(validation)} | LR {lr} | EMA {ema}",
        f"    elapsed {elapsed} | ETA {eta} | GPU {gpu}",
    ]


def summary_line(state: RunState, record: dict) -> str:
    arch = str(record.get("architecture", state.model)).upper()
    seed = record.get("seed", "not in log")
    epoch = record.get("epoch", "not in log")
    total = record.get("total_epochs", "not in log")
    train = value(record.get("train_loss_mean"))
    val = value(record.get("val_loss_mean"))
    took = duration(record.get("epoch_elapsed_seconds"))
    throughput = value(record.get("source_examples_per_second"), ".2f")
    return f"SUMMARY {arch} seed {seed} | epoch {epoch}/{total} | train {train} | val {val} | {took} | {throughput} source ex/s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only live monitor for LatentWave pretraining console logs.")
    parser.add_argument("--mamba-log", default="../runs/latent-wave/full_runs/mamba2_prose_seed0/train_console.log")
    parser.add_argument("--mamba-run-dir", default="../runs/latent-wave/full_runs/mamba2_prose_seed0")
    parser.add_argument("--vit-log", default="../runs/latent-wave/full_runs/vit_prose_seed0/train_console.log")
    parser.add_argument("--vit-run-dir", default="../runs/latent-wave/full_runs/vit_prose_seed0")
    parser.add_argument("--refresh", type=float, default=2.0, help="Log poll interval in seconds (default: 2).")
    args = parser.parse_args()
    if args.refresh <= 0:
        parser.error("--refresh must be greater than zero")

    states = [
        RunState("Mamba-2", Path(args.mamba_log).expanduser().resolve(), Path(args.mamba_run_dir).expanduser().resolve()),
        RunState("ViT", Path(args.vit_log).expanduser().resolve(), Path(args.vit_run_dir).expanduser().resolve()),
    ]
    for state in states:
        state.poll_log()
        state.find_trainer_pid()

    sys.stdout.write("LatentWave training monitor — read-only; Ctrl-C exits this monitor only.\n")
    for state in states:
        for record in state.new_summaries:
            sys.stdout.write(summary_line(state, record) + "\n")
        state.new_summaries.clear()
    lines = [line for state in states for line in dashboard_for(state)]
    for line in lines:
        sys.stdout.write(line + "\n")
    sys.stdout.flush()

    try:
        last_process_scan = time.monotonic()
        while True:
            time.sleep(args.refresh)
            summaries: list[tuple[RunState, dict]] = []
            changed = False
            for state in states:
                old_offset = state.offset
                state.poll_log()
                changed = changed or state.offset != old_offset
                summaries.extend((state, record) for record in state.new_summaries)
                state.new_summaries.clear()
            if time.monotonic() - last_process_scan >= max(2.0, args.refresh):
                for state in states:
                    old_pid = state.trainer_pid
                    state.find_trainer_pid()
                    changed = changed or state.trainer_pid != old_pid
                last_process_scan = time.monotonic()

            if not changed and not summaries:
                continue
            if summaries:
                sys.stdout.write(f"\x1b[{DASHBOARD_LINES}A")
                for state, record in summaries:
                    sys.stdout.write("\r" + ANSI_CLEAR_LINE + summary_line(state, record) + "\n")
            else:
                sys.stdout.write(f"\x1b[{DASHBOARD_LINES}A")
            for line in (line for state in states for line in dashboard_for(state)):
                sys.stdout.write("\r" + ANSI_CLEAR_LINE + line + "\n")
            sys.stdout.flush()
    except KeyboardInterrupt:
        sys.stdout.write("\nMonitor exited; training processes were not signaled.\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
