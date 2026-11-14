#!/usr/bin/env python3
"""Generate the exact-final-top SCNN classifier-only B4xT4 regression."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_GENERATOR = ROOT / "scripts" / "generate_tb_unified_h40_final_composition_scnn_v1.py"
BASE_TB = ROOT / "tb" / "tb_unified_h40_final_composition_scnn_v1.sv"
OUTPUT_TB = ROOT / "tb" / "tb_unified_h40_final_composition_scnn_t4_semantics_v1.sv"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one occurrence of {old!r}, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    subprocess.run([sys.executable, str(BASE_GENERATOR)], cwd=ROOT, check=True)
    text = BASE_TB.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "module tb_unified_h40_final_composition_scnn_v1 #(",
        "module tb_unified_h40_final_composition_scnn_t4_semantics_v1 #(",
    )

    # One physical request contains four samples and four temporal steps. The
    # shared learner sees the flattened 16-row window, while the capture RAM
    # remains bounded to the declared B4xT4 paper profile.
    for old, new in (
        (".MAX_BATCH(1),", ".MAX_BATCH(16),"),
        (".MAX_TIMESTEPS(1),", ".MAX_TIMESTEPS(4),"),
        (".SAMPLE_WIDTH(1),", ".SAMPLE_WIDTH(4),"),
        (".BATCH_COUNT_WIDTH(1),", ".BATCH_COUNT_WIDTH(5),"),
        (".TIMESTEP_WIDTH(1),", ".TIMESTEP_WIDTH(2),"),
        (".i_batch_count(1'b1),", ".i_batch_count(5'd4),"),
        (".i_timestep(1'b0),", ".i_timestep(2'd0),"),
        (".i_timestep_count(1'b1),", ".i_timestep_count(3'd4),"),
        (".i_labels_flat(4'd0),", ".i_labels_flat(64'h0000_0000_0000_0210),"),
        ("if (logit_count == 1)", "if (logit_count == 16)"),
    ):
        text = replace_once(text, old, new)

    text = text.replace("local_wait < 200000", "local_wait < 4000000")
    text = replace_once(text, "watchdog == 800000", "watchdog == 8000000")

    sequence = r'''    initial begin
        repeat (4) tick();
        rst_n = 1'b1;
        tick();

        progress_stage = 1;
        for (int tile = 0; tile < 3; tile++) begin
            clear_work(tile, 4);
            clear_work(tile, 20);
        end

        progress_stage = 2;
        for (int addr = 0; addr < 18; addr++)
            load_parameter(CONV_BASE + addr,
                1 << (SCNN_WEIGHT_FRACTIONAL_BITS - 8));
        for (int addr = 0; addr < FEATURES * CLASSES; addr++)
            load_parameter(CLASSIFIER_BASE + addr, 0);

        progress_stage = 3;
        while (!shared_backend_quiescent) tick();
        parameter_owner = 3'b100;
        tick();
        initialize_system();
        if (!initialized || !classifier_cache_valid ||
                !active_classifier_pair || parameter_version != 5 ||
                model_version != 5 || adapt_version != 7 ||
                !versions_aligned || sticky_error || shared_backend_error)
            $fatal(1, "B4xT4 initialization mismatch init/cache/pair=%0b/%0b/%0b versions=%0d/%0d/%0d error=%0b/%0b",
                initialized, classifier_cache_valid, active_classifier_pair,
                parameter_version, model_version, adapt_version, sticky_error,
                shared_backend_error);

        progress_stage = 7;
        request_batch(1'b1, 1);
        tick();
        if (dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.u_cl.i_cfg_batch_count != 5'd4 ||
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.u_cl.i_cfg_timestep_count != 3'd4)
            $fatal(1, "B4xT4 learner configuration mismatch batch=%0d timesteps=%0d",
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.u_cl.i_cfg_batch_count,
                dut.u_system.g_scnn_client.u_train_infer.u_runtime.u_runtime.u_selected_layers.u_cl.i_cfg_timestep_count);
        if (graph_count != 16 || target_graph_count != 0 ||
                student_graph_count != 16 || commit_count != 1 ||
                classifier_write_count == 0 || conv_write_count != 0 ||
                model_version != 6 || adapt_version != 8 ||
                parameter_version != 6 || !versions_aligned ||
                !active_classifier_pair || !saw_conv_invalidate ||
                conv_cache_valid || logit_count != 16 ||
                feature_wave_count != 16 || surrogate_count != 0 ||
                sticky_error || shared_backend_error || array_tile_error != 0)
            $fatal(1, "B4xT4 learning mismatch graphs=%0d target/student=%0d/%0d writes=%0d/%0d commit=%0d versions=%0d/%0d/%0d pair=%0b cache=%0b logits=%0d feature/surr=%0d/%0d error=%0b/%0b tile=%h",
                graph_count, target_graph_count, student_graph_count,
                conv_write_count, classifier_write_count, commit_count,
                model_version, adapt_version, parameter_version,
                active_classifier_pair, conv_cache_valid, logit_count,
                feature_wave_count, surrogate_count, sticky_error,
                shared_backend_error, array_tile_error);

        progress_stage = 8;
        for (int tile = 0; tile < 3; tile++) begin
            clear_work(tile, 4);
            clear_work(tile, 20);
        end

        begin
            int writes_before;
            int commits_before;
            int fills_before;
            writes_before = conv_write_count + classifier_write_count;
            commits_before = commit_count;
            fills_before = weight_fill_count;
            request_batch(1'b0, 2);
            tick();
            if (graph_count != 32 || target_graph_count != 0 ||
                    student_graph_count != 32 ||
                    conv_write_count + classifier_write_count != writes_before ||
                    commit_count != commits_before ||
                    weight_fill_count <= fills_before || !conv_cache_valid ||
                    conv_cached_version != 6 || model_version != 6 ||
                    parameter_version != 6 || !versions_aligned ||
                    logit_count != 32 || feature_wave_count != 16 ||
                    sticky_error || shared_backend_error ||
                    array_tile_error != 0 || array_active_owner != 0 ||
                    !quiescent || array_tile_idle != 16'hffff)
                $fatal(1, "B4xT4 inference mismatch graphs=%0d writes=%0d/%0d commit=%0d fills=%0d/%0d cache=%0b v=%0d logits=%0d feature=%0d owner=%0d quiet=%0b error=%0b/%0b tile=%h",
                    graph_count, conv_write_count, classifier_write_count,
                    commit_count, fills_before, weight_fill_count,
                    conv_cache_valid, conv_cached_version, logit_count,
                    feature_wave_count, array_active_owner, quiescent,
                    sticky_error, shared_backend_error, array_tile_error);
        end

        if (inference_logits[0] == learned_logits[0] &&
                inference_logits[1] == learned_logits[1] &&
                inference_logits[2] == learned_logits[2])
            $fatal(1, "B4xT4 committed classifier update was not visible to the next forward window");

        if (!quiescent || !shared_backend_quiescent || shared_backend_error ||
                sticky_error || array_tile_error != 0 || array_active_owner != 0 ||
                array_tile_idle != 16'hffff)
            $fatal(1, "B4xT4 final quiescence mismatch owner=%0d quiet=%0b/%0b error=%0b/%0b tile=%h idle=%h",
                array_active_owner, quiescent, shared_backend_quiescent,
                sticky_error, shared_backend_error, array_tile_error,
                array_tile_idle);

        $display("FINAL_COMPOSITION_SCNN_B4T4_PASS exact_top=1 classifier_only=1 physical_batch=4 timesteps=4 flattened_rows=16 graph_order=sample_major commits=1 next_forward_visible=1 learning_off_writes=0");
        $finish;
    end'''
    text, count = re.subn(
        r"    initial begin\n.*?\n    end\nendmodule",
        sequence + "\nendmodule",
        text,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise RuntimeError("could not replace the generated test sequence")

    OUTPUT_TB.write_text(text, encoding="utf-8")
    print(f"generated {OUTPUT_TB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
