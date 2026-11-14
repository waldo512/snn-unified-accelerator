from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_TB = ROOT / "tb/tb_unified_tp_c17_shared_h40_fc_srnn_integration_v1.sv"
OUTPUT_TB = ROOT / "tb/tb_unified_h40_final_composition_v1.sv"
WRAPPER = ROOT / "rtl/top/unified_h40_final_composition_top_v1.sv"


def balanced_end(text: str, start: int) -> int:
    opening = text[start]
    closing = {"(": ")", "[": "]", "{": "}"}[opening]
    depth = 0
    for index in range(start, len(text)):
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"unterminated delimiter at {start}")


def split_top_level(blob: str) -> list[str]:
    items: list[str] = []
    start = 0
    round_depth = square_depth = brace_depth = 0
    for index, char in enumerate(blob):
        if char == "(":
            round_depth += 1
        elif char == ")":
            round_depth -= 1
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        elif char == "," and not (round_depth or square_depth or brace_depth):
            items.append(blob[start:index].strip())
            start = index + 1
    tail = blob[start:].strip()
    if tail:
        items.append(tail)
    return items


def parse_named_connections(blob: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in split_top_level(blob):
        match = re.match(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", item, flags=re.S)
        if not match:
            raise ValueError(f"bad named connection: {item!r}")
        expression_start = item.index("(", match.end() - 1)
        expression_end = balanced_end(item, expression_start)
        result[match.group(1)] = item[expression_start + 1:expression_end].strip()
    return result


def instance_span(text: str, module: str, instance: str) -> tuple[int, int, dict[str, str], dict[str, str]]:
    start_match = re.search(rf"\b{re.escape(module)}\b\s*#\s*\(", text)
    if not start_match:
        raise ValueError(f"missing instance module {module}")
    parameter_start = text.index("(", start_match.start())
    parameter_end = balanced_end(text, parameter_start)
    instance_match = re.search(rf"\)\s*{re.escape(instance)}\s*\(", text[parameter_end:])
    if not instance_match:
        raise ValueError(f"missing instance {instance}")
    absolute_match_start = parameter_end + instance_match.start()
    port_start = text.index("(", absolute_match_start + 1)
    port_end = balanced_end(text, port_start)
    end = port_end + 1
    while end < len(text) and text[end].isspace():
        end += 1
    if text[end] != ";":
        raise ValueError(f"missing semicolon after {instance}")
    end += 1
    return (
        start_match.start(), end,
        parse_named_connections(text[parameter_start + 1:parameter_end]),
        parse_named_connections(text[port_start + 1:port_end]),
    )


def wrapper_port_names(text: str) -> list[str]:
    match = re.search(r"\bmodule\s+unified_h40_final_composition_top_v1\b", text)
    if not match:
        raise ValueError("wrapper module missing")
    cursor = match.end()
    parameter_start = text.index("(", text.index("#", cursor))
    parameter_end = balanced_end(text, parameter_start)
    port_start = text.index("(", parameter_end + 1)
    port_end = balanced_end(text, port_start)
    names: list[str] = []
    for declaration in split_top_level(text[port_start + 1:port_end]):
        name_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", declaration)
        if not name_match:
            raise ValueError(f"bad wrapper port: {declaration!r}")
        names.append(name_match.group(1))
    return names


text = SOURCE_TB.read_text(encoding="utf-8")
bridge_span = instance_span(
    text, "unified_srnn_backbone_m2_phase_bridge_v1", "u_srnn_phase_bridge"
)
plane_span = instance_span(
    text, "unified_tp_c17_learning_parameter_plane_v1", "u_plane"
)
system_span = instance_span(
    text, "unified_scnn_h40_array16_selected_layers_train_infer_system_v1", "u_system"
)

_, _, bridge_parameters, bridge_ports = bridge_span
_, _, plane_parameters, plane_ports = plane_span
_, _, system_parameters, system_ports = system_span

phase_to_bridge = {
    "i_srnn_phase_abort": "i_abort",
    "i_srnn_phase_capture_cfg_valid": "i_capture_cfg_valid",
    "o_srnn_phase_capture_cfg_ready": "o_capture_cfg_ready",
    "i_srnn_phase_record_count": "i_capture_record_count",
    "i_srnn_phase_start_valid": "i_sequence_start_valid",
    "o_srnn_phase_start_ready": "o_sequence_start_ready",
    "i_srnn_phase_learning_enable": "i_include_target_path",
    "i_srnn_phase_target_valid": "i_target_valid",
    "o_srnn_phase_target_ready": "o_target_ready",
    "i_srnn_phase_target_sample_id": "i_target_sample_id",
    "i_srnn_phase_target_backbone_id": "i_target_backbone_id",
    "i_srnn_phase_target_q14": "i_target_q14",
    "i_srnn_phase_target_last": "i_target_last",
    "i_srnn_phase_target_done": "i_target_done",
    "i_srnn_phase_target_error": "i_target_error",
    "o_srnn_phase_capture_done": "o_capture_done",
    "o_srnn_phase_capture_complete": "o_capture_complete",
    "o_srnn_phase_delivery_done": "o_delivery_done",
    "o_srnn_phase_busy": "o_busy",
    "o_srnn_phase_idle": "o_idle",
    "o_srnn_phase_sticky_error": "o_sticky_error",
}


def old_plane_name(wrapper_name: str) -> str | None:
    if wrapper_name in {"i_parameter_owner", "i_topology_learning_enable"}:
        return wrapper_name
    if wrapper_name.startswith(("i_store_", "o_store_", "o_backend_", "o_owner_")):
        return wrapper_name
    if wrapper_name.startswith("i_learning_"):
        return "i_" + wrapper_name[len("i_learning_"):]
    if wrapper_name.startswith("o_learning_"):
        return "o_" + wrapper_name[len("o_learning_"):]
    return None


connections: list[tuple[str, str]] = []
for wrapper_name in wrapper_port_names(WRAPPER.read_text(encoding="utf-8")):
    expression: str | None = None
    if wrapper_name in {"i_clk", "i_rst_n", "i_clear_error"}:
        expression = system_ports.get(wrapper_name, plane_ports.get(wrapper_name))
    elif wrapper_name in phase_to_bridge:
        expression = bridge_ports.get(phase_to_bridge[wrapper_name])
    else:
        plane_name = old_plane_name(wrapper_name)
        if plane_name is not None:
            expression = plane_ports.get(plane_name)
        if expression is None:
            expression = system_ports.get(wrapper_name)
    if expression is not None:
        connections.append((wrapper_name, expression))

wrapper_parameter_names = set(system_parameters) | set(plane_parameters)
forced_parameters = {
    "USE_INTERNAL_SCNN_PARAMETER_STORE",
    "ENABLE_SRNN_M2_H40_CLIENT",
    "ENABLE_SRNN_GROUP_STREAMER",
    "USE_SHARED_SRNN_H40_FORWARD",
}
combined_parameters: dict[str, str] = {}
for name, expression in system_parameters.items():
    if name in wrapper_parameter_names and name not in forced_parameters:
        combined_parameters[name] = expression
for name, expression in plane_parameters.items():
    if (name in wrapper_parameter_names and name not in combined_parameters and
            name not in forced_parameters):
        combined_parameters[name] = expression

replacement: list[str] = [
    "    // Generated compatibility instantiation: the original directed",
    "    // integration stimulus now reaches the production composition top.",
    "    unified_h40_final_composition_top_v1 #(",
]
parameter_items = list(combined_parameters.items())
for index, (name, expression) in enumerate(parameter_items):
    replacement.append(
        f"        .{name}({expression})" + ("," if index + 1 < len(parameter_items) else "")
    )
replacement.append("    ) u_final (")
for index, (name, expression) in enumerate(connections):
    replacement.append(
        f"        .{name}({expression})" + ("," if index + 1 < len(connections) else "")
    )
replacement.append("    );")
replacement.extend([
    "",
    "    // Compatibility observation aliases for the original scoreboards.",
    "    assign srnn_start = u_final.link_srnn_plane_timestep_start;",
    "    assign srnn_start_ready = u_final.link_srnn_plane_timestep_start_ready;",
    "    assign srnn_backbone_ready = u_final.link_srnn_plane_backbone_ready;",
    "    assign fc_student_trace_valid = u_final.link_fc_student_trace_valid;",
    "    assign fc_student_trace_ready = u_final.link_fc_student_trace_ready;",
    "    assign fc_student_trace_sample = u_final.link_fc_student_trace_sample;",
    "    assign fc_student_trace_hidden = u_final.link_fc_student_trace_hidden;",
    "    assign fc_student_trace_q016 = u_final.link_fc_student_trace_q016;",
    "    assign fc_logit_row_valid = u_final.link_fc_logit_row_valid;",
    "    assign fc_logit_row_ready = u_final.link_fc_logit_row_ready;",
    "    assign fc_logit_sample = u_final.link_fc_logit_sample;",
    "    assign fc_logit_label = u_final.link_fc_logit_label;",
    "    assign fc_logit_valid = u_final.link_fc_logit_valid;",
    "    assign fc_logit_ready = u_final.link_fc_logit_ready;",
    "    assign fc_logit_q6 = u_final.link_fc_logit_q6;",
    "    assign fc_cache_forward_read_valid = u_final.link_fc_forward_read_valid;",
    "    assign fc_cache_forward_read_addr = u_final.link_fc_forward_read_addr;",
    "    assign fc_cache_forward_read_tag = u_final.link_fc_forward_read_tag;",
    "    assign fc_cache_forward_read_ready = u_final.link_fc_forward_read_ready;",
    "    assign fc_cache_forward_response_valid = u_final.link_fc_forward_response_valid;",
    "    assign fc_cache_forward_response_ready = u_final.link_fc_forward_response_ready;",
    "    assign fc_cache_forward_response_tag = u_final.link_fc_forward_response_tag;",
    "    assign fc_cache_forward_response_code = u_final.link_fc_forward_response_code;",
    "    assign fc_cache_forward_response_error = u_final.link_fc_forward_response_error;",
    "    assign plane_fc_forward_read_valid = u_final.link_fc_forward_read_valid;",
    "    assign plane_fc_forward_read_ready = u_final.link_fc_forward_read_ready;",
    "    assign plane_fc_forward_read_addr = u_final.link_fc_forward_read_addr;",
    "    assign plane_fc_forward_read_tag = u_final.link_fc_forward_read_tag;",
    "    assign plane_fc_forward_response_valid = u_final.link_fc_forward_response_valid;",
    "    assign plane_fc_forward_response_ready = u_final.link_fc_forward_response_ready;",
    "    assign plane_fc_forward_response_tag = u_final.link_fc_forward_response_tag;",
    "    assign plane_fc_forward_response_code = u_final.link_fc_forward_response_code;",
    "    assign plane_fc_forward_response_error = u_final.link_fc_forward_response_error;",
    "    assign h40_cfg_valid = u_final.link_plane_to_system_srnn_h40_cfg_valid;",
    "    assign h40_cfg_ready = u_final.link_system_to_plane_srnn_h40_cfg_ready;",
    "    assign h40_cfg_error = u_final.link_system_to_plane_srnn_h40_cfg_error;",
    "    assign h40_cfg_batch_count = u_final.link_plane_to_system_srnn_h40_cfg_batch_count;",
    "    assign h40_cfg_backbone_count = u_final.link_plane_to_system_srnn_h40_cfg_backbone_count;",
    "    assign h40_cfg_hidden_count = u_final.link_plane_to_system_srnn_h40_cfg_hidden_count;",
    "    assign h40_cfg_class_count = u_final.link_plane_to_system_srnn_h40_cfg_class_count;",
    "    assign h40_cfg_wh1_base = u_final.link_plane_to_system_srnn_h40_cfg_wh1_parameter_base;",
    "    assign h40_cfg_wout_base = u_final.link_plane_to_system_srnn_h40_cfg_wout_parameter_base;",
    "    assign h40_prepare_valid = u_final.link_plane_to_system_srnn_h40_prepare_valid;",
    "    assign h40_prepare_ready = u_final.link_system_to_plane_srnn_h40_prepare_ready;",
    "    assign h40_prepare_version = u_final.link_plane_to_system_srnn_h40_prepare_version;",
    "    assign h40_prepare_done = u_final.link_system_to_plane_srnn_h40_prepare_done;",
    "    assign h40_cache_valid = u_final.link_system_to_plane_srnn_h40_cache_valid;",
    "    assign h40_cache_invalidate = u_final.link_plane_to_system_srnn_h40_cache_invalidate;",
    "    assign h40_forward_read_valid = u_final.link_system_to_plane_srnn_h40_forward_read_valid;",
    "    assign h40_forward_read_ready = u_final.link_plane_to_system_srnn_h40_forward_read_ready;",
    "    assign h40_forward_read_addr = u_final.link_system_to_plane_srnn_h40_forward_read_addr;",
    "    assign h40_forward_read_tag = u_final.link_system_to_plane_srnn_h40_forward_read_tag;",
    "    assign h40_forward_response_valid = u_final.link_plane_to_system_srnn_h40_forward_response_valid;",
    "    assign h40_forward_response_ready = u_final.link_system_to_plane_srnn_h40_forward_response_ready;",
    "    assign h40_forward_response_tag = u_final.link_plane_to_system_srnn_h40_forward_response_tag;",
    "    assign h40_forward_response_code = u_final.link_plane_to_system_srnn_h40_forward_response_code;",
    "    assign h40_forward_response_error = u_final.link_plane_to_system_srnn_h40_forward_response_error;",
    "    assign h40_timestep_start = u_final.link_plane_to_system_srnn_h40_timestep_start;",
    "    assign h40_timestep_start_ready = u_final.link_system_to_plane_srnn_h40_timestep_start_ready;",
    "    assign h40_start_timestep_id = u_final.link_plane_to_system_srnn_h40_timestep_id;",
    "    assign h40_sequence_version = u_final.link_plane_to_system_srnn_h40_sequence_version;",
    "    assign h40_include_target_path = u_final.link_plane_to_system_srnn_h40_include_target_path;",
    "    assign h40_observation_valid = u_final.link_plane_to_system_srnn_h40_observation_valid;",
    "    assign h40_observation_ready = u_final.link_system_to_plane_srnn_h40_observation_ready;",
    "    assign h40_observation_path = u_final.link_plane_to_system_srnn_h40_observation_path;",
    "    assign h40_observation_sample = u_final.link_plane_to_system_srnn_h40_observation_sample_id;",
    "    assign h40_observation_backbone = u_final.link_plane_to_system_srnn_h40_observation_backbone_id;",
    "    assign h40_observation_q14 = u_final.link_plane_to_system_srnn_h40_observation_q14;",
    "    assign h40_observation_last = u_final.link_plane_to_system_srnn_h40_observation_last;",
    "    assign h40_observation_done = u_final.link_plane_to_system_srnn_h40_observation_done;",
    "    assign h40_hidden_valid = u_final.link_system_to_plane_srnn_h40_hidden_valid;",
    "    assign h40_hidden_ready = u_final.link_plane_to_system_srnn_h40_hidden_ready;",
    "    assign h40_hidden_path = u_final.link_system_to_plane_srnn_h40_hidden_path;",
    "    assign h40_hidden_sample = u_final.link_system_to_plane_srnn_h40_hidden_sample_id;",
    "    assign h40_hidden_id = u_final.link_system_to_plane_srnn_h40_hidden_id;",
    "    assign h40_hidden_pre = u_final.link_system_to_plane_srnn_h40_hidden_membrane_pre_q14;",
    "    assign h40_hidden_next = u_final.link_system_to_plane_srnn_h40_hidden_membrane_next_q14;",
    "    assign h40_hidden_spike = u_final.link_system_to_plane_srnn_h40_hidden_spike;",
    "    assign h40_hidden_trace = u_final.link_system_to_plane_srnn_h40_hidden_trace_q14;",
    "    assign h40_hidden_feature = u_final.link_system_to_plane_srnn_h40_hidden_output_feature_q14;",
    "    assign h40_hidden_last = u_final.link_system_to_plane_srnn_h40_hidden_last;",
    "    assign h40_logit_valid = u_final.link_system_to_plane_srnn_h40_logit_valid;",
    "    assign h40_logit_ready = u_final.link_plane_to_system_srnn_h40_logit_ready;",
    "    assign h40_logit_sample = u_final.link_system_to_plane_srnn_h40_logit_sample_id;",
    "    assign h40_logit_class = u_final.link_system_to_plane_srnn_h40_logit_class_id;",
    "    assign h40_logit_q28 = u_final.link_system_to_plane_srnn_h40_logit_q28;",
    "    assign h40_logit_last = u_final.link_system_to_plane_srnn_h40_logit_last;",
    "    assign h40_result_timestep_id = u_final.link_system_to_plane_srnn_h40_timestep_id;",
    "    assign h40_timestep_done = u_final.link_system_to_plane_srnn_h40_timestep_done;",
    "    assign h40_busy = u_final.link_system_to_plane_srnn_h40_busy;",
    "    assign h40_sticky_error = u_final.link_system_to_plane_srnn_h40_sticky_error;",
    "    assign h40_protocol_error = u_final.link_system_to_plane_srnn_h40_protocol_error;",
    "    assign h40_overflow = u_final.link_system_to_plane_srnn_h40_overflow;",
])

start = bridge_span[0]
end = system_span[1]
if not (start < plane_span[0] < system_span[0] < end):
    raise ValueError("source instances are not in expected order")
text = text[:start] + "\n".join(replacement) + text[end:]
text = re.sub(
    r"\n\s*assign plane_fc_forward_read_valid = fc_cache_forward_read_valid;.*?"
    r"assign fc_cache_forward_response_error = plane_fc_forward_response_error;\s*\n",
    "\n    // FC cache/parameter links are internal to u_final.\n",
    text,
    flags=re.S,
)
text = text.replace(
    "module tb_unified_tp_c17_shared_h40_fc_srnn_integration_v1;",
    "module tb_unified_h40_final_composition_v1;",
)
text = text.replace(
    "localparam bit ENABLE_SRNN_WH1_LEARNING = 1'b1;",
    "localparam bit ENABLE_SRNN_WH1_LEARNING = 1'b0;",
)
text = text.replace("u_system.", "u_final.u_system.")
text = text.replace("u_plane.", "u_final.u_parameter_plane.")
text = text.replace(
    "u_final.u_system.u_srnn_backbone_cache_refill.",
    "u_final.u_system.g_srnn_backbone_refill.u_srnn_backbone_cache_refill.",
)
text = text.replace(
    "u_final.u_system.u_shared_weight_cache_refill.u_fc_refill.",
    "u_final.u_system.u_shared_weight_cache_refill.g_fc_refill.u_fc_refill.",
)
text = text.replace(
    "u_final.u_parameter_plane.u_srnn_m2.",
    "u_final.u_parameter_plane.g_srnn_client.u_srnn_m2.",
)
OUTPUT_TB.write_text(text, encoding="utf-8")
print(f"generated {OUTPUT_TB} with {len(connections)} connected wrapper ports")
