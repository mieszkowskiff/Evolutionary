import numpy as np
import struct
import os

# =============================================================================
# CONSTANTS (from constants.h)
# =============================================================================
MILIEU_SENSORS_N = 60
INTERNAL_SENSORS_N = 4
INPUT_NEURONS_N = MILIEU_SENSORS_N + INTERNAL_SENSORS_N  # 64
HIDDEN_NEURONS_N = 32
OUTPUT_NEURONS_N = 32
ACTIONS_N = OUTPUT_NEURONS_N  # one output neuron == one selectable action slot
ACTION_TYPES_N = 5

ACTION_NAMES = {0: 'MOVE', 1: 'EAT', 2: 'ATTACK', 3: 'REPRODUCE', 4: 'DRINK', -1: 'NONE'}


# =============================================================================
# FP8 E4M3 (NVIDIA __nv_fp8_e4m3 / e4m3fn) -> float32 decoding
# =============================================================================
def fp8_e4m3_to_float(raw_bytes):
    """Decode a buffer of __nv_fp8_e4m3 bytes into a float32 numpy array.

    Format: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits.
    No infinities. S.1111.111 is NaN. exponent==0 is subnormal.
    """
    b = np.frombuffer(raw_bytes, dtype=np.uint8)

    sign = (b >> 7) & 0x1
    exponent = (b >> 3) & 0xF
    mantissa = (b & 0x7).astype(np.float32)

    sign_mult = np.where(sign == 1, -1.0, 1.0).astype(np.float32)
    is_nan = (exponent == 15) & (mantissa == 7)
    is_subnormal = exponent == 0

    exp_val = exponent.astype(np.float32) - 7.0
    normal_val = (1.0 + mantissa / 8.0) * np.power(2.0, exp_val)
    subnormal_val = (mantissa / 8.0) * np.power(2.0, -6.0)

    val = np.where(is_subnormal, subnormal_val, normal_val)
    val = sign_mult * val
    val = np.where(is_nan, np.nan, val)
    return val.astype(np.float32)


# =============================================================================
# SINGLE RECORD READER
# =============================================================================
def read_record(f):
    """Read one tick's record from an open binary file handle positioned at
    the start of a record. Returns a dict, or None if at EOF.
    """
    header = f.read(4 * 4 + 3)  # 4 int32 + 3 bool(1 byte each)
    if len(header) < 4 * 4 + 3:
        return None  # EOF

    t, count, first_newborn_index, new_creatures_count = struct.unpack('iiii', header[:16])
    save_sensors_and_actions, save_neuron_values, save_network_weights = struct.unpack('???', header[16:19])

    record = {
        't': t,
        'count': count,
        'first_newborn_index': first_newborn_index,
        'new_creatures_count': new_creatures_count,
        'save_sensors_and_actions': save_sensors_and_actions,
        'save_neuron_values': save_neuron_values,
        'save_network_weights': save_network_weights,
    }

    # old creatures only
    record['chosen_action'] = np.frombuffer(
        f.read(first_newborn_index), dtype=np.int8
    ).copy()

    if save_neuron_values:
        input_layer_values = fp8_e4m3_to_float(
            f.read(INPUT_NEURONS_N * first_newborn_index)
        ).reshape(INPUT_NEURONS_N, first_newborn_index).T.copy()

        hidden_layer_values = fp8_e4m3_to_float(
            f.read(HIDDEN_NEURONS_N * first_newborn_index)
        ).reshape(HIDDEN_NEURONS_N, first_newborn_index).T.copy()

        output_layer_values = np.frombuffer(
            f.read(OUTPUT_NEURONS_N * first_newborn_index * 4), dtype=np.float32
        ).reshape(OUTPUT_NEURONS_N, first_newborn_index).T.copy()

        record['input_layer_values'] = input_layer_values    # (first_newborn_index, INPUT_NEURONS_N)
        record['hidden_layer_values'] = hidden_layer_values   # (first_newborn_index, HIDDEN_NEURONS_N)
        record['output_layer_values'] = output_layer_values   # (first_newborn_index, OUTPUT_NEURONS_N)

    # all creatures
    record['x'] = np.frombuffer(f.read(count * 4), dtype=np.uint32).copy()
    record['y'] = np.frombuffer(f.read(count * 4), dtype=np.uint32).copy()
    record['energy'] = np.frombuffer(f.read(count * 4), dtype=np.float32).copy()
    record['water'] = np.frombuffer(f.read(count * 4), dtype=np.float32).copy()
    record['ids'] = np.frombuffer(f.read(count * 8), dtype=np.int64).copy()

    # new creatures only
    if save_sensors_and_actions:
        sensor_x = np.empty((MILIEU_SENSORS_N, new_creatures_count), dtype=np.int8)
        sensor_y = np.empty((MILIEU_SENSORS_N, new_creatures_count), dtype=np.int8)
        sensor_type = np.empty((MILIEU_SENSORS_N, new_creatures_count), dtype=np.int8)
        for i in range(MILIEU_SENSORS_N):
            sensor_x[i] = np.frombuffer(f.read(new_creatures_count), dtype=np.int8)
            sensor_y[i] = np.frombuffer(f.read(new_creatures_count), dtype=np.int8)
            sensor_type[i] = np.frombuffer(f.read(new_creatures_count), dtype=np.int8)

        action_x = np.empty((ACTION_TYPES_N, new_creatures_count), dtype=np.int8)
        action_y = np.empty((ACTION_TYPES_N, new_creatures_count), dtype=np.int8)
        action_type = np.empty((ACTION_TYPES_N, new_creatures_count), dtype=np.int8)
        for i in range(ACTION_TYPES_N):
            action_x[i] = np.frombuffer(f.read(new_creatures_count), dtype=np.int8)
            action_y[i] = np.frombuffer(f.read(new_creatures_count), dtype=np.int8)
            action_type[i] = np.frombuffer(f.read(new_creatures_count), dtype=np.int8)

        record['sensor_x'] = sensor_x.T.copy()        # (new_creatures_count, MILIEU_SENSORS_N)
        record['sensor_y'] = sensor_y.T.copy()
        record['sensor_type'] = sensor_type.T.copy()
        record['action_x'] = action_x.T.copy()        # (new_creatures_count, ACTION_TYPES_N)
        record['action_y'] = action_y.T.copy()
        record['action_type'] = action_type.T.copy()

    if save_network_weights:
        first_matrix = np.empty((HIDDEN_NEURONS_N, INPUT_NEURONS_N, new_creatures_count), dtype=np.float32)
        second_matrix = np.empty((HIDDEN_NEURONS_N, OUTPUT_NEURONS_N, new_creatures_count), dtype=np.float32)
        bias = np.empty((HIDDEN_NEURONS_N, new_creatures_count), dtype=np.float32)

        for h in range(HIDDEN_NEURONS_N):
            for i in range(INPUT_NEURONS_N):
                first_matrix[h, i] = fp8_e4m3_to_float(f.read(new_creatures_count))
            for o in range(OUTPUT_NEURONS_N):
                second_matrix[h, o] = fp8_e4m3_to_float(f.read(new_creatures_count))
            bias[h] = fp8_e4m3_to_float(f.read(new_creatures_count))

        # -> (new_creatures_count, HIDDEN_NEURONS_N, INPUT_NEURONS_N / OUTPUT_NEURONS_N)
        record['first_matrix'] = np.transpose(first_matrix, (2, 0, 1)).copy()
        record['second_matrix'] = np.transpose(second_matrix, (2, 0, 1)).copy()
        record['bias'] = bias.T.copy()  # (new_creatures_count, HIDDEN_NEURONS_N)

    return record


# =============================================================================
# FULL-STREAM READING
# =============================================================================
# The simulation always writes to a single fixed location: build/save/, at
# the project root -- not per-experiment. Only one experiment's stream data
# exists at a time, so there's no run_dir/exp_NNNN indirection here.
BUILD_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build', 'save')
STREAM_FILENAMES = ["stream1.bin", "stream2.bin"]


def read_stream_records(stream_path):
    """Read every record out of a single stream file, in file order."""
    records = []
    if not os.path.exists(stream_path):
        return records

    with open(stream_path, 'rb') as f:
        while True:
            record = read_record(f)
            if record is None:
                break
            records.append(record)

    return records


def read_run(save_dir=None):
    """Read all records from both ping-pong streams, merged and sorted by
    tick. Returns a list of record dicts.

    save_dir: directory containing stream1.bin/stream2.bin. Defaults to the
    fixed build/save/ location (BUILD_SAVE_DIR) if not given.
    """
    if save_dir is None:
        save_dir = BUILD_SAVE_DIR

    all_records = []
    for filename in STREAM_FILENAMES:
        stream_path = os.path.join(save_dir, filename)
        all_records.extend(read_stream_records(stream_path))

    all_records.sort(key=lambda r: r['t'])
    return all_records


def available_ticks(save_dir=None):
    """Return sorted list of ticks present across both streams."""
    return [r['t'] for r in read_run(save_dir)]