import numpy as np
import struct
import os

BUILD_DIR = os.path.join(os.path.dirname(__file__), '..', 'build')

# =============================================================================
# FILE FORMATS
# =============================================================================
#
# map_XXXXXX.bin
# --------------
#   [int32]   width
#   [int32]   height
#   [float32] food     -- shape: (height, width), row-major
#   [float32] danger   -- shape: (height, width), row-major
#   [float32] creature -- shape: (height, width), row-major
#
# creatures_XXXXXX.bin
# --------------------
#   [int32]  count
#   [int32]  sensors_n
#   [int32]  actions_n
#   [uint32] x             -- shape: (count,)
#   [uint32] y             -- shape: (count,)
#   [float32] energy       -- shape: (count,)
#   [int64]  ids           -- shape: (count,)
#   [int8]   chosen_action -- shape: (count,)  -1 = no action
#   [int8]   sensor_x      -- shape: (count, sensors_n)
#   [int8]   sensor_y      -- shape: (count, sensors_n)
#   [int8]   sensor_type   -- shape: (count, sensors_n)
#   [int8]   action_x      -- shape: (count, actions_n)
#   [int8]   action_y      -- shape: (count, actions_n)
#   [int8]   action_type   -- shape: (count, actions_n)
#
# Action type values:
#   0 = MOVE, 1 = EAT, 2 = ATTACK, 3 = REPRODUCE, -1 = none
#
# =============================================================================


ACTION_NAMES = {0: 'MOVE', 1: 'EAT', 2: 'ATTACK', 3: 'REPRODUCE', -1: 'NONE'}


def read_map(tick):
    path = os.path.join(BUILD_DIR, f'map_{tick:06d}.bin')
    with open(path, 'rb') as f:
        width, height = struct.unpack('ii', f.read(8))
        n = width * height
        food     = np.frombuffer(f.read(n * 4), dtype=np.float32).reshape(height, width).copy()
        danger   = np.frombuffer(f.read(n * 4), dtype=np.float32).reshape(height, width).copy()
        creature = np.frombuffer(f.read(n * 4), dtype=np.float32).reshape(height, width).copy()
    return {
        'width': width,
        'height': height,
        'food': food,       # (height, width) float32
        'danger': danger,   # (height, width) float32
        'creature': creature,  # (height, width) float32
    }


def read_creatures(tick):
    path = os.path.join(BUILD_DIR, f'creatures_{tick:06d}.bin')
    with open(path, 'rb') as f:
        count, sensors_n, actions_n = struct.unpack('iii', f.read(12))

        x             = np.frombuffer(f.read(count * 4),              dtype=np.uint32).copy()
        y             = np.frombuffer(f.read(count * 4),              dtype=np.uint32).copy()
        energy        = np.frombuffer(f.read(count * 4),              dtype=np.float32).copy()
        ids           = np.frombuffer(f.read(count * 8),              dtype=np.int64).copy()
        chosen_action = np.frombuffer(f.read(count),                  dtype=np.int8).copy()

        # SoA layout: [creature_0_sensor_0, creature_0_sensor_1, ..., creature_1_sensor_0, ...]
        # Each sensor/action array is stored as (count, n) row-major:
        # [creature_0_slot_0, creature_0_slot_1, ..., creature_1_slot_0, ...]
        sensor_x    = np.frombuffer(f.read(count * sensors_n), dtype=np.int8).reshape(count, sensors_n).copy()
        sensor_y    = np.frombuffer(f.read(count * sensors_n), dtype=np.int8).reshape(count, sensors_n).copy()
        sensor_type = np.frombuffer(f.read(count * sensors_n), dtype=np.int8).reshape(count, sensors_n).copy()

        action_x    = np.frombuffer(f.read(count * actions_n), dtype=np.int8).reshape(count, actions_n).copy()
        action_y    = np.frombuffer(f.read(count * actions_n), dtype=np.int8).reshape(count, actions_n).copy()
        action_type = np.frombuffer(f.read(count * actions_n), dtype=np.int8).reshape(count, actions_n).copy()

    return {
        'count': count,
        'sensors_n': sensors_n,
        'actions_n': actions_n,
        'x': x,                      # (count,)            uint32
        'y': y,                      # (count,)            uint32
        'energy': energy,            # (count,)            float32
        'ids': ids,                  # (count,)            int64
        'chosen_action': chosen_action,  # (count,)        int8
        'sensor_x': sensor_x,       # (count, sensors_n)  int8
        'sensor_y': sensor_y,       # (count, sensors_n)  int8
        'sensor_type': sensor_type,  # (count, sensors_n)  int8
        'action_x': action_x,       # (count, actions_n)  int8
        'action_y': action_y,       # (count, actions_n)  int8
        'action_type': action_type,  # (count, actions_n)  int8
    }


def available_ticks():
    """Return sorted list of ticks for which map files exist in BUILD_DIR."""
    ticks = set()
    for fname in os.listdir(BUILD_DIR):
        if fname.startswith('map_') and fname.endswith('.bin'):
            try:
                ticks.add(int(fname[4:10]))
            except ValueError:
                pass
    return sorted(ticks)


if __name__ == '__main__':
    ticks = available_ticks()
    if not ticks:
        print(f'No files found in {os.path.abspath(BUILD_DIR)}')
    else:
        print(f'Available ticks: {ticks[0]} - {ticks[-1]} ({len(ticks)} files)')

        t = ticks[1]
        m = read_map(t)
        c = read_creatures(t)

        print(f'\n--- Map tick {t} ---')
        print(f'  Size: {m["width"]}x{m["height"]}')
        print(f'  Food    min/max: {m["food"].min():.3f} / {m["food"].max():.3f}')
        print(f'  Danger  min/max: {m["danger"].min():.3f} / {m["danger"].max():.3f}')
        print(f'  Creature min/max: {m["creature"].min():.3f} / {m["creature"].max():.3f}')

        print(f'\n--- Creatures tick {t} ---')
        print(f'  Count: {c["count"]}')
        print(f'  Sensors: {c["sensors_n"]}, Actions: {c["actions_n"]}')
        print(f'  Energy  min/max: {c["energy"].min():.3f} / {c["energy"].max():.3f}')
        print(f'  X min/max: {c["x"].min()} / {c["x"].max()}')
        print(f'  Y min/max: {c["y"].min()} / {c["y"].max()}')

        unique, counts = np.unique(c['chosen_action'], return_counts=True)
        print('  Chosen actions:')
        for val, cnt in zip(unique, counts):
            print(f'    {ACTION_NAMES.get(int(val), str(val))}: {cnt}')