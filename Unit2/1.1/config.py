# config.py
import re
from decimal import Decimal, ROUND_HALF_UP

# --- Constants ---
FLOORS = [-4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7] # B4, B3, B2, F1, ..., F7
MIN_FLOOR = -3
MAX_FLOOR = 7
NUM_ELEVATORS = 6
MAX_CAPACITY = 6
MOVE_TIME_PER_FLOOR = Decimal("0.4")
DOOR_OPEN_CLOSE_TIME = Decimal("0.4") # Min time doors stay open
OPEN_ACTION_TIME = Decimal("0.2") # Time taken to fully open
CLOSE_ACTION_TIME = Decimal("0.2") # Time taken to fully close
INVALID_FLOOR = 0
EPSILON = Decimal("1e-7") # Tolerance for float comparisons

# --- Floor Mapping ---
def floor_to_int(floor_str):
    if floor_str.startswith('F'):
        return int(floor_str[1:])
    elif floor_str.startswith('B'):
        return -int(floor_str[1:])
    else:
        raise ValueError(f"Invalid floor string: {floor_str}")

def int_to_floor(floor_int):
    if floor_int > 0:
        return f"F{floor_int}"
    elif floor_int < 0:
        return f"B{-floor_int}"
    else:
        raise ValueError("Floor 0 does not exist.") # According to B1/F1 mapping

# --- Output Parsing ---
# Example: [ 11.8010]CLOSE-F3-6
OUTPUT_PATTERN = re.compile(r"\[\s*(\d+\.\d+)\s*\](ARRIVE|OPEN|CLOSE|IN|OUT)-(\S+)-(\d+)(?:-(\d+))?")
# Groups: 1: timestamp, 2: action, 3: floor/passengerID, 4: elevatorID/floor, 5: elevatorID (optional for IN/OUT)

# Example: [1.0]295-PRI-20-FROM-F1-TO-F5-BY-4
INPUT_PATTERN = re.compile(r"\[\s*(\d+\.\d+)\s*\](\d+)-PRI-(\d+)-FROM-(\S+)-TO-(\S+)-BY-(\d+)")
# Groups: 1: timestamp, 2: passengerID, 3: priority, 4: from_floor, 5: to_floor, 6: elevatorID

if __name__ == '__main__':
    # Simple tests for floor mapping
    assert floor_to_int("F1") == 1
    assert floor_to_int("B4") == -3
    assert int_to_floor(7) == "F7"
    assert int_to_floor(-1) == "B1"
    print("Config tests passed.")