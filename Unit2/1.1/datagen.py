# datagen.py
import random
import time
from decimal import Decimal, ROUND_HALF_UP # Import ROUND_HALF_UP
import math # For math.nextafter
import sys

# --- Attempt to import from config, provide fallbacks ---
try:
    from config import FLOORS, NUM_ELEVATORS, int_to_floor, floor_to_int, MIN_FLOOR, MAX_FLOOR, INVALID_FLOOR
except ImportError:
    print("Warning: Using default config values in datagen.py")
    FLOORS = [-3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Example: B3 to F10
    NUM_ELEVATORS = 6
    MIN_FLOOR = -3
    MAX_FLOOR = 10
    INVALID_FLOOR = 0
    def floor_to_int(floor_str):
        if floor_str.startswith('F'): return int(floor_str[1:])
        elif floor_str.startswith('B'): return -(int(floor_str[1:]) - 1)
        else: raise ValueError(f"Invalid floor string: {floor_str}")
    def int_to_floor(floor_int):
        if floor_int > 0: return f"F{floor_int}"
        elif floor_int <= 0: return f"B{-floor_int + 1}"
        else: raise ValueError("Invalid floor int")

# --- Data Generation Parameters ---
TARGET_ELEVATOR_ID = 1
OTHER_ELEVATOR_PROBABILITY = 0.05
MAX_TIME_DELTA = 0.2
MAX_BURST_SIZE = 4
TIME_QUANTIZE = Decimal("0.1")

def generate_floor():
    """Generates a valid floor number based on config."""
    while True:
        floor = random.randint(MIN_FLOOR, MAX_FLOOR)
        if floor != INVALID_FLOOR:
            return floor

def generate_data(num_requests, output_file="stdin.txt", is_hack_data=False):
    """
    Generates passenger requests, focusing on density and a target elevator.
    First request arrives after T=1.0s.
    Hack mode currently only limits max time to 50s.
    """
    if not (1 <= num_requests <= 200):
        raise ValueError(f"Invalid number of requests: {num_requests}. Must be 1-200.")

    requests = []
    used_passenger_ids = set()

    # --- Modified Time Initialization ---
    # Start time after 1.0 second, with a small random offset up to MAX_TIME_DELTA
    initial_offset = Decimal(random.uniform(0.0, MAX_TIME_DELTA)).quantize(TIME_QUANTIZE, rounding=ROUND_HALF_UP)
    current_time = Decimal("1.0") + initial_offset
    # Ensure it's strictly > 1.0 if offset could be 0
    if current_time <= Decimal("1.0"):
         try: current_time = Decimal(str(math.nextafter(1.0, float('inf')))).quantize(TIME_QUANTIZE, rounding=ROUND_HALF_UP)
         except AttributeError: current_time = Decimal("1.0") + Decimal(str(sys.float_info.epsilon)).quantize(TIME_QUANTIZE, rounding=ROUND_HALF_UP)
    # --- End Modified Time Initialization ---


    for i in range(num_requests):
        # --- Determine Timestamp (Handle Bursts) ---
        # Only increment time if starting a new timestamp group (not first request)
        if i > 0 and random.randint(1, MAX_BURST_SIZE) == 1:
             time_increment = Decimal(random.uniform(0.0, MAX_TIME_DELTA)).quantize(TIME_QUANTIZE, rounding=ROUND_HALF_UP)
             new_time = current_time + time_increment
             try: new_time_float = math.nextafter(float(new_time), float('inf'))
             except AttributeError: new_time_float = float(new_time) + sys.float_info.epsilon
             current_time = Decimal(str(new_time_float)).quantize(TIME_QUANTIZE, rounding=ROUND_HALF_UP)

        # Apply hack mode time limit
        if is_hack_data and current_time > Decimal("50.0"):
             current_time = Decimal("50.0")

        # Generate unique passenger ID
        while True:
            passenger_id = random.randint(1, num_requests * 2 + 100)
            if passenger_id not in used_passenger_ids:
                used_passenger_ids.add(passenger_id); break

        # Generate floors
        while True:
            from_floor_int = generate_floor()
            to_floor_int = generate_floor()
            if from_floor_int != to_floor_int: break
        from_floor_str = int_to_floor(from_floor_int)
        to_floor_str = int_to_floor(to_floor_int)

        # Generate priority
        priority = random.randint(1, 20)

        # Assign Elevator (Focus on TARGET_ELEVATOR_ID)
        elevator_id = TARGET_ELEVATOR_ID
        if random.random() < OTHER_ELEVATOR_PROBABILITY and NUM_ELEVATORS > 1:
            while True:
                other_id = random.randint(1, NUM_ELEVATORS)
                if other_id != TARGET_ELEVATOR_ID: elevator_id = other_id; break

        # Add request line
        requests.append(f"[{current_time}]{passenger_id}-PRI-{priority}-FROM-{from_floor_str}-TO-{to_floor_str}-BY-{elevator_id}")

    # --- Write to File ---
    try:
        with open(output_file, "w", encoding='utf-8') as f:
            for req in requests:
                f.write(req + "\n")
        last_ts = current_time # Use the final timestamp generated
        print(f"Generated {len(requests)} requests in '{output_file}'. Last timestamp: {last_ts}")
        return requests
    except IOError as e:
        print(f"Error writing to output file '{output_file}': {e}")
        raise

if __name__ == '__main__':
    # Example usage
    try:
        print("Generating default data (15 requests, target E1, dense, start > 1s):")
        generate_data(15)

        print("-" * 20)
        print("Generating 'hack' data (30 requests, target E1, dense, max 50s, start > 1s):")
        generate_data(30, is_hack_data=True, output_file="hack_stdin.txt")

    except ValueError as e: print(f"Error: {e}")
    except IOError as e: print(f"File Error: {e}")