# -- coding: utf-8 --

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import queue  # Used for GUI updates from threads
import time
import random
import os
import json
import re
import sys
import subprocess
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import traceback

# --- Platform Specific Import ---
# Ensure this runs only once and handles the wexpect dependency check cleanly
# NOTE: wexpect/pexpect is NOT actually used in the current logic,
# which relies on standard subprocess. This section is kept for historical
# context or future potential use but doesn't affect the current execution flow.
try:
    if sys.platform == 'win32':
        # Check if wexpect is *intended* to be used later, but current code doesn't use it.
        # For now, we just ensure Tkinter is okay.
        pass
        # try:
        #     import wexpect as pexpect # Use wexpect on Windows
        # except ImportError:
        #     # Use tk.Tk().withdraw() to allow messagebox even before main app loop starts
        #     root = tk.Tk()
        #     root.withdraw()
        #     messagebox.showerror("Missing Dependency", "'wexpect' potentially required for future features.\nPlease consider running: pip install wexpect")
        #     # root.destroy() # Don't exit, let the app continue without it for now
    # Add elif for 'linux' or 'darwin' if pexpect support is needed there
    # elif sys.platform in ['linux', 'darwin']:
    #     try:
    #         import pexpect
    #     except ImportError:
    #         messagebox.showerror("Missing Dependency", "'pexpect' required. pip install pexpect")
    #         sys.exit(1)
    else:
        # Only show error if OS is truly unsupported *and* platform-specific libs were intended
        # For now, assume standard subprocess is fine cross-platform.
        # root = tk.Tk()
        # root.withdraw()
        # messagebox.showerror("Unsupported OS", f"Platform-specific features might be limited on: {sys.platform}.")
        # root.destroy()
        # sys.exit(1)
        pass # Assume standard subprocess works
except tk.TclError:
    print("ERROR: Could not initialize Tkinter or show error dialog.", file=sys.stderr)
    print("Ensure Tkinter is installed correctly.", file=sys.stderr)
    sys.exit(1)
except SystemExit: # Catch sys.exit calls
    sys.exit(1) # Re-raise to ensure exit
except Exception as e: # Catch unexpected errors during import
    print(f"ERROR during platform specific import check: {e}", file=sys.stderr)
    # Try showing a messagebox if possible
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Import Error", f"An unexpected error occurred during setup:\n{e}")
        root.destroy()
    except Exception:
        pass # Ignore if messagebox fails too
    sys.exit(1)

# --- Configuration ---
FLOORS = [f"B{i}" for i in range(4, 0, -1)] + [f"F{i}" for i in range(1, 8)]
FLOOR_ORDER = {floor: i for i, floor in enumerate(FLOORS)} # For distance calculation
FLOOR_INDEX_TO_NAME = {v: k for k, v in FLOOR_ORDER.items()} # <<< NEW >>> Reverse mapping
ELEVATOR_IDS = list(range(1, 7))
NUM_ELEVATORS = 6
PRIORITY_RANGE = (1, 100) # <<< MODIFIED as per HW7 spec >>>
SCHEDULE_SPEEDS = [0.2, 0.3, 0.4, 0.5]
PROCESS_ENCODING = 'utf-8' # Or 'gbk' if needed for specific environments
STDIN_FILENAME = "stdin.txt"
FAILURES_DIR = "batch_failures"
MANUAL_FAILURES_DIR = "_Manual_Failures" # Subdir within FAILURES_DIR
DEFAULT_CONCURRENCY = max(1, os.cpu_count() // 2 if os.cpu_count() else 4) # Safer default concurrency
T_MAX_HUTEST = 220.0 # Max time for 互测 (peer testing)

# --- HW7 Specific Configuration <<< NEW >>> ---
ALLOWED_UPDATE_TARGET_FLOORS = [f"F{i}" for i in range(1, 6)] # F1-F5 (Target floor range for UPDATE)
UPDATE_T_COMPLETE_MAX = 6.0 # s (From UPDATE-ACCEPT to UPDATE-END)
UPDATE_T_RESET_MIN = 1.0   # s (Min duration from UPDATE-BEGIN to UPDATE-END)
DOUBLE_CAR_SPEED = 0.2     # s/floor (Fixed speed after update)
MAX_ARRIVES_BEFORE_UPDATE_BEGIN = 2 # Max ARRIVEs between UPDATE-ACCEPT and UPDATE-BEGIN for *each* elevator

# --- Validation Constants ---
DEFAULT_SPEED = 0.4 # s/floor
MIN_DOOR_TIME = 0.4 # s (>= 0.4s means minimal open duration is 0.4s)
CAPACITY = 6
SCHE_MAX_ARRIVES_BEFORE_BEGIN = 2 # Max ARRIVEs between SCHE-ACCEPT and SCHE-BEGIN
SCHE_MAX_COMPLETE_TIME = 6.0 # s (From SCHE-ACCEPT to SCHE-END)
ALLOWED_SCHE_TARGET_FLOORS = ['B2', 'B1', 'F1', 'F2', 'F3', 'F4', 'F5'] # Target floor range for SCHE
TIMESTAMP_TOLERANCE = 0.005 # Small tolerance for float comparisons
SPEED_TOLERANCE_FACTOR_LOW = 0.85 # Allow slightly faster than expected
SPEED_TOLERANCE_FACTOR_HIGH = 1.6 # Allow significantly slower (e.g., first move)

# --- Add these near other Validation Constants --- <<< NEW >>>
TIMEOUT_CHECK_INTERVAL = 1.0 # seconds - How often to run timeout checks
TIMEOUT_MOVE = 15.0          # Max time allowed between actions while MOVING
TIMEOUT_DOOR_OPEN = 8.0      # Max time door can stay open without subsequent action (IN/OUT/CLOSE)
TIMEOUT_RECEIVE_TO_ACTION = 12.0 # Max time idle with request before *any* action
TIMEOUT_UPDATE_BEGUN = 10.0  # Max time elevator can be in the UPDATE-BEGUN silent phase
# Note: SCHE/UPDATE completion times are already checked explicitly at END events

# --- Timestamp Helper ---
def parse_output_timestamp(line):
    """Extracts the timestamp from a log line. Allows whitespace flexibility."""
    # Matches [ xxx.xxx ] or [xxx.xxx] at the start
    match = re.match(r'^\s*\[\s*(\d+\.\d+)\s*\]', line)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

# --- Floor Helper ---
def floor_diff(floor1, floor2):
    """Calculates absolute difference in floor levels."""
    if floor1 in FLOOR_ORDER and floor2 in FLOOR_ORDER:
        return abs(FLOOR_ORDER[floor1] - FLOOR_ORDER[floor2])
    return float('inf') # Invalid floor comparison

# --- Data Generation Function (Improved Randomness & UPDATE Limit) ---
def generate_test_data(num_passengers, num_schedules, num_updates, max_time):
    """
    Generates test data with improved randomness and UPDATE constraints.
    - Mixes request generation types better.
    - Limits each elevator to participate in at most one UPDATE.
    - Limits total UPDATE commands to at most 3.
    """
    requests_timed = []
    used_passenger_ids = set()
    # Earliest time the *next* command (SCHE or UPDATE) can be sent for an elevator
    elevator_next_cmd_allowed_time = {eid: 0.0 for eid in ELEVATOR_IDS}
    # Track live config and update participation <<< MODIFIED >>>
    current_elevator_config = {eid: {'active': True, 'shaft': eid, 'double_car': False, 'updated': False} for eid in ELEVATOR_IDS}
    update_count_total = 0 # <<< NEW >>> Track total updates generated

    ALL_SYSTEM_FLOORS = FLOORS

    # --- Combined Generation Loop ---
    # Determine total requests to generate
    total_requests_to_generate = num_passengers + num_schedules + min(num_updates, 3) # Limit updates <= 3
    generated_counts = {'PRI': 0, 'SCHE': 0, 'UPDATE': 0}
    request_types = ['PRI'] * num_passengers + ['SCHE'] * num_schedules + ['UPDATE'] * min(num_updates, 3)
    random.shuffle(request_types) # Shuffle types for better mixing

    MAX_GENERATION_ATTEMPTS = total_requests_to_generate * 5 # Limit overall attempts

    for attempt in range(MAX_GENERATION_ATTEMPTS):
        if len(requests_timed) >= total_requests_to_generate: # Stop if we have enough
             break
        if not request_types: # Stop if no more types left to try
             break

        req_type = request_types.pop(0) # Get next type from shuffled list

        # --- Try to generate PRI ---
        if req_type == 'PRI':
            pid = random.randint(1, 9999)
            while pid in used_passenger_ids: pid = random.randint(1, 9999)
            used_passenger_ids.add(pid)
            from_fl = random.choice(ALL_SYSTEM_FLOORS)
            possible_to_fl = [f for f in ALL_SYSTEM_FLOORS if f != from_fl]
            if not possible_to_fl: request_types.append(req_type); continue # Put back and try later
            to_fl = random.choice(possible_to_fl)
            priority = random.randint(PRIORITY_RANGE[0], PRIORITY_RANGE[1])
            # <<< MODIFIED: More flexible send time for PRI >>>
            min_send_time_pri = 0.1 # Passengers can arrive early
            # Allow passengers throughout, but maybe less dense at the very end
            max_send_time_pri = max(min_send_time_pri + 1.0, max_time * 0.95)
            if min_send_time_pri >= max_send_time_pri : request_types.append(req_type); continue
            send_time = round(random.uniform(min_send_time_pri, max_send_time_pri), 4)
            cmd = f"{pid}-PRI-{priority}-FROM-{from_fl}-TO-{to_fl}"
            requests_timed.append((send_time, cmd))
            generated_counts['PRI'] += 1

        # --- Try to generate SCHE ---
        elif req_type == 'SCHE':
            if not ALLOWED_SCHE_TARGET_FLOORS: request_types.append(req_type); continue
            available_sche_elevators = [eid for eid, cfg in current_elevator_config.items() if cfg['active']]
            if not available_sche_elevators: request_types.append(req_type); continue
            eid = random.choice(available_sche_elevators)
            speed = random.choice(SCHEDULE_SPEEDS)
            target_fl = random.choice(ALLOWED_SCHE_TARGET_FLOORS)
            min_send_time = max(elevator_next_cmd_allowed_time[eid], 0.5) # SCHE needs some delay
            # <<< MODIFIED: Allow SCHE earlier, not just at the end >>>
            max_send_time = max(min_send_time + 1.0, max_time - SCHE_MAX_COMPLETE_TIME - 2.0) # Leave buffer
            if min_send_time >= max_send_time: request_types.append(req_type); continue # Cannot fit, try later
            send_time = round(random.uniform(min_send_time, max_send_time), 4)
            cmd = f"SCHE-{eid}-{speed}-{target_fl}"
            requests_timed.append((send_time, cmd))
            # Update next allowed time for this elevator
            next_available = send_time + SCHE_MAX_COMPLETE_TIME + 2.0 # SCHE completion buffer
            elevator_next_cmd_allowed_time[eid] = max(elevator_next_cmd_allowed_time[eid], next_available)
            generated_counts['SCHE'] += 1

        # --- Try to generate UPDATE ---
        elif req_type == 'UPDATE':
             # Check total update limit <<< NEW >>>
             if update_count_total >= 3: continue # Already generated max updates

             # Find available elevators that haven't been updated yet <<< MODIFIED >>>
             possible_a = [eid for eid, cfg in current_elevator_config.items() if cfg['active'] and not cfg['double_car'] and not cfg['updated']]
             possible_b = list(possible_a)
             if len(possible_a) < 2: continue # Need two eligible, un-updated elevators

             eid_a = random.choice(possible_a)
             possible_b.remove(eid_a)
             eid_b = random.choice(possible_b)

             if not ALLOWED_UPDATE_TARGET_FLOORS: continue
             target_fl = random.choice(ALLOWED_UPDATE_TARGET_FLOORS)

             # Determine earliest send time <<< MODIFIED >>>
             min_send_time_a = elevator_next_cmd_allowed_time[eid_a]
             min_send_time_b = elevator_next_cmd_allowed_time[eid_b]
             min_overall_send_time = max(min_send_time_a, min_send_time_b, 1.0) # UPDATE needs more delay

             # <<< MODIFIED: Allow UPDATE earlier >>>
             max_send_time = max(min_overall_send_time + 1.0, max_time - UPDATE_T_COMPLETE_MAX - 5.0) # Leave ample buffer

             if min_overall_send_time >= max_send_time: request_types.append(req_type); continue # Cannot fit, try later

             send_time = round(random.uniform(min_overall_send_time, max_send_time), 4)
             cmd = f"UPDATE-{eid_a}-{eid_b}-{target_fl}"
             requests_timed.append((send_time, cmd))

             # Update next allowed times for *both* elevators <<< MODIFIED >>>
             next_available = send_time + UPDATE_T_COMPLETE_MAX + 8.0 # Longer buffer after UPDATE
             elevator_next_cmd_allowed_time[eid_a] = next_available
             elevator_next_cmd_allowed_time[eid_b] = next_available

             # Update simulated elevator config <<< MODIFIED >>>
             current_elevator_config[eid_a]['shaft'] = eid_b
             current_elevator_config[eid_a]['double_car'] = True
             current_elevator_config[eid_a]['updated'] = True # Mark as updated
             current_elevator_config[eid_b]['double_car'] = True
             current_elevator_config[eid_b]['updated'] = True # Mark as updated
             update_count_total += 1 # Increment total count <<< NEW >>>
             generated_counts['UPDATE'] += 1

        # If a request type couldn't be generated in this attempt, put it back if needed
        # (Handled by continue statements above for SCHE/UPDATE if conditions not met)

    # --- Final Log and Sort ---
    print(f"DEBUG Data Generation: Generated {generated_counts['PRI']}/{num_passengers} PRI, "
          f"{generated_counts['SCHE']}/{num_schedules} SCHE, "
          f"{generated_counts['UPDATE']}/{min(num_updates, 3)} UPDATE.")
    if len(requests_timed) < total_requests_to_generate:
        print(f"WARNING: Could only generate {len(requests_timed)}/{total_requests_to_generate} requests.")

    requests_timed.sort(key=lambda x: x[0])
    formatted_lines = [f"[{t:.4f}]{cmd}" for t, cmd in requests_timed]
    return formatted_lines

# --- DETAILED VALIDATION FUNCTION (Updated for HW7 + Timing/DoubleCar Checks) ---
def validate_output(output_log_lines, input_lines):
    """
    Performs detailed validation of elevator output log against rules (HW7).
    Returns a list of issue strings. An empty list means validation passed.
    """
    issues = []
    last_ts = -1.0
    current_time = 0.0 # Tracks the latest valid timestamp seen

    # --- State Tracking ---
    elevators = {} # Key: eid (int)
    passengers = {} # Key: pid (int)
    received_requests = {} # Key: pid (int), Value: eid (int) that received the request
    passenger_requests_from_input = {} # Key: pid (int), Value: {'from': str, 'to': str}

    # <<< MODIFIED: Initialize elevator states with HW7 fields and timing helpers >>>
    for eid in ELEVATOR_IDS:
        elevators[eid] = {
            'id': eid,
            'floor': 'F1',      # Current floor (str)
            'door': 'CLOSED',   # 'OPEN' or 'CLOSED'
            'motion': 'IDLE',   # 'IDLE' or 'MOVING'
            'passengers': set(),# Set of passenger IDs (int) inside
            'capacity': CAPACITY,
            'speed': DEFAULT_SPEED, # Current speed (float, s/floor)
            'last_action_ts': 0.0, # Timestamp of the last action by this elevator
            'last_floor_ts': 0.0, # Timestamp of the last ARRIVE event (still useful for sequence checks)
            'last_open_ts': 0.0,  # Timestamp of the last OPEN event
            'last_close_ts': 0.0, # Timestamp of the last CLOSE event
            'last_departure_ts': 0.0, # <<< NEW >>> Timestamp of last CLOSE, SCHE-BEGIN, or UPDATE-END that initiated movement
            # SCHE State (existing)
            'sche_active': False,
            'sche_phase': None,
            'sche_target_floor': None,
            'sche_speed': None,
            'sche_accept_ts': -1.0,
            'sche_begin_ts': -1.0,
            'sche_end_ts': -1.0,
            'sche_arrives_since_accept': 0,
            'sche_opened_at_target': False,
             # HW7 UPDATE State (existing/modified)
            'is_active': True,          # Conceptually active?
            'min_floor': 'B4',
            'max_floor': 'F7',
            'shaft_id': eid,
            'is_double_car': False,
            'update_status': None,
            'update_role': None,
            'update_partner_id': None,
            'update_target_floor': None,
            'update_accept_ts': -1.0,
            'update_begin_ts': -1.0,
            'update_end_ts': -1.0,
            'update_arrives_since_accept': 0,
            'update_emptied_after_accept': False,
            'just_updated': False,      # Allows one action post-UPDATE without certain checks
        }

    # --- Initialize passenger states from input ---
    # (This part remains unchanged from your provided code)
    for line in input_lines:
        ts_input = parse_output_timestamp(line) or 0.0
        action_part_input = line.split(']', 1)[-1].strip() if ts_input > 0 else line.strip()
        match_pri = re.match(r'(\d+)-PRI-(\d+)-FROM-([BF]\d+)-TO-([BF]\d+)', action_part_input)
        match_sche = re.match(r'SCHE-(\d+)-([\d.]+)-([BF]\d+)', action_part_input)
        match_update_input = re.match(r'UPDATE-A(\d+)-B(\d+)-([BF]\d+)', action_part_input) # Original format like UPDATE-A6-B3-F1

        if match_pri:
            try:
                pid = int(match_pri.group(1))
                prio = int(match_pri.group(2))
                from_f = match_pri.group(3)
                to_f = match_pri.group(4)
                if pid not in passengers:
                     passengers[pid] = {
                        'id': pid, 'location': from_f, 'destination': to_f, 'state': 'WAITING',
                        'assigned_elevator': None, 'request_time': ts_input,
                     }
                     passenger_requests_from_input[pid] = {'from': from_f, 'to': to_f}
            except (ValueError, IndexError):
                 issues.append(f"INPUT WARNING: Malformed passenger request in input: {line}")
        elif match_sche:
              try:
                  eid = int(match_sche.group(1))
                  speed = float(match_sche.group(2))
                  target_fl = match_sche.group(3)
              except (ValueError, IndexError):
                   issues.append(f"INPUT WARNING: Malformed SCHE command in input: {line}")
        # No need to parse UPDATE input format for validation logic

    # --- Regex Patterns (Compile for efficiency) ---
    # (Unchanged - patterns look correct based on previous debug)
    patterns = {
        'ARRIVE': re.compile(r'ARRIVE-([BF]\d+)-(\d+)$'),
        'OPEN':   re.compile(r'OPEN-([BF]\d+)-(\d+)$'),
        'CLOSE':  re.compile(r'CLOSE-([BF]\d+)-(\d+)$'),
        'IN':     re.compile(r'IN-(\d+)-([BF]\d+)-(\d+)$'),
        'OUT_S':  re.compile(r'OUT-S-(\d+)-([BF]\d+)-(\d+)$'),
        'OUT_F':  re.compile(r'OUT-F-(\d+)-([BF]\d+)-(\d+)$'),
        'RECEIVE':re.compile(r'RECEIVE-(\d+)-(\d+)$'),
        'ACCEPT': re.compile(r'SCHE-ACCEPT-(\d+)-([\d.]+)-([BF]\d+)$'),
        'BEGIN':  re.compile(r'SCHE-BEGIN-(\d+)$'),
        'END':    re.compile(r'SCHE-END-(\d+)$'),
        'UPDATE_ACCEPT': re.compile(r'UPDATE-ACCEPT-(\d+)-(\d+)-([BF]\d+)$'),
        'UPDATE_BEGIN':  re.compile(r'UPDATE-BEGIN-(\d+)-(\d+)$'),
        'UPDATE_END':    re.compile(r'UPDATE-END-(\d+)-(\d+)$'),
    }

    # --- Process Log Lines ---
    for line_num, raw_line in enumerate(output_log_lines, 1):
        line = raw_line.strip()
        if not line or line.startswith('#') or line.startswith('--'):
            continue

        ts = parse_output_timestamp(line)
        action_part = line # Default

        # Validate and update timestamp (Unchanged)
        if ts is not None:
            if ts < last_ts - TIMESTAMP_TOLERANCE:
                issues.append(f"[L{line_num} @ {ts:.4f}] Timestamp decreased: {ts:.4f} is less than previous {last_ts:.4f} in: {raw_line}")
            if ts >= last_ts - TIMESTAMP_TOLERANCE:
                 current_time = ts
                 last_ts = max(ts, last_ts)
            else:
                 current_time = last_ts
                 issues.append(f"[L{line_num} @ {ts:.4f}] WARNING: Timestamp decreased significantly, processing event using last valid time {last_ts:.4f}")

            action_part = line.split(']', 1)[-1].strip()
            if not action_part:
                 issues.append(f"[L{line_num} @ {current_time:.4f}] Line has timestamp but no action: {raw_line}")
                 continue
        else:
            if not any(action_part.startswith(kw) for kw in ['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL', 'Result:', '---', '[', '#', 'at ', '\tat ']):
                 issues.append(f"[L{line_num}] Malformed Line / Missing Timestamp for core action?: {raw_line}")
                 issues.append(f"[L{line_num} @ {current_time:.4f}] WARNING: Action processed using last known time due to missing timestamp: {raw_line}")

        # --- Match Action --- (Unchanged)
        matched_action = None
        action_data = None
        for action, pattern in patterns.items():
            match = pattern.match(action_part)
            if match:
                matched_action = action
                action_data = match.groups()
                break

        if not matched_action:
            if not any(action_part.startswith(kw) for kw in ['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL', 'Result:', '---', '[', '#', 'at ', '\tat ']):
                 issues.append(f"[L{line_num} @ {current_time:.4f}] Unknown or Malformed Action Format: {action_part}")
            continue

        # --- Get Elevator/Passenger State (Unchanged - logic seems okay) ---
        eid = None; eid_a = None; eid_b = None; pid = None
        e = None; e_a = None; e_b = None; p = None
        current_event_time = ts if ts is not None else current_time

        try:
            if matched_action in ['ARRIVE', 'OPEN', 'CLOSE', 'BEGIN', 'END']: eid = int(action_data[-1])
            elif matched_action in ['IN', 'OUT_S', 'OUT_F']: pid = int(action_data[0]); eid = int(action_data[-1])
            elif matched_action == 'RECEIVE': pid = int(action_data[0]); eid = int(action_data[1])
            elif matched_action == 'ACCEPT': eid = int(action_data[0])
            elif matched_action == 'UPDATE_ACCEPT': eid_a = int(action_data[0]); eid_b = int(action_data[1])
            elif matched_action == 'UPDATE_BEGIN' or matched_action == 'UPDATE_END': eid_a = int(action_data[0]); eid_b = int(action_data[1])

            if eid is not None:
                if eid not in elevators: issues.append(f"[L{line_num} @ {current_event_time:.4f}] {matched_action}: Invalid elevator ID {eid}: {raw_line}"); continue
                e = elevators[eid]
                if not e['is_active'] and matched_action not in ['UPDATE_ACCEPT', 'UPDATE_BEGIN', 'UPDATE_END']: issues.append(f"[L{line_num} @ {current_event_time:.4f}] {matched_action}: Action on inactive elevator E{eid}: {raw_line}"); continue
            if eid_a is not None:
                 if eid_a not in elevators: issues.append(f"[L{line_num} @ {current_event_time:.4f}] {matched_action}: Invalid A elevator ID {eid_a}: {raw_line}"); continue
                 e_a = elevators[eid_a]
                 if not e_a['is_active'] and matched_action in ['UPDATE_ACCEPT', 'UPDATE_BEGIN', 'UPDATE_END']: issues.append(f"[L{line_num} @ {current_event_time:.4f}] {matched_action}: A elevator E{eid_a} is already inactive: {raw_line}"); continue
            if eid_b is not None:
                 if eid_b not in elevators: issues.append(f"[L{line_num} @ {current_event_time:.4f}] {matched_action}: Invalid B elevator ID {eid_b}: {raw_line}"); continue
                 e_b = elevators[eid_b]
                 if not e_b['is_active'] and matched_action in ['UPDATE_ACCEPT', 'UPDATE_BEGIN', 'UPDATE_END']: issues.append(f"[L{line_num} @ {current_event_time:.4f}] {matched_action}: B elevator E{eid_b} is inactive: {raw_line}"); continue
            if pid is not None:
                if pid not in passengers:
                    if matched_action in ['IN', 'OUT_S', 'OUT_F', 'RECEIVE']: issues.append(f"[L{line_num} @ {current_event_time:.4f}] {matched_action}: Unknown passenger ID {pid}: {raw_line}")
                    if matched_action == 'RECEIVE': continue
                else: p = passengers[pid]
        except (ValueError, IndexError) as id_err:
            issues.append(f"[L{line_num} @ {current_event_time:.4f}] Error parsing ID for action '{matched_action}': {id_err} in {raw_line}")
            continue

        # --- Process Matched Action ---
        try:
            # --- ARRIVE --- <<< MODIFIED: Timing, Double-Car Checks >>>
            if matched_action == 'ARRIVE':
                floor = action_data[0]
                if e is None: continue

                # Check 0: UPDATE Silent Phase Violation (Unchanged)
                if e['update_status'] == 'BEGUN':
                     issues.append(f"[L{line_num} @ {current_event_time:.4f}] UPDATE VIOLATION: E{eid} ARRIVE-{floor} during UPDATE silent phase: {raw_line}")
                     continue

                # Check 1: Door must be closed (Unchanged)
                if e['door'] != 'CLOSED':
                     issues.append(f"[L{line_num} @ {current_event_time:.4f}] E{eid} ARRIVE-{floor} door not CLOSED (was {e['door']}): {raw_line}")

                # Check 2: Floor validity, Sequence
                previous_floor = e['floor'] # Get floor before state update
                if floor not in FLOOR_ORDER:
                     issues.append(f"[L{line_num} @ {current_event_time:.4f}] E{eid} ARRIVE invalid floor '{floor}': {raw_line}")
                     # Update state to stop at invalid floor, but don't proceed with other checks
                     e['floor'] = floor; e['motion'] = 'IDLE'; e['last_floor_ts'] = current_event_time
                     e['last_action_ts'] = current_event_time
                     continue # Stop ARRIVE processing here

                diff = floor_diff(previous_floor, floor)
                if e['last_floor_ts'] > 0: # Check sequence only if not the first recorded move for this elevator
                     if diff == 0:
                          if not e['just_updated']: # Allow same floor ARRIVE immediately after UPDATE-END
                               issues.append(f"[L{line_num} @ {current_event_time:.4f}] WARN: E{eid} ARRIVE at same floor {floor} it was last at ({previous_floor}).")
                     elif diff != 1:
                          issues.append(f"[L{line_num} @ {current_event_time:.4f}] E{eid} ARRIVE floor jump: {previous_floor}->{floor} (diff {diff}): {raw_line}")

                # <<< MODIFIED: Check 3: Timing (using last_departure_ts) >>>
                current_speed = e['speed']
                # Use last_departure_ts if available and makes sense (i.e., elevator was MOVING)
                # last_departure_ts is set on CLOSE, SCHE-BEGIN(moving), UPDATE-END
                start_time_for_move = e['last_departure_ts'] if e['last_departure_ts'] > e['last_floor_ts'] and e['last_departure_ts'] > 0 else e['last_floor_ts']

                if start_time_for_move > 0 and diff == 1 and current_speed is not None and current_speed > 0:
                    time_elapsed = current_event_time - start_time_for_move
                    expected_time = current_speed
                    # Use slightly wider tolerance, especially for high bound
                    low_bound = expected_time * SPEED_TOLERANCE_FACTOR_LOW - TIMESTAMP_TOLERANCE
                    high_bound = expected_time * (SPEED_TOLERANCE_FACTOR_HIGH + 0.2) + TIMESTAMP_TOLERANCE # Increased high factor to 1.8

                    # Avoid triggering on very small elapsed times or zero speed
                    if time_elapsed > 0.010 and not (low_bound <= time_elapsed <= high_bound):
                         timing_basis = "last departure" if start_time_for_move == e['last_departure_ts'] else "last arrive"
                         issues.append(f"[L{line_num} @ {current_event_time:.4f}] WARN: E{eid} ARRIVE timing inconsistent? {time_elapsed:.4f}s (since {timing_basis} @ {start_time_for_move:.4f}) vs ~{expected_time:.4f}s ({previous_floor}->{floor}, speed={current_speed})")

                elif diff == 1 and (current_speed is None or current_speed <= 0):
                    issues.append(f"[L{line_num} @ {current_event_time:.4f}] WARN: E{eid} ARRIVE check invalid speed ({current_speed}) for timing calc.")


                # Check 4: SCHE Arrive Count (Unchanged)
                if e['sche_active'] and e['sche_phase'] == 'ACCEPTED':
                    e['sche_arrives_since_accept'] += 1
                    if e['sche_arrives_since_accept'] > SCHE_MAX_ARRIVES_BEFORE_BEGIN:
                        issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} had >{SCHE_MAX_ARRIVES_BEFORE_BEGIN} ARRIVEs after ACCEPT before BEGIN: {raw_line}")

                # Check 5: UPDATE Arrive Count (Unchanged)
                if e['update_status'] == 'ACCEPTED':
                    e['update_arrives_since_accept'] += 1
                    if e['update_arrives_since_accept'] > MAX_ARRIVES_BEFORE_UPDATE_BEGIN:
                        issues.append(f"[L{line_num} @ {current_event_time:.4f}] UPDATE VIOLATION: E{eid} had >{MAX_ARRIVES_BEFORE_UPDATE_BEGIN} ARRIVEs after UPDATE-ACCEPT before UPDATE-BEGIN: {raw_line}")


                # <<< NEW/MODIFIED Check 6: Double Car State Consistency (Range, Collision, Speed) >>>
                partner_e = None
                if e['is_double_car'] and e['update_partner_id'] is not None:
                    partner_eid = e['update_partner_id']
                    if partner_eid in elevators and elevators[partner_eid]['is_active'] and elevators[partner_eid]['shaft_id'] == e['shaft_id']:
                         partner_e = elevators[partner_eid]
                         # Check partner's double car status for consistency
                         if not partner_e['is_double_car'] or partner_e['update_partner_id'] != eid:
                              issues.append(f"[L{line_num} @ {current_event_time:.4f}] INTERNAL WARN: E{eid} is double_car, but partner E{partner_eid} state inconsistent.")
                              partner_e = None # Don't use inconsistent partner for checks

                # A: Range Check (apply BEFORE collision check)
                if not (FLOOR_ORDER[e['min_floor']] <= FLOOR_ORDER[floor] <= FLOOR_ORDER[e['max_floor']]):
                     issues.append(f"[L{line_num} @ {current_event_time:.4f}] VIOLATION: E{eid} ARRIVE at {floor} outside its allowed range ({e['min_floor']}-{e['max_floor']}): {raw_line}")
                     # Don't stop processing, but log violation

                # B: Collision Check
                if partner_e:
                     partner_floor_idx = FLOOR_ORDER.get(partner_e['floor'])
                     my_new_floor_idx = FLOOR_ORDER.get(floor)

                     if my_new_floor_idx is not None and partner_floor_idx is not None:
                          if my_new_floor_idx == partner_floor_idx:
                               issues.append(f"[L{line_num} @ {current_event_time:.4f}] COLLISION VIOLATION: E{eid} ARRIVE at {floor}, same floor as partner E{partner_e['id']} (at {partner_e['floor']}): {raw_line}")
                          elif e['update_role'] == 'A' and partner_e['update_role'] == 'B' and my_new_floor_idx < partner_floor_idx:
                               issues.append(f"[L{line_num} @ {current_event_time:.4f}] COLLISION VIOLATION: E{eid}(A) ARRIVE at {floor} below E{partner_e['id']}(B) at {partner_e['floor']}: {raw_line}")
                          elif e['update_role'] == 'B' and partner_e['update_role'] == 'A' and my_new_floor_idx > partner_floor_idx:
                               issues.append(f"[L{line_num} @ {current_event_time:.4f}] COLLISION VIOLATION: E{eid}(B) ARRIVE at {floor} above E{partner_e['id']}(A) at {partner_e['floor']}: {raw_line}")

                # C: Speed Check (for double cars, check if speed is actually correct)
                if e['is_double_car']:
                     # Allow some tolerance around the expected double car speed
                     speed_diff = abs(e['speed'] - DOUBLE_CAR_SPEED)
                     if speed_diff > 0.01 + TIMESTAMP_TOLERANCE: # Allow tiny floating point differences
                          issues.append(f"[L{line_num} @ {current_event_time:.4f}] STATE VIOLATION (WARN): E{eid} (Double Car) current speed is {e['speed']:.4f}, expected {DOUBLE_CAR_SPEED:.4f}. Did not update correctly after UPDATE-END?")
                          # This warning directly targets the observed lty.jar issue


                # Update state AFTER all checks for this ARRIVE event
                e['floor'] = floor
                e['motion'] = 'IDLE' # Arriving always means motion stops momentarily
                e['last_floor_ts'] = current_event_time

                # Reset just_updated flag after first successful action (ARRIVE is an action)
                if e['just_updated']:
                     e['just_updated'] = False

                e['last_action_ts'] = current_event_time

            # --- OPEN --- <<< MODIFIED: Add Double-Car Range Check >>>
            elif matched_action == 'OPEN':
                floor = action_data[0]
                if e is None: continue
                # Check 0: UPDATE Silent Phase Violation (Unchanged)
                if e['update_status'] == 'BEGUN':
                     issues.append(f"[L{line_num} @ {current_event_time:.4f}] UPDATE VIOLATION: E{eid} OPEN-{floor} during UPDATE silent phase: {raw_line}")
                     continue
                # Check 1: Location Consistency (Unchanged)
                if e['floor'] != floor: issues.append(f"[L{line_num} @ {current_event_time:.4f}] E{eid} OPEN at {floor} but is currently at {e['floor']}: {raw_line}")
                # Check 2: Door State (Unchanged)
                if e['door'] != 'CLOSED': issues.append(f"[L{line_num} @ {current_event_time:.4f}] E{eid} OPEN command received but door was already {e['door']}: {raw_line}")
                # Check 3: Motion State (Unchanged)
                if e['motion'] != 'IDLE': issues.append(f"[L{line_num} @ {current_event_time:.4f}] WARN/INTERNAL?: E{eid} OPEN command received but motion state was {e['motion']} (Should be IDLE): {raw_line}")
                # Check 4: SCHE Violation (Unchanged)
                if e['sche_active'] and e['sche_phase'] == 'BEGUN' and e['floor'] != e['sche_target_floor']: issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} OPEN at {floor} during SCHE move phase (target {e['sche_target_floor']}): {raw_line}")

                # <<< NEW Check 5: Double Car Range Check >>>
                if e['is_double_car']:
                     if not (FLOOR_ORDER[e['min_floor']] <= FLOOR_ORDER[floor] <= FLOOR_ORDER[e['max_floor']]):
                          issues.append(f"[L{line_num} @ {current_event_time:.4f}] VIOLATION: E{eid}(Double) OPEN at {floor} outside its allowed range ({e['min_floor']}-{e['max_floor']}): {raw_line}")

                # Update state (Unchanged)
                e['door'] = 'OPEN'
                e['last_open_ts'] = current_event_time
                e['motion'] = 'IDLE'
                e['last_action_ts'] = current_event_time

                # Set flag if this OPEN meets the SCHE target requirement (Unchanged)
                if e['sche_active'] and e['sche_phase'] == 'BEGUN' and e['floor'] == e['sche_target_floor']: e['sche_opened_at_target'] = True

            # --- CLOSE --- <<< MODIFIED: Update last_departure_ts >>>
            elif matched_action == 'CLOSE':
                floor = action_data[0]
                if e is None: continue
                # Check 0: UPDATE Silent Phase Violation (Unchanged)
                if e['update_status'] == 'BEGUN':
                     issues.append(f"[L{line_num} @ {current_event_time:.4f}] UPDATE VIOLATION: E{eid} CLOSE-{floor} during UPDATE silent phase: {raw_line}")
                     continue
                # Check 1: Location Consistency (Unchanged)
                if e['floor'] != floor: issues.append(f"[L{line_num} @ {current_event_time:.4f}] E{eid} CLOSE at {floor} but is currently at {e['floor']}: {raw_line}")
                # Check 2: Door State (Unchanged)
                if e['door'] != 'OPEN': issues.append(f"[L{line_num} @ {current_event_time:.4f}] E{eid} CLOSE command received but door was already {e['door']}: {raw_line}")
                # Check 3: Timing (Unchanged)
                time_since_open = current_event_time - e['last_open_ts']
                if e['last_open_ts'] > 0 and time_since_open < (MIN_DOOR_TIME - TIMESTAMP_TOLERANCE): issues.append(f"[L{line_num} @ {current_event_time:.4f}] E{eid} CLOSE too fast at {floor} ({time_since_open:.4f}s < {MIN_DOOR_TIME}s): {raw_line}")

                # <<< NEW Check 4: Double Car Range Check >>> (Less critical for CLOSE, but for consistency)
                if e['is_double_car']:
                     if not (FLOOR_ORDER[e['min_floor']] <= FLOOR_ORDER[floor] <= FLOOR_ORDER[e['max_floor']]):
                          issues.append(f"[L{line_num} @ {current_event_time:.4f}] VIOLATION (WARN): E{eid}(Double) CLOSE at {floor} outside its allowed range ({e['min_floor']}-{e['max_floor']}): {raw_line}")

                # Update state
                e['door'] = 'CLOSED'
                e['last_close_ts'] = current_event_time
                e['motion'] = 'IDLE' # Assume IDLE until next ARRIVE confirms movement
                e['last_action_ts'] = current_event_time

                # <<< MODIFIED: Update departure time >>>
                # Closing the door signifies the potential start of movement
                e['last_departure_ts'] = current_event_time


            # --- IN --- <<< MODIFIED: Add Double-Car Range Check >>>
            elif matched_action == 'IN':
                pid_str, floor, eid_str = action_data; pid = int(pid_str)
                if e is None: continue
                # Check 0: UPDATE Silent Phase Violation (Unchanged)
                if e['update_status'] == 'BEGUN':
                     issues.append(f"[L{line_num} @ {current_event_time:.4f}] UPDATE VIOLATION: P{pid} IN E{eid} during UPDATE silent phase: {raw_line}")
                     continue
                # Perform other checks, set err_in flag on failure (Unchanged logic)
                err_in = False
                if e['floor'] != floor: issues.append(f"[L{line_num} @ {current_event_time:.4f}] P{pid} IN E{eid} at {floor}, E@ {e['floor']}"); err_in=True
                if e['door'] != 'OPEN': issues.append(f"[L{line_num} @ {current_event_time:.4f}] P{pid} IN E{eid} at {floor}, door {e['door']}"); err_in=True
                if p is None: issues.append(f"[L{line_num} @ {current_event_time:.4f}] P{pid} IN E{eid}, unknown passenger ID"); err_in=True
                else:
                    if p['location'] != floor: issues.append(f"[L{line_num} @ {current_event_time:.4f}] P{pid} IN E{eid} at {floor}, P.loc '{p['location']}'"); err_in=True
                    if p['state'] != 'WAITING': issues.append(f"[L{line_num} @ {current_event_time:.4f}] P{pid} IN E{eid}, state {p['state']} != WAITING"); err_in=True
                    assigned_eid = received_requests.get(pid)
                    if assigned_eid is None: issues.append(f"[L{line_num} @ {current_event_time:.4f}] P{pid} IN E{eid} without RECEIVE"); err_in=True
                    elif assigned_eid != eid: issues.append(f"[L{line_num} @ {current_event_time:.4f}] P{pid} IN E{eid}, but RECEIVE'd by E{assigned_eid}"); err_in=True
                if len(e['passengers']) >= e['capacity']: issues.append(f"[L{line_num} @ {current_event_time:.4f}] E{eid} full (cap {e['capacity']}), cannot IN P{pid}"); err_in=True
                if e['sche_active'] and e['sche_phase'] == 'BEGUN': issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: P{pid} IN E{eid} during SCHE"); err_in=True

                # <<< NEW Check: Double Car Range Check >>>
                if e['is_double_car']:
                     if not (FLOOR_ORDER[e['min_floor']] <= FLOOR_ORDER[floor] <= FLOOR_ORDER[e['max_floor']]):
                           issues.append(f"[L{line_num} @ {current_event_time:.4f}] VIOLATION: P{pid} IN E{eid}(Double) at {floor}, outside elevator range ({e['min_floor']}-{e['max_floor']})"); err_in = True

                # Update state only if no critical errors and passenger known (Unchanged logic)
                if not err_in and p is not None:
                    e['passengers'].add(pid); p['location'] = eid; p['state'] = 'TRANSIT'; p['assigned_elevator'] = eid
                e['last_action_ts'] = current_event_time


            # --- OUT_S / OUT_F --- <<< MODIFIED: Add Double-Car Range Check >>>
            elif matched_action in ['OUT_S', 'OUT_F']:
                pid_str, floor, eid_str = action_data; pid = int(pid_str)
                if e is None: continue
                # Check 0: UPDATE Silent Phase Check (Allow OUT during ACCEPTED) (Unchanged)
                if e['update_status'] == 'BEGUN':
                     issues.append(f"[L{line_num} @ {current_event_time:.4f}] UPDATE VIOLATION: P{pid} {matched_action} E{eid} during UPDATE silent phase: {raw_line}")
                     # Allow state update but log violation

                # Perform other checks, set err_out flag on failure (Unchanged logic)
                err_out = False
                if e['floor'] != floor: issues.append(f"[L{line_num} @ {current_event_time:.4f}] P{pid} {matched_action} E{eid} at {floor}, E@ {e['floor']}"); err_out = True
                if e['door'] != 'OPEN': issues.append(f"[L{line_num} @ {current_event_time:.4f}] P{pid} {matched_action} E{eid} at {floor}, door {e['door']}"); err_out = True
                if pid not in e['passengers']: issues.append(f"[L{line_num} @ {current_event_time:.4f}] P{pid} {matched_action} E{eid} at {floor}, P not inside"); err_out = True
                elif p is None: issues.append(f"[L{line_num} @ {current_event_time:.4f}] P{pid} {matched_action} E{eid}, unknown passenger ID"); err_out = True
                elif p['state'] != 'TRANSIT' or p['location'] != eid: issues.append(f"[L{line_num} @ {current_event_time:.4f}] P{pid} {matched_action} E{eid}, bad P state (st={p['state']}, loc={p['location']})"); err_out = True

                # SCHE Check (Unchanged)
                is_sche_out_at_target = e['sche_active'] and e['sche_phase'] == 'BEGUN' and e['floor'] == e['sche_target_floor'] and e['door'] == 'OPEN'
                if e['sche_active'] and e['sche_phase'] == 'BEGUN' and not is_sche_out_at_target:
                     issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: P{pid} {matched_action} E{eid} during SCHE move/before target open"); err_out = True

                # <<< NEW Check: Double Car Range Check >>>
                if e['is_double_car']:
                     if not (FLOOR_ORDER[e['min_floor']] <= FLOOR_ORDER[floor] <= FLOOR_ORDER[e['max_floor']]):
                          issues.append(f"[L{line_num} @ {current_event_time:.4f}] VIOLATION: P{pid} {matched_action} E{eid}(Double) at {floor}, outside elevator range ({e['min_floor']}-{e['max_floor']})"); err_out = True

                # Clear receive state (Unchanged)
                if pid in received_requests:
                     del received_requests[pid]
                     if p is not None and p['assigned_elevator'] == eid: p['assigned_elevator'] = None

                # Update elevator state (remove passenger) (Unchanged)
                passenger_was_inside = False
                if pid in e['passengers']: e['passengers'].remove(pid); passenger_was_inside = True

                # Check if elevator becomes empty after UPDATE-ACCEPT (Unchanged)
                if passenger_was_inside and e['update_status'] == 'ACCEPTED' and not e['passengers']:
                     e['update_emptied_after_accept'] = True

                # Update passenger state based on OUT type (Unchanged logic)
                if p is not None:
                    p['location'] = floor
                    if p['assigned_elevator'] == eid: p['assigned_elevator'] = None
                    is_destination = (floor == p['destination'])
                    if matched_action == 'OUT_S':
                        if is_destination: p['state'] = 'ARRIVED'
                        else: issues.append(f"[L{line_num} @ {current_event_time:.4f}] SEMANTIC VIOLATION: P{pid} OUT_S at {floor} != dest {p['destination']}. Setting WAITING."); p['state'] = 'WAITING'
                    elif matched_action == 'OUT_F':
                        if not is_destination: p['state'] = 'WAITING'
                        else: issues.append(f"[L{line_num} @ {current_event_time:.4f}] SEMANTIC VIOLATION: P{pid} OUT_F at destination {floor}. Setting ARRIVED."); p['state'] = 'ARRIVED'
                elif not err_out:
                    issues.append(f"[L{line_num} @ {current_event_time:.4f}] WARN: P{pid} {matched_action} E{eid}, unknown passenger ID, state not updated.")

                e['last_action_ts'] = current_event_time

            # --- RECEIVE --- (Unchanged - Logic seems correct)
            elif matched_action == 'RECEIVE':
                pid_str, eid_str = action_data; pid = int(pid_str)
                if p is None or e is None: continue
                if e['update_status'] == 'BEGUN':
                     issues.append(f"[L{line_num} @ {current_event_time:.4f}] UPDATE VIOLATION: E{eid} RECEIVE P{pid} during UPDATE silent phase: {raw_line}")
                     continue

                is_needed_to_move = not e['passengers'] and e['motion'] == 'IDLE' and not e['sche_active'] and not e['just_updated']
                if not is_needed_to_move: pass # Allow redundant RECEIVE

                err_receive = False
                if not isinstance(p['location'], str) or not re.match(r'[BF]\d+', p['location']): issues.append(f"[L{line_num} @ {current_event_time:.4f}] RECEIVE-{pid}-{eid} invalid: P{pid} not on floor (loc={p['location']})"); err_receive=True
                if p['state'] != 'WAITING': issues.append(f"[L{line_num} @ {current_event_time:.4f}] RECEIVE-{pid}-{eid} invalid: P{pid} state is {p['state']} (must be WAITING)"); err_receive=True
                current_assignment = received_requests.get(pid)
                if current_assignment is not None and current_assignment != eid: issues.append(f"[L{line_num} @ {current_event_time:.4f}] RECEIVE-{pid}-{eid} invalid: P{pid} already assigned to E{current_assignment}"); err_receive=True
                if e['sche_active'] and e['sche_phase'] == 'BEGUN': issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} RECEIVE P{pid} during SCHE move"); err_receive=True

                if not err_receive:
                     received_requests[pid] = eid; p['assigned_elevator'] = eid
                e['last_action_ts'] = current_event_time

            # --- SCHE-ACCEPT --- (Unchanged)
            elif matched_action == 'ACCEPT': # SCHE-ACCEPT
                eid_str, speed_str, target_floor = action_data
                if e is None: continue
                if e['update_status'] is not None: issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE/UPDATE CONFLICT: E{eid} received SCHE-ACCEPT while in UPDATE status '{e['update_status']}': {raw_line}"); continue
                try: speed = float(speed_str)
                except ValueError: issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE-ACCEPT E{eid} invalid speed '{speed_str}'"); continue
                if target_floor not in FLOOR_ORDER: issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE-ACCEPT E{eid} invalid target floor '{target_floor}'"); continue
                if target_floor not in ALLOWED_SCHE_TARGET_FLOORS: issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} ACCEPT target floor '{target_floor}' not allowed.")
                if speed not in SCHEDULE_SPEEDS: issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} ACCEPT invalid speed '{speed}'")
                if e['sche_active']: issues.append(f"[L{line_num} @ {current_event_time:.4f}] WARN: E{eid} new SCHE-ACCEPT while active (Phase: {e['sche_phase']}). Overwriting.")
                e['sche_active'] = True; e['sche_phase'] = 'ACCEPTED'; e['sche_target_floor'] = target_floor
                e['sche_speed'] = speed; e['sche_accept_ts'] = current_event_time
                e['sche_begin_ts'] = -1.0; e['sche_end_ts'] = -1.0; e['sche_arrives_since_accept'] = 0; e['sche_opened_at_target'] = False
                e['last_action_ts'] = current_event_time

            # --- SCHE-BEGIN --- <<< MODIFIED: Update last_departure_ts >>>
            elif matched_action == 'BEGIN': # SCHE-BEGIN
                eid_str, = action_data
                if e is None: continue
                if e['update_status'] is not None: issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE/UPDATE CONFLICT: E{eid} received SCHE-BEGIN while in UPDATE status '{e['update_status']}': {raw_line}"); continue

                valid_begin = True
                if not e['sche_active'] or e['sche_phase'] != 'ACCEPTED': issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} BEGIN invalid state (active={e['sche_active']}, phase={e['sche_phase']})"); valid_begin = False
                if e['door'] != 'CLOSED': issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} BEGIN door not CLOSED ({e['door']})"); valid_begin = False
                if e['motion'] != 'IDLE': issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} BEGIN while {e['motion']}"); valid_begin = False
                # Arrive limit check already done in ARRIVE

                if valid_begin:
                    e['sche_phase'] = 'BEGUN'; e['sche_begin_ts'] = current_event_time
                    if e['sche_speed'] is not None: e['speed'] = e['sche_speed']
                    else: issues.append(f"[L{line_num} @ {current_event_time:.4f}] INTERNAL ERR: E{eid} SCHE-BEGIN sche_speed is None."); e['speed'] = DEFAULT_SPEED

                    if e['floor'] != e['sche_target_floor']:
                         e['motion'] = 'MOVING'
                         # <<< MODIFIED: Update departure time >>>
                         e['last_departure_ts'] = current_event_time
                    else:
                         e['motion'] = 'IDLE'
                         # Reset departure time if starting idle at target? Or leave as is? Let's leave it.

                    # Cancel receives (Unchanged)
                    pids_to_cancel = [pid_c for pid_c, assigned_eid in received_requests.items() if assigned_eid == eid]
                    for pid_cancel in pids_to_cancel:
                         del received_requests[pid_cancel]
                         if pid_cancel in passengers and passengers[pid_cancel]['state'] == 'WAITING': passengers[pid_cancel]['assigned_elevator'] = None
                else:
                    issues.append(f"[L{line_num} @ {current_event_time:.4f}] E{eid} failed SCHE-BEGIN checks, state remains {e['sche_phase']}")
                e['last_action_ts'] = current_event_time

            # --- SCHE-END --- (Unchanged)
            elif matched_action == 'END': # SCHE-END
                eid_str, = action_data
                if e is None: continue
                if e['update_status'] is not None:
                      issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE/UPDATE CONFLICT: E{eid} received SCHE-END while in UPDATE status '{e['update_status']}': {raw_line}")
                      # Reset SCHE state anyway
                      e['sche_active'] = False; e['sche_phase'] = None; e['sche_target_floor'] = None; e['sche_speed'] = None
                      e['sche_accept_ts'] = -1.0; e['sche_begin_ts'] = -1.0; e['sche_end_ts'] = current_event_time
                      e['sche_arrives_since_accept'] = 0; e['sche_opened_at_target'] = False
                      e['speed'] = DEFAULT_SPEED; e['motion'] = 'IDLE'; e['last_action_ts'] = current_event_time
                      continue
                valid_end = True
                if not e['sche_active'] or e['sche_phase'] != 'BEGUN': issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} END invalid state (active={e['sche_active']}, phase={e['sche_phase']})"); valid_end = False
                if e['floor'] != e['sche_target_floor']: issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} END at {e['floor']} != target {e['sche_target_floor']}"); valid_end = False
                if e['door'] != 'CLOSED': issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} END door not CLOSED ({e['door']})"); valid_end = False
                if e['passengers']: issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} END with passengers: {list(e['passengers'])}"); valid_end = False
                if e['floor'] == e['sche_target_floor'] and not e['sche_opened_at_target']: issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} END at target {e['floor']} without OPENING door during SCHE."); valid_end = False
                if e['sche_accept_ts'] <= 0: issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE INTERNAL ERR: E{eid} END check bad accept_ts ({e['sche_accept_ts']})."); valid_end = False
                else:
                    completion_time = current_event_time - e['sche_accept_ts']
                    if completion_time > (SCHE_MAX_COMPLETE_TIME + TIMESTAMP_TOLERANCE): issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE TIMING VIOLATION: E{eid} completion time {completion_time:.4f}s > {SCHE_MAX_COMPLETE_TIME}s"); valid_end = False

                # Reset state (Unchanged)
                e['sche_active'] = False; e['sche_phase'] = None; e['sche_target_floor'] = None; e['sche_speed'] = None
                e['sche_accept_ts'] = -1.0; e['sche_begin_ts'] = -1.0; e['sche_end_ts'] = current_event_time
                e['sche_arrives_since_accept'] = 0; e['sche_opened_at_target'] = False
                e['speed'] = DEFAULT_SPEED; e['motion'] = 'IDLE'; e['last_action_ts'] = current_event_time


            # --- UPDATE_ACCEPT --- (Unchanged)
            elif matched_action == 'UPDATE_ACCEPT':
                 eid_a_str, eid_b_str, target_floor = action_data
                 if e_a is None or e_b is None: continue
                 accept_ok = True
                 if e_a['update_status'] is not None or e_b['update_status'] is not None: issues.append(f"[L{line_num} @ {current_event_time:.4f}] UPDATE VIOLATION: ACCEPT received, but E{eid_a} or E{eid_b} already in UPDATE."); accept_ok = False
                 if e_a['sche_active'] or e_b['sche_active']: issues.append(f"[L{line_num} @ {current_event_time:.4f}] UPDATE VIOLATION: ACCEPT received, but E{eid_a} or E{eid_b} in SCHE."); accept_ok = False
                 if target_floor not in FLOOR_ORDER: issues.append(f"[L{line_num} @ {current_event_time:.4f}] UPDATE VIOLATION: Invalid target floor '{target_floor}' in ACCEPT."); accept_ok = False
                 elif target_floor not in ALLOWED_UPDATE_TARGET_FLOORS: issues.append(f"[L{line_num} @ {current_event_time:.4f}] UPDATE VIOLATION: Target floor '{target_floor}' not allowed for UPDATE."); accept_ok = False

                 if accept_ok:
                      ts_accept = current_event_time
                      for elev, role, partner_id in [(e_a, 'A', eid_b), (e_b, 'B', eid_a)]:
                           elev['update_status'] = 'ACCEPTED'; elev['update_role'] = role
                           elev['update_partner_id'] = partner_id; elev['update_target_floor'] = target_floor
                           elev['update_accept_ts'] = ts_accept; elev['update_begin_ts'] = -1.0
                           elev['update_end_ts'] = -1.0; elev['update_arrives_since_accept'] = 0
                           elev['update_emptied_after_accept'] = not elev['passengers']
                           elev['last_action_ts'] = current_event_time
                      issues.append(f"[L{line_num} @ {ts_accept:.4f}] INFO: Recognized UPDATE-ACCEPT E{eid_a}(A)-E{eid_b}(B)-{target_floor}. Elevators must now empty.")
                 else: issues.append(f"[L{line_num} @ {current_event_time:.4f}] UPDATE: Ignoring invalid ACCEPT for E{eid_a}/E{eid_b}.")

            # --- UPDATE_BEGIN --- (Unchanged)
            elif matched_action == 'UPDATE_BEGIN':
                 eid_a_str, eid_b_str = action_data
                 if e_a is None or e_b is None: continue
                 valid_begin = True
                 if not (e_a['update_status'] == 'ACCEPTED' and e_b['update_status'] == 'ACCEPTED'): issues.append(f"[L{line_num} @ {current_event_time:.4f}] UPDATE VIOLATION: BEGIN state not ACCEPTED (A:{e_a['update_status']}, B:{e_b['update_status']})"); valid_begin = False
                 elif not (e_a['update_role'] == 'A' and e_b['update_role'] == 'B' and e_a['update_partner_id'] == eid_b and e_b['update_partner_id'] == eid_a): issues.append(f"[L{line_num} @ {current_event_time:.4f}] UPDATE VIOLATION: BEGIN role/partner mismatch"); valid_begin = False
                 if not e_a['update_emptied_after_accept']: issues.append(f"[L{line_num} @ {current_event_time:.4f}] UPDATE VIOLATION: E{eid_a}(A) BEGIN before emptied after ACCEPT"); valid_begin = False
                 if not e_b['update_emptied_after_accept']: issues.append(f"[L{line_num} @ {current_event_time:.4f}] UPDATE VIOLATION: E{eid_b}(B) BEGIN before emptied after ACCEPT"); valid_begin = False
                 if e_a['passengers']: issues.append(f"[L{line_num} @ {current_event_time:.4f}] UPDATE VIOLATION: E{eid_a}(A) BEGIN with passengers inside."); valid_begin = False
                 if e_b['passengers']: issues.append(f"[L{line_num} @ {current_event_time:.4f}] UPDATE VIOLATION: E{eid_b}(B) BEGIN with passengers inside."); valid_begin = False
                 if e_a['door'] != 'CLOSED': issues.append(f"[L{line_num} @ {current_event_time:.4f}] UPDATE VIOLATION: E{eid_a}(A) BEGIN door {e_a['door']}"); valid_begin = False
                 if e_b['door'] != 'CLOSED': issues.append(f"[L{line_num} @ {current_event_time:.4f}] UPDATE VIOLATION: E{eid_b}(B) BEGIN door {e_b['door']}"); valid_begin = False
                 if e_a['motion'] != 'IDLE': issues.append(f"[L{line_num} @ {current_event_time:.4f}] UPDATE VIOLATION: E{eid_a}(A) BEGIN while {e_a['motion']}"); valid_begin = False
                 if e_b['motion'] != 'IDLE': issues.append(f"[L{line_num} @ {current_event_time:.4f}] UPDATE VIOLATION: E{eid_b}(B) BEGIN while {e_b['motion']}"); valid_begin = False

                 if valid_begin:
                      ts_begin = current_event_time
                      e_a['update_status'] = 'BEGUN'; e_a['update_begin_ts'] = ts_begin; e_a['last_action_ts'] = ts_begin
                      e_b['update_status'] = 'BEGUN'; e_b['update_begin_ts'] = ts_begin; e_b['last_action_ts'] = ts_begin
                      # Cancel RECEIVEs
                      pids_to_cancel_update = [pid_c for pid_c, assigned_eid in received_requests.items() if assigned_eid in [eid_a, eid_b]]
                      cancelled_count = 0
                      for pid_cancel in pids_to_cancel_update:
                           del received_requests[pid_cancel]
                           if pid_cancel in passengers and passengers[pid_cancel]['state'] == 'WAITING': passengers[pid_cancel]['assigned_elevator'] = None
                           cancelled_count += 1
                      if cancelled_count > 0: issues.append(f"[L{line_num} @ {ts_begin:.4f}] INFO: Cancelled {cancelled_count} RECEIVE(s) for E{eid_a}/E{eid_b} due to UPDATE-BEGIN.")
                 else: issues.append(f"[L{line_num} @ {current_event_time:.4f}] UPDATE: Failed BEGIN checks for E{eid_a}/E{eid_b}.")

            # --- UPDATE_END --- <<< MODIFIED: Update last_departure_ts >>>
            elif matched_action == 'UPDATE_END':
                 eid_a_str, eid_b_str = action_data
                 if e_a is None or e_b is None: continue
                 valid_end = True
                 if not (e_a['update_status'] == 'BEGUN' and e_b['update_status'] == 'BEGUN'): issues.append(f"[L{line_num} @ {current_event_time:.4f}] UPDATE VIOLATION: END state not BEGUN (A:{e_a['update_status']}, B:{e_b['update_status']})"); valid_end = False
                 ts_end = current_event_time
                 if e_a['update_accept_ts'] <= 0 or e_b['update_accept_ts'] <= 0: issues.append(f"[L{line_num} @ {ts_end:.4f}] UPDATE INTERNAL ERR: Bad accept_ts for END check."); valid_end = False
                 elif e_a['update_begin_ts'] <= 0 or e_b['update_begin_ts'] <= 0: issues.append(f"[L{line_num} @ {ts_end:.4f}] UPDATE INTERNAL ERR: Bad begin_ts for END check."); valid_end = False
                 else:
                      completion_time = ts_end - e_a['update_accept_ts']
                      reset_time = ts_end - e_a['update_begin_ts']
                      if completion_time > (UPDATE_T_COMPLETE_MAX + TIMESTAMP_TOLERANCE): issues.append(f"[L{line_num} @ {ts_end:.4f}] UPDATE TIMING VIOLATION: Completion time {completion_time:.4f}s > {UPDATE_T_COMPLETE_MAX}s"); valid_end = False
                      if reset_time < (UPDATE_T_RESET_MIN - TIMESTAMP_TOLERANCE): issues.append(f"[L{line_num} @ {ts_end:.4f}] UPDATE TIMING VIOLATION: Reset time {reset_time:.4f}s < {UPDATE_T_RESET_MIN}s"); valid_end = False
                 if e_a['passengers'] or e_b['passengers']: issues.append(f"[L{line_num} @ {ts_end:.4f}] UPDATE VIOLATION: END with passengers (A:{list(e_a['passengers'])}, B:{list(e_b['passengers'])})."); valid_end = False
                 if e_a['door'] != 'CLOSED' or e_b['door'] != 'CLOSED': issues.append(f"[L{line_num} @ {ts_end:.4f}] UPDATE VIOLATION: END with door open (A:{e_a['door']}, B:{e_b['door']})."); valid_end = False

                 if valid_end:
                      target_floor = e_a['update_target_floor']
                      if target_floor is None or target_floor not in FLOOR_ORDER:
                           issues.append(f"[L{line_num} @ {ts_end:.4f}] UPDATE INTERNAL ERR: Cannot determine target floor for END state update.");
                           for elev in [e_a, e_b]: elev['update_status'] = None; elev['update_end_ts'] = ts_end
                      else:
                            target_floor_idx = FLOOR_ORDER[target_floor]
                            floor_b_idx = target_floor_idx - 1
                            floor_a_idx = target_floor_idx + 1
                            if floor_b_idx not in FLOOR_INDEX_TO_NAME or floor_a_idx not in FLOOR_INDEX_TO_NAME:
                                issues.append(f"[L{line_num} @ {ts_end:.4f}] UPDATE INTERNAL ERR: Calculated post-update floor index out of bounds for target {target_floor}. A:{floor_a_idx}, B:{floor_b_idx}");
                                for elev in [e_a, e_b]: elev['update_status'] = None; elev['update_end_ts'] = ts_end
                            else:
                                final_floor_b = FLOOR_INDEX_TO_NAME[floor_b_idx]
                                final_floor_a = FLOOR_INDEX_TO_NAME[floor_a_idx]

                                # Elevator A (Upper) - State Update
                                e_a['is_active'] = True; e_a['floor'] = final_floor_a; e_a['shaft_id'] = eid_b
                                e_a['is_double_car'] = True; e_a['speed'] = DOUBLE_CAR_SPEED
                                e_a['min_floor'] = target_floor; e_a['max_floor'] = 'F7'
                                e_a['just_updated'] = True; e_a['motion'] = 'IDLE'; e_a['door'] = 'CLOSED'
                                e_a['last_action_ts'] = ts_end; e_a['last_floor_ts'] = ts_end
                                e_a['last_departure_ts'] = ts_end # <<< MODIFIED: Update departure time >>>

                                # Elevator B (Lower) - State Update
                                e_b['is_active'] = True; e_b['floor'] = final_floor_b; e_b['shaft_id'] = eid_b
                                e_b['is_double_car'] = True; e_b['speed'] = DOUBLE_CAR_SPEED
                                e_b['min_floor'] = 'B4'; e_b['max_floor'] = target_floor
                                e_b['just_updated'] = True; e_b['motion'] = 'IDLE'; e_b['door'] = 'CLOSED'
                                e_b['last_action_ts'] = ts_end; e_b['last_floor_ts'] = ts_end
                                e_b['last_departure_ts'] = ts_end # <<< MODIFIED: Update departure time >>>

                      # Reset update status fields, keep role/partner
                      for elev in [e_a, e_b]:
                           elev['update_status'] = None; elev['update_target_floor'] = None
                           elev['update_accept_ts'] = -1.0; elev['update_begin_ts'] = -1.0
                           elev['update_end_ts'] = ts_end; elev['update_arrives_since_accept'] = 0
                           elev['update_emptied_after_accept'] = False
                 else:
                      issues.append(f"[L{line_num} @ {ts_end:.4f}] UPDATE: Failed END checks for E{eid_a}/E{eid_b}.")
                      for elev in [e_a, e_b]: elev['update_status'] = None; elev['update_end_ts'] = ts_end


        except (ValueError, IndexError, TypeError, KeyError) as e_state:
             issues.append(f"[L{line_num} @ {current_time:.4f}] STATE/PARSE ERROR action '{matched_action}': {type(e_state).__name__}: {e_state} in: {raw_line}")
             issues.append(traceback.format_exc())

    # --- Final Checks --- <<< MODIFIED: Final check message >>>
    final_time = current_time
    if final_time > T_MAX_HUTEST: issues.append(f"FINAL CHECK: Time Limit Exceeded: {final_time:.4f}s > {T_MAX_HUTEST}s.")

    # Check passengers (Unchanged)
    for pid, p_state in passengers.items():
        if p_state['state'] != 'ARRIVED':
             issues.append(f"FINAL CHECK: P{pid} not ARRIVED (State: {p_state['state']}, Loc: {p_state['location']}, Dest: {p_state['destination']}).")
        elif p_state['location'] != p_state['destination']:
             issues.append(f"FINAL CHECK: P{pid} ARRIVED but final location '{p_state['location']}' != destination '{p_state['destination']}'.")

    # Check elevators
    active_elevator_ids_final = set()
    shaft_occupancy = {}
    final_elevator_states_str = {} # Store string representation for potential debug output

    for eid, e_state in elevators.items():
        is_logically_active = e_state['is_active']
        # Store state string *before* adding to active set
        final_elevator_states_str[eid] = f"E{eid}[Active:{is_logically_active}, Floor:{e_state['floor']}, Speed:{e_state['speed']:.2f}, Shaft:{e_state['shaft_id']}, Double:{e_state['is_double_car']}, Partner:{e_state['update_partner_id']}, Role:{e_state['update_role']}, Range:{e_state['min_floor']}-{e_state['max_floor']}, Door:{e_state['door']}, Psngrs:{len(e_state['passengers'])}]"

        if is_logically_active:
             active_elevator_ids_final.add(eid)
             current_shaft = e_state['shaft_id']
             if current_shaft not in shaft_occupancy: shaft_occupancy[current_shaft] = set()
             shaft_occupancy[current_shaft].add(eid)

        if is_logically_active:
            if e_state['passengers']: issues.append(f"FINAL CHECK: Active E{eid} finished with passengers: {list(e_state['passengers'])}")
            if e_state['door'] != 'CLOSED': issues.append(f"FINAL CHECK: Active E{eid} finished door {e_state['door']}")
        if e_state['sche_active']: issues.append(f"FINAL CHECK: E{eid} finished SCHE active (Phase: {e_state['sche_phase']})")
        if e['update_status'] is not None: issues.append(f"FINAL CHECK: E{eid} finished in UPDATE status '{e['update_status']}'")


    # Iterate again for consistency checks (reading state, not modifying)
    # This second loop is fine as state isn't modified between loops.
    for eid, e_state in elevators.items():
        if e_state['is_double_car']:
             partner_id = e_state['update_partner_id']
             if partner_id is None or partner_id not in elevators:
                  issues.append(f"FINAL CHECK: Double Car E{eid} missing valid partner ID.")
             # <<< MODIFIED: More informative message if partner not in final active set >>>
             elif partner_id not in active_elevator_ids_final:
                  partner_final_state_str = final_elevator_states_str.get(partner_id, "Partner state not found (partner ID invalid?)")
                  issues.append(f"FINAL CHECK: Double Car E{eid}'s partner E{partner_id} was not considered active at end. Partner's Final State: {partner_final_state_str}")
             # Check consistency only if partner is considered active
             elif elevators[partner_id]['shaft_id'] != e_state['shaft_id']:
                  issues.append(f"FINAL CHECK: Double Car E{eid}'s partner E{partner_id} ended in different shaft {elevators[partner_id]['shaft_id']} vs {e_state['shaft_id']}.")
             elif not elevators[partner_id]['is_double_car'] or elevators[partner_id]['update_partner_id'] != eid:
                  issues.append(f"FINAL CHECK: Double Car E{eid}'s partner E{partner_id} state inconsistent (not double or wrong partner ID).")
             # Final position check now done within shaft check loop below


    # Shaft/Double Car Checks (structure unchanged, relies on active_elevator_ids_final)
    occupied_shafts = set(elevators[eid]['shaft_id'] for eid in active_elevator_ids_final)
    for shaft_id in occupied_shafts:
        occupants = shaft_occupancy.get(shaft_id, set())
        if not occupants: continue

        if len(occupants) > 2: issues.append(f"FINAL CHECK: Shaft {shaft_id} has >2 occupants: {occupants}")
        elif len(occupants) == 2:
            e1_id, e2_id = list(occupants)
            # Ensure both are actually marked as active before proceeding
            if e1_id not in active_elevator_ids_final or e2_id not in active_elevator_ids_final:
                 issues.append(f"FINAL CHECK: Shaft {shaft_id} occupants {occupants} inconsistency - one or both not in active set.")
                 continue # Skip detailed checks if basic activity is inconsistent

            e1, e2 = elevators[e1_id], elevators[e2_id]
            # Double car config consistency (already partially checked above, but good redundancy)
            if not (e1['is_double_car'] and e2['is_double_car'] and e1['update_partner_id'] == e2_id and e2['update_partner_id'] == e1_id):
                 issues.append(f"FINAL CHECK: Shaft {shaft_id} occupants {occupants} config consistency fail (is_double_car/partner_id).")
            # Role check
            roles = {e1.get('update_role'), e2.get('update_role')}
            if roles != {'A', 'B'}:
                 issues.append(f"FINAL CHECK: Double cars E{e1_id}, E{e2_id} role mismatch. Roles: E{e1_id}:{e1.get('update_role')}, E{e2_id}:{e2.get('update_role')}.")
            # Final collision/position check
            if e1['floor'] == e2['floor']:
                 issues.append(f"FINAL CHECK: Double cars E{e1_id}, E{e2_id} ended on same floor {e1['floor']}.")
            else:
                 # Re-verify relative position based on roles A/B
                 upper_e = e1 if e1.get('update_role') == 'A' else (e2 if e2.get('update_role') == 'A' else None)
                 lower_e = e1 if e1.get('update_role') == 'B' else (e2 if e2.get('update_role') == 'B' else None)
                 if upper_e and lower_e: # Check only if roles are correct A and B
                      upper_floor_idx = FLOOR_ORDER.get(upper_e['floor'])
                      lower_floor_idx = FLOOR_ORDER.get(lower_e['floor'])
                      if upper_floor_idx is not None and lower_floor_idx is not None and upper_floor_idx < lower_floor_idx:
                           issues.append(f"FINAL CHECK: Double cars ended upper E{upper_e['id']}({upper_e['floor']}) below lower E{lower_e['id']}({lower_e['floor']}).")
                 # else: Role mismatch already logged above

        elif len(occupants) == 1:
             single_eid = list(occupants)[0]
             if elevators[single_eid]['is_double_car']:
                   issues.append(f"FINAL CHECK: E{single_eid} in Shaft {shaft_id} marked as double_car but partner missing/inactive.")


    # Check pending receives (Unchanged)
    passengers_assigned_pending = [p_id for p_id, assigned_eid in received_requests.items() if assigned_eid is not None]
    if passengers_assigned_pending: issues.append(f"FINAL CHECK: Finished with pending RECEIVE assignments for Ps: {passengers_assigned_pending}")

    return issues


# --- Task Function for Thread Pool ---
# (_run_one_batch_test_task remains largely unchanged but calls the updated validate_output)
def _run_one_batch_test_task(test_id, input_lines, jar_path, datainput_exe_path, stop_event_ref):
    """
    Runs a single test case (datainput | java -jar ...) in a temporary directory.
    Determines success based on Java exit code 0 AND no validation VIOLATIONS/ERRORS (ignores WARNs).
    """
    output_log_lines = []
    validation_issues = ["Task did not complete successfully."] # Default if validation doesn't run
    success = False
    return_code_java = -1 # Default indicates not run or crashed before exit code retrieval
    return_code_datainput = -1
    p_datainput = None
    p_java = None
    execution_finished_without_error = False # Tracks if process finished without timeout/major setup error
    start_time = time.monotonic()
    abs_jar_path = os.path.abspath(jar_path)
    abs_datainput_path = os.path.abspath(datainput_exe_path)
    stdout_java_remaining = ""
    stderr_java_str = ""
    stderr_data_str = ""
    # Define timeouts early (assuming T_MAX_HUTEST is globally defined)
    java_wait_timeout = T_MAX_HUTEST + 30.0
    datainput_wait_timeout = 10.0

    try:
        # --- Pre-checks ---
        if not os.path.exists(abs_datainput_path):
             raise FileNotFoundError(f"datainput.exe not found: {abs_datainput_path}")
        if not os.path.exists(abs_jar_path):
             raise FileNotFoundError(f"Elevator JAR not found: {abs_jar_path}")
        if stop_event_ref.is_set():
             raise InterruptedError("Batch stop requested before task start")

        # --- Create Temporary Directory ---
        with tempfile.TemporaryDirectory() as temp_dir:
            stdin_path = os.path.join(temp_dir, STDIN_FILENAME) # Assuming STDIN_FILENAME is global
            try:
                # Assuming PROCESS_ENCODING is global
                with open(stdin_path, 'w', encoding=PROCESS_ENCODING, errors='replace') as f:
                    for line in input_lines: f.write(line + '\n')
            except IOError as e:
                raise IOError(f"Failed to write stdin file {stdin_path}: {e}") from e

            # --- Prepare Commands & Start Processes ---
            datainput_cmd = [abs_datainput_path]
            java_cmd = ['java', '-jar', abs_jar_path]
            common_popen_kwargs = {
                "cwd": temp_dir, "text": True, "encoding": PROCESS_ENCODING,
                "errors": 'replace',
                "creationflags": subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            }

            p_datainput = subprocess.Popen(
                datainput_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **common_popen_kwargs
            )
            # Check immediate termination
            if p_datainput.poll() is not None:
                 try: stderr_data_str = p_datainput.stderr.read() if p_datainput.stderr else ""
                 except Exception: pass
                 raise RuntimeError(f"DataInput terminated immediately. RC={p_datainput.returncode}. Stderr: {stderr_data_str[:500]}")
            if p_datainput.stdout is None: raise RuntimeError("DataInput failed to redirect stdout.")

            p_java = subprocess.Popen(
                java_cmd, stdin=p_datainput.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **common_popen_kwargs
            )
            # Close *our* handle to datainput's stdout. Java process still holds its end.
            # Need to be careful closing streams. Let communicate handle stdin pipe closing implicitly.
            # if p_datainput.stdout: p_datainput.stdout.close() # Potentially risky, might cause issues if java reads slowly

            # Check immediate termination
            if p_java.poll() is not None:
                 try: stderr_java_str = p_java.stderr.read() if p_java.stderr else ""
                 except Exception: pass
                 # Ensure datainput is terminated if java fails early
                 if p_datainput.poll() is None: p_datainput.kill()
                 raise RuntimeError(f"Java terminated immediately. RC={p_java.returncode}. Stderr: {stderr_java_str[:500]}")
            if p_java.stdout is None: raise RuntimeError("Java failed to redirect stdout.")

            # --- Read Java Output Line-by-Line ---
            try:
                # Read stdout line by line as it comes in
                while True:
                    if p_java.stdout is None: break # Exit if stdout closed
                    line = p_java.stdout.readline()
                    if not line: break # End of stream
                    if stop_event_ref.is_set():
                        # Attempt graceful termination
                        if p_java.poll() is None: p_java.terminate()
                        if p_datainput.poll() is None: p_datainput.terminate()
                        raise InterruptedError("Batch stop requested during output reading")
                    line = line.strip() # Strip whitespace/newlines
                    if line: output_log_lines.append(line)
            except IOError as e: # Catch direct IO errors on the stream e.g. reading after killed
                output_log_lines.append(f"WARNING: IOError reading Java stdout (process likely died): {e}")
            # Do NOT explicitly close p_java.stdout here; let communicate() handle it.

            # --- Wait for Processes and Get Results ---

            # --- Handle Java Process ---
            try:
                 # communicate() reads remaining stdout/stderr and waits
                 # Important: Don't pass input to communicate() as stdin is piped from datainput
                 stdout_java_remaining, stderr_java_str_comm = p_java.communicate(timeout=java_wait_timeout)
                 if not stderr_java_str: stderr_java_str = stderr_java_str_comm # Append stderr if not already captured
                 return_code_java = p_java.returncode
            except subprocess.TimeoutExpired:
                 output_log_lines.append(f"ERROR: Timeout ({java_wait_timeout}s) waiting for Java process.")
                 p_java.kill()
                 stderr_after_timeout = "[Process timed out, stderr capture might be incomplete]"
                 stdout_after_timeout = "[Process timed out, stdout capture might be incomplete]"
                 try: stderr_after_timeout += "\n" + (p_java.stderr.read() if p_java.stderr else "")
                 except Exception: pass
                 try: stdout_after_timeout += "\n" + (p_java.stdout.read() if p_java.stdout else "")
                 except Exception: pass
                 if not stderr_java_str: stderr_java_str = stderr_after_timeout
                 stdout_java_remaining = stdout_after_timeout
                 return_code_java = -99 # Indicate timeout
            except ValueError as e_comm_java: # Catch ValueError from communicate's _readerthread
                 output_log_lines.append(f"WARNING: ValueError during java communicate (process likely ended abruptly): {e_comm_java}")
                 stderr_java_str = "[stderr unavailable due to ValueError during communicate]"
                 stdout_java_remaining = "[stdout unavailable due to ValueError during communicate]"
                 if p_java.poll() is not None: return_code_java = p_java.returncode
                 else: return_code_java = -97 # Indicate communication error state
            except Exception as e_comm_java_other: # Catch other potential errors during communicate
                 output_log_lines.append(f"ERROR during java communicate: {e_comm_java_other}")
                 return_code_java = -98 # Indicate general communication error

            # Add any remaining stdout caught by communicate or after timeout/error
            stdout_java_remaining = stdout_java_remaining.strip() if stdout_java_remaining else ""
            if stdout_java_remaining:
                 output_log_lines.append("--- Java stdout (remaining/after error) ---")
                 output_log_lines.extend(stdout_java_remaining.splitlines())
                 output_log_lines.append("--- end remaining stdout ---")

            # --- Handle DataInput Process ---
            # Ensure datainput finishes or is terminated
            if p_datainput.poll() is None: # Only communicate if still running
                try:
                    # Datainput shouldn't produce much stdout after pipe closed, focus on stderr
                    _ignored_stdout_data, stderr_data_str_comm = p_datainput.communicate(timeout=datainput_wait_timeout)
                    if not stderr_data_str: stderr_data_str = stderr_data_str_comm
                    return_code_datainput = p_datainput.returncode
                except subprocess.TimeoutExpired:
                    output_log_lines.append(f"ERROR: Timeout ({datainput_wait_timeout}s) waiting for datainput.")
                    p_datainput.kill(); stderr_data_str = "[Process timed out]"; return_code_datainput = -99
                except ValueError as e_comm_data:
                     output_log_lines.append(f"WARNING: ValueError during datainput communicate: {e_comm_data}")
                     stderr_data_str = "[stderr unavailable due to ValueError during communicate]"
                     if p_datainput.poll() is not None: return_code_datainput = p_datainput.returncode
                     else: return_code_datainput = -97
                except Exception as e_comm_data_other:
                    output_log_lines.append(f"ERROR communicating with datainput process: {e_comm_data_other}")
                    return_code_datainput = -98
            else: # datainput already finished
                if return_code_datainput == -1: return_code_datainput = p_datainput.returncode
                # Read any remaining stderr if it finished before communicate
                try:
                     if not stderr_data_str and p_datainput.stderr:
                          stderr_data_str = p_datainput.stderr.read()
                except Exception: pass


            # --- Process Finished ---
            execution_finished_without_error = (return_code_java >= 0)

            # --- Append Stderr to Logs ---
            stderr_data_str = stderr_data_str.strip() if stderr_data_str else ""
            stderr_java_str = stderr_java_str.strip() if stderr_java_str else ""
            if stderr_data_str:
                output_log_lines.append(f"--- datainput stderr (Exit Code: {return_code_datainput}) ---")
                output_log_lines.extend(stderr_data_str.splitlines()); output_log_lines.append(f"--- end datainput stderr ---")
            if stderr_java_str:
                output_log_lines.append(f"--- Java stderr (Exit Code: {return_code_java}) ---")
                output_log_lines.extend(stderr_java_str.splitlines()); output_log_lines.append(f"--- end Java stderr ---")
            if return_code_datainput not in [0, -1, None]: output_log_lines.append(f"WARNING: datainput.exe exited code {return_code_datainput}")
            if return_code_java > 0: output_log_lines.append(f"WARNING: Java process exited code {return_code_java}")

            # --- Perform Detailed Validation ---
            # Assuming validate_output function exists and is correct
            if execution_finished_without_error:
                validation_start_time = time.monotonic()
                # Ensure validate_output is defined and accessible in this scope
                validation_issues = validate_output(output_log_lines, input_lines) # Calls the updated validator
                validation_duration = time.monotonic() - validation_start_time
                if validation_duration > 10.0: output_log_lines.append(f"--- Validation took {validation_duration:.2f} seconds ---")
            elif return_code_java == -99: validation_issues = [f"Java process timed out ({java_wait_timeout}s). Validation skipped."]
            else: validation_issues = [f"Java process error/early exit (Code: {return_code_java}). Validation skipped."]

            # --- Determine Overall Success (Ignoring Warnings) ---
            java_ok = (return_code_java == 0)
            actual_errors_violations = []
            # Define prefixes/keywords indicating non-failure states or allowed messages
            allowed_prefixes_to_ignore = ("WARN:", "DEBUG:", "INFO:", "INPUT WARNING:")
            allowed_non_core_prefixes = ('DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL', 'Result:', '---', '[', '#', 'at ', '\tat ')
            # Define keywords indicating definite failure
            # <<< Added UPDATE VIOLATION, COLLISION VIOLATION >>>
            error_keywords = ("VIOLATION:", "ERROR:", "FATAL", "SEMANTIC VIOLATION:", "FINAL CHECK:", "INTERNAL ERR:", "UPDATE VIOLATION:", "COLLISION VIOLATION:")

            if validation_issues: # Check list exists
                for issue in validation_issues:
                    if issue and isinstance(issue, str): # Check item validity
                        issue_stripped = issue.strip()
                        is_ignorable_prefix = issue_stripped.startswith(allowed_prefixes_to_ignore)
                        is_error_type = any(keyword in issue_stripped for keyword in error_keywords)

                        # Check specifically for Malformed/Unknown related to CORE actions
                        is_malformed_core = False
                        if "Malformed Line / Missing Timestamp" in issue_stripped:
                             line_content_part = issue_stripped.split(":", 1)[-1].strip()
                             if not any(line_content_part.startswith(kw) for kw in allowed_non_core_prefixes):
                                  is_malformed_core = True
                        is_unknown_core = False
                        if "Unknown or Malformed Action Format" in issue_stripped:
                             line_content_part = issue_stripped.split(":", 1)[-1].strip()
                             if not any(line_content_part.startswith(kw) for kw in allowed_non_core_prefixes):
                                  is_unknown_core = True

                        # Count as error if it contains error keywords AND is NOT explicitly just a warning/info,
                        # OR if it's a malformed/unknown core action.
                        if (is_error_type and not is_ignorable_prefix) or is_malformed_core or is_unknown_core:
                            actual_errors_violations.append(issue)

            validation_ok = not actual_errors_violations # OK only if NO actual errors/violations
            success = java_ok and validation_ok

            # --- Construct Final Log Message ---
            if success: output_log_lines.append(f"-- Result: PASSED (Java Code: {return_code_java}, Validation OK) --")
            else:
                fail_reasons = []
                if return_code_java == -99: fail_reasons.append(f"Java timeout ({java_wait_timeout}s)")
                elif return_code_java < 0: fail_reasons.append(f"Java error ({return_code_java})")
                elif not java_ok: fail_reasons.append(f"Java failed (Code: {return_code_java})")
                if not validation_ok: fail_reasons.append("Validation failed")
                output_log_lines.append(f"-- Result: FAILED ({', '.join(fail_reasons)}) --")

    # --- Exception Handlers ---
    except InterruptedError: success = False; validation_issues = ["Batch run stopped by user."]; output_log_lines.append("--- Task Interrupted ---"); return_code_java = -96
    except FileNotFoundError as e: success = False; validation_issues = [f"File not found error: {e}"]; output_log_lines.append(f"FATAL ERROR: File Not Found - {e}"); return_code_java = -95
    except (RuntimeError, IOError, OSError) as e: success = False; validation_issues = [f"Process/IO error: {e}"]; output_log_lines.append(f"FATAL ERROR: Process/IO Error - {e}"); output_log_lines.append(traceback.format_exc()); return_code_java = -94
    except Exception as e: success = False; validation_issues = [f"Unexpected Python error: {type(e).__name__}"]; output_log_lines.append(f"FATAL PYTHON ERROR: {e}"); output_log_lines.append(traceback.format_exc()); return_code_java = -93
    finally: # Final Cleanup
        # Ensure processes are terminated
        for p_name, p in [("Java", p_java), ("DataInput", p_datainput)]:
            if p and p.poll() is None:
                try: p.kill(); p.wait(timeout=0.5)
                except Exception: pass # Ignore errors during kill
            # Close any streams that might still be open on our side
            for stream_name in ['stdin', 'stdout', 'stderr']:
                stream = getattr(p, stream_name, None)
                if stream and not stream.closed:
                    try: stream.close()
                    except Exception: pass
        end_time = time.monotonic()
        output_log_lines.append(f"-- Task {test_id} duration: {end_time - start_time:.2f} seconds --")

    # Final Status Adjustment
    if not execution_finished_without_error and validation_issues == ["Task did not complete successfully."]:
        if return_code_java < 0: validation_issues = [f"Task failed: Java process issue (Code: {return_code_java})."]
        else: validation_issues = ["Task failed during setup/execution."]

    # Return original validation_issues list (incl warnings) for logging
    return (test_id, success, "\n".join(output_log_lines), validation_issues, input_lines)


# --- Main Application Class ---
class ElevatorTesterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Elevator Tester - HW7 Enhanced") # <<< Title Update >>>
        self.geometry("1200x850") # Wider and taller

        # Style configuration
        self.style = ttk.Style(self)
        self.style.theme_use('clam') # Or 'vista', 'xpnative', 'default'

        # --- File Paths ---
        self.jar_folder_path = tk.StringVar()
        self.datainput_path = tk.StringVar()

        # --- Test Data Storage (for manual tests) ---
        self.test_data_sets = {} # { "test_name": [input_line1, ...] }
        self.current_test_name = None

        # --- Batch Processing State ---
        self.batch_thread = None
        self.stop_batch_event = threading.Event()
        self.message_queue = queue.Queue() # For thread-safe GUI updates
        self.is_batch_running = False
        self.current_jar_being_tested = tk.StringVar(value="N/A") # Track current JAR
        self.batch_test_counters = {'processed': 0, 'total': 0, 'passed': 0, 'failed': 0} # Track progress within current JAR

        # --- Generation Parameters <<< MODIFIED >>>---
        self.gen_passengers = tk.IntVar(value=20) # Default more passengers
        self.gen_schedules = tk.IntVar(value=5)  # Default more schedules
        self.gen_updates = tk.IntVar(value=1) # <<< NEW >>> Default to 1 UPDATE command
        self.gen_max_time = tk.DoubleVar(value=T_MAX_HUTEST) # Default to peer test limit
        self.batch_num_tests = tk.IntVar(value=50) # Default more tests per JAR
        self.batch_concurrency = tk.IntVar(value=DEFAULT_CONCURRENCY)

        # --- Failures Directory ---
        self.base_failures_dir = os.path.abspath(FAILURES_DIR)
        try:
            os.makedirs(self.base_failures_dir, exist_ok=True)
            # Also create manual subdir preemptively
            os.makedirs(os.path.join(self.base_failures_dir, MANUAL_FAILURES_DIR), exist_ok=True)
        except OSError as e:
            # Schedule error display after main loop starts
            self.after(100, lambda: messagebox.showerror("Directory Error", f"Could not create base failures directory '{self.base_failures_dir}':\n{e}", parent=self))

        # --- Build GUI ---
        self._create_widgets()
        self.protocol("WM_DELETE_WINDOW", self._quit) # Handle window close button

        # Load datainput path from a simple config file if it exists
        self._load_config()


    def _load_config(self):
        """Loads simple config like datainput path."""
        config_file = "elevator_tester_config.json"
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if 'datainput_path' in config and os.path.exists(config['datainput_path']):
                        self.datainput_path.set(config['datainput_path'])
                        self._log_output(f"Loaded datainput path from {config_file}")
                    if 'jar_folder_path' in config and os.path.isdir(config['jar_folder_path']):
                        self.jar_folder_path.set(config['jar_folder_path'])
                        self._log_output(f"Loaded JAR folder path from {config_file}")

        except Exception as e:
            self._log_output(f"Warning: Could not load config from {config_file}: {e}")

    def _save_config(self):
        """Saves simple config like datainput path."""
        config_file = "elevator_tester_config.json"
        config = {}
        di_path = self.datainput_path.get()
        jf_path = self.jar_folder_path.get()
        if di_path: config['datainput_path'] = di_path
        if jf_path: config['jar_folder_path'] = jf_path

        if config: # Only save if there's something to save
            try:
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
            except Exception as e:
                self._log_output(f"Warning: Could not save config to {config_file}: {e}")


    def _create_widgets(self):
        # --- Top Frame: File Selections ---
        top_frame = ttk.Frame(self, padding="10")
        top_frame.pack(fill=tk.X, side=tk.TOP, pady=(0, 5))
        top_frame.columnconfigure(1, weight=1) # Make entry expand

        # JAR Folder Selection
        ttk.Label(top_frame, text="Elevator JAR Folder:", anchor="w").grid(row=0, column=0, padx=(0, 5), pady=2, sticky="w")
        ttk.Entry(top_frame, textvariable=self.jar_folder_path, width=80).grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        ttk.Button(top_frame, text="Browse...", command=self._select_jar_folder, width=10).grid(row=0, column=2, padx=(5, 0), pady=2)

        # datainput EXE Selection
        ttk.Label(top_frame, text="datainput EXE:", anchor="w").grid(row=1, column=0, padx=(0, 5), pady=2, sticky="w")
        ttk.Entry(top_frame, textvariable=self.datainput_path, width=80).grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        ttk.Button(top_frame, text="Browse...", command=self._select_datainput, width=10).grid(row=1, column=2, padx=(5, 0), pady=2)

        # --- Parameters Frame (Combined Generation & Batch) --- <<< MODIFIED >>>
        params_frame = ttk.Frame(self, padding="5")
        params_frame.pack(fill=tk.X, padx=5, pady=5)

        gen_group = ttk.LabelFrame(params_frame, text="Data Generation Parameters", padding="5")
        gen_group.pack(side=tk.LEFT, padx=(0, 10), fill=tk.Y)
        ttk.Label(gen_group, text="Passengers:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(gen_group, textvariable=self.gen_passengers, width=6).grid(row=0, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(gen_group, text="Schedules:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(gen_group, textvariable=self.gen_schedules, width=6).grid(row=1, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(gen_group, text="Updates:").grid(row=2, column=0, padx=5, pady=2, sticky="w") # <<< NEW >>>
        ttk.Entry(gen_group, textvariable=self.gen_updates, width=6).grid(row=2, column=1, padx=5, pady=2, sticky="w") # <<< NEW >>>
        ttk.Label(gen_group, text="Max Time (s):").grid(row=3, column=0, padx=5, pady=2, sticky="w") # <<< Row updated >>>
        ttk.Entry(gen_group, textvariable=self.gen_max_time, width=6).grid(row=3, column=1, padx=5, pady=2, sticky="w") # <<< Row updated >>>

        batch_group = ttk.LabelFrame(params_frame, text="Batch Testing Parameters", padding="5")
        batch_group.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(batch_group, text="Tests per JAR:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(batch_group, textvariable=self.batch_num_tests, width=6).grid(row=0, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(batch_group, text="Concurrency:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(batch_group, textvariable=self.batch_concurrency, width=6).grid(row=1, column=1, padx=5, pady=2, sticky="w")

        # --- Batch Controls Frame ---
        batch_controls_frame = ttk.Frame(self, padding="5")
        batch_controls_frame.pack(fill=tk.X, padx=5, pady=0)

        self.start_batch_button = ttk.Button(batch_controls_frame, text="Start Batch Test", command=self._start_batch_test, width=18)
        self.start_batch_button.pack(side=tk.LEFT, padx=(0, 10))
        self.stop_batch_button = ttk.Button(batch_controls_frame, text="Stop Batch Test", command=self._stop_batch_test, state=tk.DISABLED, width=18)
        self.stop_batch_button.pack(side=tk.LEFT, padx=0)

        # Progress Labels within Batch Controls
        progress_frame = ttk.Frame(batch_controls_frame, padding=(10, 0))
        progress_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(progress_frame, text="Current JAR:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(progress_frame, textvariable=self.current_jar_being_tested, relief=tk.SUNKEN, width=30, anchor='w').pack(side=tk.LEFT, padx=(0, 10))
        self.batch_progress_label = ttk.Label(progress_frame, text="Status: Idle", relief=tk.SUNKEN, anchor='w')
        self.batch_progress_label.pack(side=tk.LEFT, fill=tk.X, expand=True)


        # --- Main Area: Paned Window ---
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Left Pane: Manual Test Management ---
        left_pane = ttk.Frame(main_pane, padding="5", width=350) # Give initial width
        left_pane.pack_propagate(False) # Prevent shrinking
        main_pane.add(left_pane, weight=1) # Less weight initially

        ttk.Label(left_pane, text="Manual Test Cases:", font="-weight bold").pack(anchor=tk.W)
        listbox_frame = ttk.Frame(left_pane)
        listbox_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 5))
        self.test_listbox = tk.Listbox(listbox_frame, height=8) # Slightly taller
        listbox_scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.test_listbox.yview)
        self.test_listbox.config(yscrollcommand=listbox_scrollbar.set)
        listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.test_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.test_listbox.bind('<<ListboxSelect>>', self._on_test_select)

        test_button_frame = ttk.Frame(left_pane)
        test_button_frame.pack(fill=tk.X)
        ttk.Button(test_button_frame, text="Load...", command=self._load_tests, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(test_button_frame, text="Save Sel...", command=self._save_selected_test, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(test_button_frame, text="Remove Sel", command=self._remove_selected_test, width=10).pack(side=tk.LEFT, padx=2)

        ttk.Label(left_pane, text="Selected Input Preview:", font="-weight bold").pack(anchor=tk.W, pady=(10, 0))
        self.input_text = scrolledtext.ScrolledText(left_pane, height=12, wrap=tk.WORD, state=tk.DISABLED, font=("Consolas", 9))
        self.input_text.pack(fill=tk.BOTH, expand=True, pady=(5, 5))

        self.run_manual_button = ttk.Button(left_pane, text="Run Selected Manual Test", command=self._run_manual_selected_test, state=tk.DISABLED)
        self.run_manual_button.pack(pady=5, anchor=tk.CENTER)

        # --- Right Pane: Output Log ---
        right_pane = ttk.Frame(main_pane, padding="5", width=650)
        right_pane.pack_propagate(False)
        main_pane.add(right_pane, weight=3) # More weight

        # Add Clear Log button
        clear_log_button = ttk.Button(right_pane, text="Clear Log", command=self._clear_output_log)
        clear_log_button.pack(anchor=tk.NE, pady=(0,3))

        self.output_log_text = scrolledtext.ScrolledText(right_pane, wrap=tk.WORD, state=tk.DISABLED, font=("Consolas", 9)) # Monospaced font
        self.output_log_text.pack(fill=tk.BOTH, expand=True)

    # --- GUI Logic Methods ---

    def _select_jar_folder(self):
        """Selects a folder containing JAR files."""
        path = filedialog.askdirectory(
            title="Select Folder Containing Elevator JAR(s)",
            initialdir=self.jar_folder_path.get() or '.' # Start from previous or current
        )
        if path:
            self.jar_folder_path.set(path)
            self._log_output(f"JAR folder set to: {path}")
            self._save_config() # Save selection

    def _select_datainput(self):
        """Selects the datainput executable."""
        path = filedialog.askopenfilename(
            title="Select datainput Executable",
            filetypes=[("Executable files", "*.exe"), ("All files", ".*")],
            initialdir=os.path.dirname(self.datainput_path.get()) if self.datainput_path.get() else '.'
        )
        if path:
            self.datainput_path.set(path)
            self._log_output(f"datainput EXE set to: {path}")
            self._save_config() # Save selection

    # --- Manual Test Methods ---
    def _update_test_listbox(self):
        """Updates the listbox with current manual test names."""
        if not hasattr(self, 'test_listbox') or not self.test_listbox.winfo_exists(): return
        try:
            # Store selection before clearing
            selected_indices = self.test_listbox.curselection()
            selected_value = self.test_listbox.get(selected_indices[0]) if selected_indices else None

            self.test_listbox.delete(0, tk.END)
            sorted_names = sorted(self.test_data_sets.keys())
            for name in sorted_names:
                self.test_listbox.insert(tk.END, name)

            # Restore selection if item still exists
            if selected_value in sorted_names:
                try:
                    idx = sorted_names.index(selected_value)
                    self.test_listbox.selection_set(idx)
                    self.test_listbox.activate(idx) # Ensure visible
                    self.test_listbox.see(idx)
                except ValueError: pass # Should not happen
            else:
                 # If previous selection removed, clear preview and disable run button
                 self.current_test_name = None
                 self._display_input_data([])
                 if hasattr(self, 'run_manual_button') and self.run_manual_button.winfo_exists():
                     self.run_manual_button.config(state=tk.DISABLED)

        except tk.TclError: pass # Ignore if widget destroyed


    def _on_test_select(self, event=None):
        """Handles selection changes in the manual test listbox."""
        if not hasattr(self, 'test_listbox') or not self.test_listbox.winfo_exists(): return
        try:
            selected_indices = self.test_listbox.curselection()
            if not selected_indices:
                self.current_test_name = None
                self._display_input_data([])
                if hasattr(self, 'run_manual_button') and self.run_manual_button.winfo_exists():
                    self.run_manual_button.config(state=tk.DISABLED)
                return

            selected_name = self.test_listbox.get(selected_indices[0])
            if selected_name != self.current_test_name: # Update only if changed
                self.current_test_name = selected_name
                if self.current_test_name in self.test_data_sets:
                    self._display_input_data(self.test_data_sets[self.current_test_name])
                else:
                    self._display_input_data([]) # Should not happen

            # Enable Run button only if batch is not running AND a valid test is selected
            if hasattr(self, 'run_manual_button') and self.run_manual_button.winfo_exists():
                if not self.is_batch_running and self.current_test_name:
                    self.run_manual_button.config(state=tk.NORMAL)
                else:
                    self.run_manual_button.config(state=tk.DISABLED)
        except tk.TclError: pass # Ignore if widget destroyed


    def _display_input_data(self, data_lines):
        """Displays the input data for the selected manual test."""
        if not hasattr(self, 'input_text') or not self.input_text.winfo_exists(): return
        try:
            self.input_text.config(state=tk.NORMAL)
            self.input_text.delete('1.0', tk.END)
            if data_lines and self.current_test_name:
                display_str = f"# Test Case: {self.current_test_name}\n"
                display_str += f"# Lines: {len(data_lines)}\n"
                display_str += "-" * 20 + "\n"
                display_str += "\n".join(data_lines)
                self.input_text.insert('1.0', display_str)
            self.input_text.config(state=tk.DISABLED)
            self.input_text.yview_moveto(0.0) # Scroll to top
        except tk.TclError: pass # Ignore if widget destroyed


    def _load_tests(self):
        """Loads manual test cases from a JSON file."""
        if self.is_batch_running:
             messagebox.showwarning("Busy", "Cannot load tests while batch is running.", parent=self)
             return
        filepath = filedialog.askopenfilename(
            title="Load Manual Test Data (JSON: {name: [lines,...]})",
            filetypes=[("JSON files", "*.json"), ("All files", ".*")]
        )
        if not filepath: return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            if not isinstance(loaded_data, dict):
                raise ValueError("JSON root must be a dictionary (test_name -> list of strings)")

            count_loaded = 0; count_skipped = 0; count_overwritten = 0
            loaded_names = []
            for name, data in loaded_data.items():
                if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
                    self._log_output(f"WARNING: Skipping test '{name}' in {os.path.basename(filepath)} - invalid format (must be list of strings).")
                    count_skipped += 1; continue

                overwrite = False
                if name in self.test_data_sets:
                    if messagebox.askyesno("Confirm Overwrite", f"Manual test set '{name}' already exists. Overwrite?", parent=self):
                         overwrite = True; count_overwritten += 1
                    else:
                         count_skipped += 1; continue # Skip if user selects No

                self.test_data_sets[name] = data
                if not overwrite: count_loaded += 1
                loaded_names.append(name) # Track names added/overwritten in this load operation

            self._update_test_listbox() # Update listbox after processing file
            if loaded_names: # If any tests were loaded/overwritten
                # Select the first test loaded from this file
                try:
                    # Find index in potentially sorted listbox view
                    listbox_items = self.test_listbox.get(0, tk.END)
                    idx = listbox_items.index(loaded_names[0])
                    self.test_listbox.selection_clear(0, tk.END)
                    self.test_listbox.selection_set(idx)
                    self._on_test_select() # Trigger display update & button state
                except ValueError: pass # Item not found, should not happen

            info_msg = f"Loaded {count_loaded} new test(s)"
            if count_overwritten: info_msg += f", Overwrote {count_overwritten}"
            if count_skipped: info_msg += f", Skipped {count_skipped}"
            info_msg += f" from {os.path.basename(filepath)}"
            messagebox.showinfo("Load Complete", info_msg, parent=self)
            self._log_output(info_msg)

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load test data from {os.path.basename(filepath)}:\n{e}", parent=self)
            self._log_output(f"ERROR loading test data: {e}")


    def _save_selected_test(self):
        """Saves the currently selected manual test case to a JSON file."""
        if not self.current_test_name or self.current_test_name not in self.test_data_sets:
            messagebox.showwarning("Warning", "No manual test case selected to save.", parent=self)
            return

        # Sanitize name for filename suggestion
        safe_name = re.sub(r'[^\w\-]+', '_', self.current_test_name)
        filepath = filedialog.asksaveasfilename(
            title="Save Selected Manual Test Data (JSON format)",
            defaultextension=".json",
            initialfile=f"{safe_name}.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath: return

        data_to_save = {self.current_test_name: self.test_data_sets[self.current_test_name]}
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2)
            msg = f"Saved test '{self.current_test_name}' to {os.path.basename(filepath)}"
            messagebox.showinfo("Success", msg, parent=self)
            self._log_output(msg)
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save test data:\n{e}", parent=self)
            self._log_output(f"ERROR saving test data: {e}")


    def _remove_selected_test(self):
        """Removes the currently selected manual test case."""
        if self.is_batch_running:
             messagebox.showwarning("Busy", "Cannot modify tests while batch is running.", parent=self)
             return
        if not self.current_test_name or self.current_test_name not in self.test_data_sets:
            messagebox.showwarning("Warning", "No manual test case selected to remove.", parent=self)
            return

        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to remove the manual test set '{self.current_test_name}'?", parent=self):
            removed_name = self.current_test_name
            if removed_name in self.test_data_sets:
                del self.test_data_sets[removed_name]
            self.current_test_name = None # Clear current selection state
            self._update_test_listbox() # Update list (will clear preview/disable button)
            self._log_output(f"Removed manual test: {removed_name}")


    # --- Batch Testing Methods ---
    def _validate_batch_inputs(self): # <<< MODIFIED >>>
        """Validates all inputs required for batch testing. Returns True if valid, False otherwise."""
        jar_folder = self.jar_folder_path.get()
        datainput_exe = self.datainput_path.get()
        num_tests = self.batch_num_tests.get()
        concurrency = self.batch_concurrency.get()
        num_p = self.gen_passengers.get()
        num_s = self.gen_schedules.get()
        num_u = self.gen_updates.get() # <<< NEW >>> Get updates count
        max_t = self.gen_max_time.get()

        if not jar_folder or not os.path.isdir(jar_folder):
            messagebox.showerror("Input Error", "Please select a valid Elevator JAR folder.", parent=self); return False
        if not datainput_exe or not os.path.exists(datainput_exe) or not os.path.isfile(datainput_exe):
            messagebox.showerror("Input Error", "Please select a valid datainput executable file.", parent=self); return False
        try: # Validate numeric inputs
            if not (1 <= num_tests <= 100000): raise ValueError("Tests per JAR must be between 1 and 100,000.")
            if not (1 <= concurrency <= 128): raise ValueError("Concurrency must be between 1 and 128.")
            if not (0 <= num_p <= 10000): raise ValueError("Passengers must be between 0 and 10,000.")
            if not (0 <= num_s <= 1000): raise ValueError("Schedules must be between 0 and 1,000.")
            if not (0 <= num_u <= 500): raise ValueError("Updates must be between 0 and 500.") # <<< NEW >>> Check updates
            if not (10.0 <= max_t <= T_MAX_HUTEST + 120.0): raise ValueError(f"Max Time must be between 10.0 and {T_MAX_HUTEST + 120.0} seconds.") # Increased min time slightly
            if num_p == 0 and num_s == 0 and num_u == 0: raise ValueError("Number of passengers, schedules, or updates must be greater than 0.") # <<< MODIFIED >>>
        except (tk.TclError, ValueError) as e: # Catch invalid int/float entry or range error
             messagebox.showerror("Input Error", f"Invalid parameter value:\n{e}", parent=self); return False

        # Check for JAR files
        try:
            jar_files = [f for f in os.listdir(jar_folder) if f.lower().endswith('.jar')]
            if not jar_files:
                 messagebox.showerror("Input Error", f"No .jar files found in the selected folder:\n{jar_folder}", parent=self); return False
            self.found_jar_paths = [os.path.join(jar_folder, f) for f in jar_files] # Store found paths
        except OSError as e:
             messagebox.showerror("Input Error", f"Error accessing JAR folder:\n{e}", parent=self); return False

        return True # All checks passed

    def _start_batch_test(self): # <<< MODIFIED >>>
        """Starts the batch testing process."""
        if self.is_batch_running:
            messagebox.showwarning("Busy", "A batch test is already running.", parent=self)
            return
        if not self._validate_batch_inputs():
            return # Validation failed, message already shown

        # Get validated parameters
        jar_paths = self.found_jar_paths # Use paths found during validation
        datainput_exe = self.datainput_path.get()
        num_tests = self.batch_num_tests.get(); concurrency = self.batch_concurrency.get()
        num_p = self.gen_passengers.get(); num_s = self.gen_schedules.get()
        num_u = self.gen_updates.get() # <<< NEW >>> Get updates count
        max_t = self.gen_max_time.get()

        # Log start info
        self._clear_output_log()
        self._log_output("--- Starting New Batch Test (HW7 Mode) ---") # <<< Title Update >>>
        self._log_output(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._log_output(f"JAR Folder: {self.jar_folder_path.get()} ({len(jar_paths)} JARs)")
        self._log_output(f"DataInput EXE: {datainput_exe}")
        self._log_output(f"Tests per JAR: {num_tests}, Concurrency: {concurrency}")
        self._log_output(f"Gen Params: P={num_p}, S={num_s}, U={num_u}, MaxT={max_t:.1f}s") # <<< MODIFIED >>> Log updates count
        self._log_output(f"Failure logs will be saved under: {self.base_failures_dir}")
        self._log_output("-" * 30)

        # Update GUI state
        self.is_batch_running = True
        self.start_batch_button.config(state=tk.DISABLED)
        self.stop_batch_button.config(state=tk.NORMAL)
        # Disable manual run button if it exists
        if hasattr(self, 'run_manual_button') and self.run_manual_button.winfo_exists():
             self.run_manual_button.config(state=tk.DISABLED)
        # Consider disabling parameter entries
        # ...
        self.batch_progress_label.config(text="Status: Initializing...")
        self.current_jar_being_tested.set("Preparing...")
        self.stop_batch_event.clear() # Ensure stop flag is reset

        # Start the manager thread <<< MODIFIED: Pass num_u >>>
        self.batch_thread = threading.Thread(
            target=self._run_batch_manager,
            args=(jar_paths, num_tests, concurrency, datainput_exe, num_p, num_s, num_u, max_t), # Pass num_u
            daemon=True # Allows app to exit even if thread hangs (though we try to join)
        )
        self.batch_thread.start()
        self.after(100, self._process_batch_queue) # Start polling queue for updates


    def _stop_batch_test(self):
        """Signals the batch testing thread to stop gracefully."""
        if not self.is_batch_running:
            # self._log_output("DEBUG: Stop requested but batch not running.") # Debug log
            return

        if self.batch_thread and self.batch_thread.is_alive():
            self._log_output("--- Stop Requested --- Signalling worker threads...")
            # Update GUI immediately to show stopping state
            # Progress label update will be handled by manager thread noticing the stop event
            self.batch_progress_label.config(text="Status: Stopping... (Waiting for active tasks)")
            # Signal the event
            self.stop_batch_event.set()
            # Disable stop button to prevent multiple clicks
            self.stop_batch_button.config(state=tk.DISABLED)
            # We don't re-enable start button here; the manager thread does that on clean exit.
        else:
            # Thread already finished or wasn't running, reset state just in case
            self._log_output("Stop requested, but batch thread is not active. Resetting GUI.")
            self._reset_gui_after_batch()


    def _reset_gui_after_batch(self):
        """Resets GUI controls to idle state after batch finishes or stops."""
        self.is_batch_running = False
        if hasattr(self, 'start_batch_button') and self.start_batch_button.winfo_exists():
            self.start_batch_button.config(state=tk.NORMAL)
        if hasattr(self, 'stop_batch_button') and self.stop_batch_button.winfo_exists():
            self.stop_batch_button.config(state=tk.DISABLED)
        # Re-enable manual run button based on current selection state
        self._on_test_select() # This handles the button state based on selection and is_batch_running
        # Re-enable any parameter widgets if they were disabled
        # ... (add code here if parameter widgets were disabled) ...
        self.batch_thread = None
        # Don't clear stop event here, let start clear it.

    def _run_batch_manager(self, jar_paths, num_tests_per_jar, concurrency, datainput_exe_path, num_p, num_s, num_u, max_t): # <<< MODIFIED >>>
        """
        Manages the batch execution: generates data, runs tests per JAR, handles stop signals. (Worker Thread)
        Sends updates to the GUI via the message_queue.
        """
        overall_summary = {} # { jar_name: {'passed': p, 'failed': f, 'skipped': s} }
        total_jars = len(jar_paths)
        start_batch_time = time.monotonic()

        try:
            # 1. Generate all test data upfront <<< MODIFIED: Pass num_u >>>
            self.message_queue.put(("progress", f"Status: Generating {num_tests_per_jar} test cases..."))
            self.message_queue.put(("current_jar", "Generating Data..."))
            all_test_inputs = []
            gen_start_time = time.monotonic()
            for i in range(num_tests_per_jar):
                if self.stop_batch_event.is_set():
                    raise InterruptedError("Stop requested during test data generation")
                input_lines = generate_test_data(num_p, num_s, num_u, max_t) # Pass num_u
                all_test_inputs.append(input_lines)
                if (i + 1) % (max(1, num_tests_per_jar // 20)) == 0: # Update progress less frequently
                    progress_pct = (i + 1) / num_tests_per_jar * 100
                    self.message_queue.put(("progress", f"Status: Generating test cases... {progress_pct:.0f}%"))
            gen_duration = time.monotonic() - gen_start_time
            self.message_queue.put(("log", f"Data generation complete ({num_tests_per_jar} cases) in {gen_duration:.2f}s."))

            # 2. Iterate through each JAR file
            for jar_index, jar_path in enumerate(jar_paths):
                if self.stop_batch_event.is_set():
                    self.message_queue.put(("log", f"Stop requested. Skipping remaining {total_jars - jar_index} JARs."))
                    # Mark remaining as skipped
                    for skipped_jar_path in jar_paths[jar_index:]:
                         skipped_jar_name = os.path.basename(skipped_jar_path)
                         overall_summary[skipped_jar_name] = {'passed': 0, 'failed': 0, 'skipped': num_tests_per_jar}
                    break # Exit JAR loop

                jar_name = os.path.basename(jar_path)
                self.message_queue.put(("current_jar", f"{jar_name} ({jar_index+1}/{total_jars})"))
                self.message_queue.put(("log", f"\n--- [{jar_index+1}/{total_jars}] Testing JAR: {jar_name} ---"))
                self.message_queue.put(("reset_counters", (0, num_tests_per_jar))) # Reset counters for this JAR

                # Create JAR-specific failure directory
                safe_jar_dir_name = re.sub(r'[^\w\-]+', '_', jar_name.replace('.jar', ''))
                jar_failure_dir = os.path.join(self.base_failures_dir, safe_jar_dir_name)
                try:
                    os.makedirs(jar_failure_dir, exist_ok=True)
                except OSError as e:
                    self.message_queue.put(("log", f"ERROR: Could not create failure directory '{jar_failure_dir}': {e}. Failures for this JAR may be lost or saved in base dir."))
                    jar_failure_dir = self.base_failures_dir # Fallback (less ideal)

                # Initialize results for this JAR
                jar_passed = 0; jar_failed = 0; jar_processed = 0
                all_futures = {} # { future: test_id } for tracking

                # 3. Run tests for the current JAR using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix=f"{safe_jar_dir_name}_Worker") as executor:
                    self.message_queue.put(("progress", f"Status [JAR {jar_index+1}]: Submitting {num_tests_per_jar} tests..."))

                    # Submit all tasks for this JAR
                    for i in range(num_tests_per_jar):
                        if self.stop_batch_event.is_set():
                            self.message_queue.put(("progress", f"Status [JAR {jar_index+1}]: Stop requested during submission. Only {i} tests submitted."))
                            break # Stop submitting new tasks

                        test_id = f"{safe_jar_dir_name}_T{i+1:04d}" # Unique test ID
                        future = executor.submit(
                            _run_one_batch_test_task, # Calls the task function (which uses updated validator)
                            test_id,
                            all_test_inputs[i], # Use pre-generated data
                            jar_path,
                            datainput_exe_path,
                            self.stop_batch_event # Pass shared stop event
                        )
                        all_futures[future] = test_id

                    # If stopped during submission, cancel pending futures
                    if self.stop_batch_event.is_set():
                        cancelled_count = 0
                        for future in list(all_futures.keys()): # Iterate copy as we modify
                             if future.cancel():
                                  cancelled_count += 1
                                  test_id = all_futures.pop(future) # Remove cancelled future
                        if cancelled_count > 0:
                             self.message_queue.put(("log", f"Cancelled {cancelled_count} pending tasks for {jar_name} due to stop request."))

                    # 4. Process results as they complete for this JAR
                    self.message_queue.put(("progress", f"Status [JAR {jar_index+1}]: Running {len(all_futures)} tests..."))
                    for future in as_completed(all_futures):
                        test_id = all_futures[future] # Get test_id associated with this future
                        jar_processed += 1
                        try:
                            # Get result (may raise exception if task threw one)
                            _tid_res, success, output_log, validation_issues, input_lines = future.result()
                            if _tid_res != test_id: self.message_queue.put(("log", f"WARNING: Result ID mismatch! Expected {test_id}, got {_tid_res}"))

                            # Log result and update counters
                            if success:
                                jar_passed += 1
                                log_msg = f"  Result [{jar_processed}/{len(all_futures)}]: {test_id} -> PASSED"
                                self.message_queue.put(("update_counters", (jar_processed, jar_passed, jar_failed)))
                                self.message_queue.put(("log", log_msg))
                            else:
                                jar_failed += 1
                                log_msg = f"  Result [{jar_processed}/{len(all_futures)}]: {test_id} -> FAILED"
                                if validation_issues: log_msg += f" ({validation_issues[0][:80]}{'...' if len(validation_issues[0])>80 else ''})"
                                self.message_queue.put(("update_counters", (jar_processed, jar_passed, jar_failed)))
                                self.message_queue.put(("log", log_msg))

                                # Save failure artifacts
                                try:
                                    safe_test_id = re.sub(r'[^\w\.\-]+', '_', test_id)
                                    failure_stdin_path = os.path.join(jar_failure_dir, f"stdin_{safe_test_id}.txt")
                                    failure_log_path = os.path.join(jar_failure_dir, f"output_{safe_test_id}.log")

                                    with open(failure_stdin_path, 'w', encoding=PROCESS_ENCODING, errors='replace') as f_in:
                                        f_in.write(f"# Test ID: {test_id}\n# JAR: {jar_name}\n# Status: FAILED\n---\n")
                                        f_in.write("\n".join(input_lines))
                                    with open(failure_log_path, 'w', encoding=PROCESS_ENCODING, errors='replace') as f_log:
                                        f_log.write(f"# Test ID: {test_id}\n# JAR File: {jar_name}\n# Status: FAILED\n")
                                        f_log.write(f"# Validation Issues:\n")
                                        for issue in validation_issues: f_log.write(f"#   - {issue}\n")
                                        f_log.write("\n" + "="*10 + " Input (" + str(len(input_lines)) + " lines) " + "="*10 + "\n")
                                        f_log.write("\n".join(input_lines))
                                        f_log.write("\n\n" + "="*10 + " Full Output Log " + "="*10 + "\n")
                                        f_log.write(output_log)

                                except Exception as e_save:
                                     err_save_msg = f"    ERROR saving failure details for {test_id}: {e_save}"
                                     self.message_queue.put(("log", err_save_msg))

                        except Exception as exc: # Handle error *getting* the result
                            jar_failed += 1
                            err_msg = f"  Result [{jar_processed}/{len(all_futures)}]: {test_id} -> ERROR (Task Exception: {exc})"
                            self.message_queue.put(("update_counters", (jar_processed, jar_passed, jar_failed)))
                            self.message_queue.put(("log", err_msg))
                            tb_lines = traceback.format_exc().splitlines()
                            self.message_queue.put(("log", f"    Task Traceback:\n      " + "\n      ".join(tb_lines)))
                            # Save minimal error log
                            try:
                                safe_test_id = re.sub(r'[^\w\.\-]+', '_', test_id)
                                error_log_path = os.path.join(jar_failure_dir, f"error_{safe_test_id}.log")
                                with open(error_log_path, 'w', encoding=PROCESS_ENCODING, errors='replace') as f_err:
                                      f_err.write(f"# Test ID: {test_id}\n# JAR: {jar_name}\n# Status: ERROR\n\n")
                                      f_err.write(f"Error retrieving task result:\n{exc}\n\n")
                                      f_err.write("Task Traceback:\n")
                                      f_err.write(traceback.format_exc())
                            except Exception as e_save_err:
                                self.message_queue.put(("log", f"    ERROR saving error details: {e_save_err}"))

                        # Update overall progress label
                        current_progress_text = f"Status [JAR {jar_index+1}]: Processed {jar_processed}/{num_tests_per_jar} (P:{jar_passed}, F:{jar_failed})"
                        if self.stop_batch_event.is_set(): current_progress_text += " [Stopping...]"
                        self.message_queue.put(("progress", current_progress_text))

                # End of tests for current JAR
                self.message_queue.put(("log", f"--- Finished JAR: {jar_name} - Passed: {jar_passed}, Failed: {jar_failed} ---"))
                overall_summary[jar_name] = {'passed': jar_passed, 'failed': jar_failed, 'skipped': num_tests_per_jar - jar_processed}

        except InterruptedError: # Catch stop during generation or between JARs
             self.message_queue.put(("progress", "Status: Batch stopped by user."))
             self.message_queue.put(("log", "--- Batch Run Interrupted ---"))
             # Summary will reflect completed JARs and potentially skipped ones
        except Exception as e_manager:
             # Error during generation or manager setup/loop
             self.message_queue.put(("log", f"--- FATAL BATCH MANAGER ERROR ---"))
             self.message_queue.put(("log", f"Error: {e_manager}"))
             self.message_queue.put(("log", traceback.format_exc()))
             self.message_queue.put(("progress", "Status: FATAL ERROR"))
        finally:
             # --- Batch Finished or Stopped ---
             total_duration = time.monotonic() - start_batch_time
             self.message_queue.put(("log", f"\n--- Batch Run Finished in {total_duration:.2f} seconds ---"))
             # Signal GUI that batch is finished, sending final summary
             self.message_queue.put(("finished", overall_summary))


    def _process_batch_queue(self):
        """Processes messages from the batch manager thread. Runs in GUI Thread via self.after()."""
        try:
            while True: # Process all available messages in the queue currently
                if not self.winfo_exists(): return # Stop if window closed

                msg_type, data = self.message_queue.get_nowait()

                if msg_type == "progress":
                    if hasattr(self, 'batch_progress_label') and self.batch_progress_label.winfo_exists(): self.batch_progress_label.config(text=data)
                elif msg_type == "log":
                    self._log_output(data) # _log_output handles widget check and scheduling
                elif msg_type == "current_jar":
                     if hasattr(self, 'current_jar_being_tested'): self.current_jar_being_tested.set(data)
                elif msg_type == "reset_counters":
                    self.batch_test_counters['processed'] = data[0]
                    self.batch_test_counters['total'] = data[1]
                    self.batch_test_counters['passed'] = 0
                    self.batch_test_counters['failed'] = 0
                elif msg_type == "update_counters":
                    self.batch_test_counters['processed'] = data[0]
                    self.batch_test_counters['passed'] = data[1]
                    self.batch_test_counters['failed'] = data[2]
                    # Update progress label with detailed counts
                    total = self.batch_test_counters['total']
                    # Safely get JAR name part
                    current_jar_full = self.current_jar_being_tested.get() if hasattr(self, 'current_jar_being_tested') else "N/A"
                    jar_name_match = re.match(r"([^\s\(]+)", current_jar_full)
                    jar_name_str = jar_name_match.group(1) if jar_name_match else current_jar_full

                    prog_text = f"Status [{jar_name_str}]: " \
                                f"Processed {data[0]}/{total} (P:{data[1]}, F:{data[2]})"
                    if self.stop_batch_event.is_set(): prog_text += " [Stopping...]"
                    if hasattr(self, 'batch_progress_label') and self.batch_progress_label.winfo_exists(): self.batch_progress_label.config(text=prog_text)

                elif msg_type == "finished":
                    overall_summary = data
                    self._log_output("\n" + "="*15 + " Overall Batch Summary " + "="*15)
                    total_passed_all = 0; total_failed_all = 0; total_skipped_all = 0
                    jars_processed_count = 0

                    for jar_name, results in overall_summary.items():
                        jars_processed_count += 1
                        p = results.get('passed', 0)
                        f = results.get('failed', 0)
                        s = results.get('skipped', 0)
                        total_passed_all += p; total_failed_all += f; total_skipped_all += s
                        summary_line = f"  {jar_name}: Passed={p}, Failed={f}"
                        if s > 0: summary_line += f", Skipped/Cancelled={s}"
                        self._log_output(summary_line)

                    self._log_output("-" * (30 + len(" Overall Batch Summary ")))
                    final_msg = f"Batch Complete: Processed {jars_processed_count} JAR(s). " \
                                f"Total Passed={total_passed_all}, Total Failed={total_failed_all}, Total Skipped={total_skipped_all}"
                    self._log_output(final_msg)

                    if hasattr(self, 'batch_progress_label') and self.batch_progress_label.winfo_exists():
                        self.batch_progress_label.config(text=f"Status: Finished ({total_passed_all}P, {total_failed_all}F, {total_skipped_all}S)")
                    if hasattr(self, 'current_jar_being_tested'):
                        self.current_jar_being_tested.set("Finished")

                    # Reset GUI state
                    self._reset_gui_after_batch()
                    return # Stop polling queue for this batch run

        except queue.Empty:
            # No more messages currently, normal condition
            pass
        except Exception as e:
            # Ensure error message uses string representation of StringVar if needed
            jar_context = self.current_jar_being_tested.get() if hasattr(self, 'current_jar_being_tested') else 'N/A'
            print(f"ERROR processing batch queue (JAR context: {jar_context}): {e}\n{traceback.format_exc()}", file=sys.stderr) # Debug
            self._log_output(f"--- ERROR processing GUI queue: {e} ---")
            # Attempt to reset state on error
            if self.is_batch_running:
                self._log_output("--- Batch ending due to queue processing error ---")
                if hasattr(self, 'batch_progress_label') and self.batch_progress_label.winfo_exists(): self.batch_progress_label.config(text="Status: Error processing results")
                if hasattr(self, 'current_jar_being_tested'): self.current_jar_being_tested.set("Error")
                self._reset_gui_after_batch() # Reset buttons etc.

        # Reschedule check ONLY if batch is still marked as running
        if self.is_batch_running:
            if self.batch_thread and self.batch_thread.is_alive():
                self.after(150, self._process_batch_queue) # Check again after 150ms
            else:
                # Thread died unexpectedly without sending "finished" or stop incomplete
                if self.is_batch_running: # Check flag again, might have been reset by error handler
                    self._log_output("--- WARNING: Batch thread ended unexpectedly or stop incomplete ---")
                    if hasattr(self, 'batch_progress_label') and self.batch_progress_label.winfo_exists(): self.batch_progress_label.config(text="Status: Ended Unexpectedly/Error")
                    if hasattr(self, 'current_jar_being_tested'): self.current_jar_being_tested.set("Error/Ended")
                    self._reset_gui_after_batch()


    # --- Manual Test Execution ---
    def _validate_manual_inputs(self):
        """Validates inputs needed for a manual test run."""
        jar_folder = self.jar_folder_path.get()
        datainput_exe = self.datainput_path.get()

        if not jar_folder or not os.path.isdir(jar_folder):
             messagebox.showerror("Input Error", "Please select a valid Elevator JAR folder first.", parent=self); return False
        if not datainput_exe or not os.path.exists(datainput_exe) or not os.path.isfile(datainput_exe):
             messagebox.showerror("Input Error", "Please select the datainput executable.", parent=self); return False
        if not self.current_test_name or self.current_test_name not in self.test_data_sets:
             messagebox.showerror("Input Error", "Please select a manual test case from the list.", parent=self); return False
        if not self.test_data_sets[self.current_test_name]:
             messagebox.showwarning("Warning", f"Manual test case '{self.current_test_name}' is empty. Cannot run.", parent=self); return False

        try:
             self.found_jar_paths_manual = [f for f in os.listdir(jar_folder) if f.lower().endswith('.jar')]
             if not self.found_jar_paths_manual:
                  messagebox.showerror("Input Error", f"No .jar files found in the selected folder:\n{jar_folder}", parent=self); return False
        except OSError as e:
             messagebox.showerror("Input Error", f"Error accessing JAR folder:\n{e}", parent=self); return False

        return True

    def _run_manual_selected_test(self):
        """Runs the selected manual test case against a user-chosen JAR."""
        if self.is_batch_running:
            messagebox.showwarning("Busy", "Batch test is running. Cannot start manual test.", parent=self)
            return
        if not self._validate_manual_inputs():
            return

        # Get validated data
        test_data_lines = self.test_data_sets[self.current_test_name]
        jar_folder = self.jar_folder_path.get()
        datainput_exe = self.datainput_path.get()
        available_jars = self.found_jar_paths_manual

        # Ask user which JAR to use
        manual_jar_path = None
        if len(available_jars) == 1:
            confirmed = messagebox.askyesno("Confirm JAR", f"Run manual test '{self.current_test_name}' using this JAR?\n\n{available_jars[0]}", parent=self)
            if confirmed:
                manual_jar_path = os.path.join(jar_folder, available_jars[0])
            else: return # User cancelled confirmation
        else:
            selected_jar_name = self._ask_which_jar(available_jars)
            if not selected_jar_name: return # User cancelled dialog
            manual_jar_path = os.path.join(jar_folder, selected_jar_name)

        if not manual_jar_path: # Should be handled above, but double-check
            messagebox.showerror("Internal Error", "Failed to determine JAR file for manual test.", parent=self)
            return

        # --- Proceed with manual test run ---
        self._clear_output_log()
        self._log_output(f"--- Starting Manual Test: {self.current_test_name} ---")
        self._log_output(f"Using JAR: {os.path.basename(manual_jar_path)}")
        self._log_output(f"Using datainput: {os.path.basename(datainput_exe)}")
        self._log_output(f"Input Lines: {len(test_data_lines)}")
        self._log_output("-" * 20)


        # Update GUI state for manual run
        if hasattr(self, 'run_manual_button') and self.run_manual_button.winfo_exists(): self.run_manual_button.config(state=tk.DISABLED)
        if hasattr(self, 'start_batch_button') and self.start_batch_button.winfo_exists(): self.start_batch_button.config(state=tk.DISABLED) # Disable batch during manual run
        if hasattr(self, 'current_jar_being_tested'): self.current_jar_being_tested.set(f"Manual: {os.path.basename(manual_jar_path)}")
        if hasattr(self, 'batch_progress_label') and self.batch_progress_label.winfo_exists(): self.batch_progress_label.config(text="Status: Running Manual Test...")

        # Run in a separate thread using the same task function
        # Use a lambda to wrap the call and send results back
        def manual_run_wrapper():
            dummy_stop = threading.Event() # Manual run cannot be stopped via GUI button easily
            # Sanitize test name for ID
            safe_test_name = re.sub(r'[^\w\-]+', '_', self.current_test_name)
            test_id = f"Manual_{safe_test_name}"
            result = None
            try:
                # Pass the correct absolute paths
                result = _run_one_batch_test_task(
                    test_id,
                    test_data_lines,
                    manual_jar_path, # Absolute path already joined
                    datainput_exe,   # Absolute path assumed from selection/config
                    dummy_stop
                )
            except Exception as manual_err:
                # Catch errors within the task execution call itself
                self.message_queue.put(("log", f"--- ERROR starting manual test task ---"))
                self.message_queue.put(("log", str(manual_err)))
                self.message_queue.put(("log", traceback.format_exc()))
                result = (test_id, False, f"Error during task execution:\n{manual_err}", [str(manual_err)], test_data_lines) # Synthesize a failure result
            finally:
                # Always signal finish, sending result (even if it's an error result)
                self.message_queue.put(("manual_finished", result)) # Send tuple back

        # Start the thread
        threading.Thread(target=manual_run_wrapper, daemon=True).start()
        # Start polling the queue for manual results
        self.after(100, self._process_manual_queue)

    def _ask_which_jar(self, jar_files):
        """Simple modal dialog to select a JAR from a list."""
        dialog = tk.Toplevel(self)
        dialog.title("Select JAR for Manual Test")
        dialog.geometry("450x300")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)

        ttk.Label(dialog, text="Multiple JARs found in the selected folder.\nPlease select one to use for this manual test:").pack(pady=10, padx=10)

        listbox_frame = ttk.Frame(dialog)
        listbox_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        listbox = tk.Listbox(listbox_frame, height=10)
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=listbox.yview)
        listbox.config(yscrollcommand=scrollbar.set)

        sorted_jars = sorted(jar_files)
        for item in sorted_jars: listbox.insert(tk.END, item)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        if sorted_jars: listbox.selection_set(0); listbox.see(0); listbox.activate(0) # Default selection

        selected_jar = tk.StringVar()

        def on_ok():
            idx = listbox.curselection()
            if idx: selected_jar.set(listbox.get(idx[0]))
            else: selected_jar.set("") # Should not happen with default
            dialog.destroy()

        def on_cancel():
            selected_jar.set("")
            dialog.destroy()

        # Add double-click binding
        listbox.bind("<Double-Button-1>", lambda e: on_ok())

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=(5, 10))
        ok_button = ttk.Button(button_frame, text="OK", command=on_ok, width=12)
        ok_button.pack(side=tk.LEFT, padx=10)
        cancel_button = ttk.Button(button_frame, text="Cancel", command=on_cancel, width=12)
        cancel_button.pack(side=tk.LEFT, padx=10)

        dialog.protocol("WM_DELETE_WINDOW", on_cancel)
        # Center dialog (optional, might depend on window manager)
        # dialog.update_idletasks()
        # x = self.winfo_rootx() + (self.winfo_width() // 2) - (dialog.winfo_width() // 2)
        # y = self.winfo_rooty() + (self.winfo_height() // 2) - (dialog.winfo_height() // 2)
        # dialog.geometry(f"+{x}+{y}")

        self.wait_window(dialog) # Wait for dialog to close
        return selected_jar.get()


    def _process_manual_queue(self):
        """Processes messages from the manual test worker thread. Runs in GUI thread."""
        try:
            while True: # Process all available messages
                if not self.winfo_exists(): return

                msg_type, data = self.message_queue.get_nowait()

                if msg_type == "log":
                    self._log_output(data)
                elif msg_type == "manual_finished":
                    result_tuple = data
                    # --- Process the result ---
                    if result_tuple:
                        test_id, success, output_log, validation_issues, input_lines = result_tuple

                        self._log_output("\n" + "="*10 + f" Manual Test '{self.current_test_name}' Finished " + "="*10)
                        if success:
                            self._log_output("Result: PASSED")
                            if hasattr(self, 'batch_progress_label') and self.batch_progress_label.winfo_exists(): self.batch_progress_label.config(text="Status: Manual Test PASSED")
                        else:
                            self._log_output(f"Result: FAILED")
                            if hasattr(self, 'batch_progress_label') and self.batch_progress_label.winfo_exists(): self.batch_progress_label.config(text="Status: Manual Test FAILED")
                            if validation_issues:
                                self._log_output("Validation/Process Issues:")
                                for issue in validation_issues: self._log_output(f"  - {issue}")

                            # Save failure artifacts
                            try:
                                manual_fail_dir = os.path.join(self.base_failures_dir, MANUAL_FAILURES_DIR)
                                os.makedirs(manual_fail_dir, exist_ok=True)
                                safe_test_id_base = re.sub(r'[^\w\-]+', '_', test_id)
                                ts_suffix = time.strftime("%Y%m%d_%H%M%S")
                                fail_prefix = os.path.join(manual_fail_dir, f"{safe_test_id_base}_{ts_suffix}")
                                fail_in = f"{fail_prefix}_stdin.txt"
                                fail_log = f"{fail_prefix}_output.log"
                                # Get JAR name used for this manual test
                                jar_name_manual = "Unknown"
                                if hasattr(self, 'current_jar_being_tested'):
                                     jar_name_manual = self.current_jar_being_tested.get().replace('Manual: ','')

                                with open(fail_in, 'w', encoding=PROCESS_ENCODING, errors='replace') as f_in:
                                    f_in.write(f"# Manual Test ID: {test_id}\n# JAR: {jar_name_manual}\n# Status: FAILED\n---\n")
                                    f_in.write("\n".join(input_lines))
                                with open(fail_log, 'w', encoding=PROCESS_ENCODING, errors='replace') as f_log:
                                    f_log.write(f"# Manual Test ID: {test_id}\n# JAR File: {jar_name_manual}\n# Status: FAILED\n")
                                    f_log.write(f"# Validation/Process Issues:\n")
                                    for issue in validation_issues: f_log.write(f"#   - {issue}\n")
                                    f_log.write("\n" + "="*10 + " Input (" + str(len(input_lines)) + " lines) " + "="*10 + "\n")
                                    f_log.write("\n".join(input_lines))
                                    f_log.write("\n\n" + "="*10 + " Full Output Log " + "="*10 + "\n")
                                    f_log.write(output_log)
                                self._log_output(f"(Failure details saved in '{os.path.join(MANUAL_FAILURES_DIR, os.path.basename(fail_prefix))}*')")
                            except Exception as e_save_manual:
                                self._log_output(f"(ERROR saving manual failure details: {e_save_manual})")

                        # Log full output
                        self._log_output("\n" + "-"*10 + " Full Output Log " + "-"*10)
                        log_lines_split = output_log.splitlines()
                        log_chunk_size = 200 # Log in chunks to avoid freezing GUI
                        for i in range(0, len(log_lines_split), log_chunk_size):
                            chunk = "\n".join(log_lines_split[i:i+log_chunk_size])
                            # Use after_idle to schedule logging without blocking queue processing
                            self.after_idle(self._log_output, chunk)
                    else:
                        # Result tuple was None (error during task call)
                        self._log_output("--- Manual test finished with internal execution error (see logs above) ---")
                        if hasattr(self, 'batch_progress_label') and self.batch_progress_label.winfo_exists(): self.batch_progress_label.config(text="Status: Manual Test ERROR")

                    # --- Reset GUI state after manual run ---
                    if not self.is_batch_running: # Ensure batch didn't start in the meantime
                        if hasattr(self, 'start_batch_button') and self.start_batch_button.winfo_exists(): self.start_batch_button.config(state=tk.NORMAL)
                        current_progress_text = ""
                        if hasattr(self, 'batch_progress_label') and self.batch_progress_label.winfo_exists():
                             current_progress_text = self.batch_progress_label['text']
                             if "Manual Test" in current_progress_text:
                                  # Only reset status if it was showing Manual Test status
                                  if "PASSED" in current_progress_text:
                                       self.batch_progress_label.config(text="Status: Idle")
                                  # Otherwise leave FAILED/ERROR message displayed
                        if hasattr(self, 'current_jar_being_tested'): self.current_jar_being_tested.set("N/A")
                        self._on_test_select() # Re-evaluate manual button state
                    return # Stop polling queue for this manual run

        except queue.Empty:
            # No more messages currently
            pass
        except Exception as e:
            jar_context = self.current_jar_being_tested.get() if hasattr(self, 'current_jar_being_tested') else 'N/A'
            print(f"ERROR processing manual queue (JAR context: {jar_context}): {e}\n{traceback.format_exc()}", file=sys.stderr) # Debug
            self._log_output(f"--- ERROR processing manual queue: {e} ---")
            # Attempt to reset state on error?
            if not self.is_batch_running:
                if hasattr(self, 'start_batch_button') and self.start_batch_button.winfo_exists(): self.start_batch_button.config(state=tk.NORMAL)
                if hasattr(self, 'batch_progress_label') and self.batch_progress_label.winfo_exists(): self.batch_progress_label.config(text="Status: Error processing results")
                if hasattr(self, 'current_jar_being_tested'): self.current_jar_being_tested.set("Error")
                self._on_test_select()

        # Reschedule check ONLY if manual run is still indicated as running
        # Check if run_manual_button exists and is disabled, AND batch is not running
        manual_running_indicated = False
        if hasattr(self, 'run_manual_button') and self.run_manual_button.winfo_exists():
             manual_running_indicated = (self.run_manual_button['state'] == tk.DISABLED and not self.is_batch_running)

        if self.winfo_exists() and manual_running_indicated:
            self.after(150, self._process_manual_queue)


    # --- Utility Methods ---
    def _clear_output_log(self):
        """Clears the main output log text area."""
        if not hasattr(self, 'output_log_text') or not self.output_log_text.winfo_exists(): return
        try:
            self.output_log_text.config(state=tk.NORMAL)
            self.output_log_text.delete('1.0', tk.END)
            self.output_log_text.config(state=tk.DISABLED)
        except tk.TclError: pass # Ignore error if widget is destroyed


    def _log_output(self, message):
        """Safely logs messages to the ScrolledText widget. Runs in GUI thread or schedules call."""
        # Check if called from non-GUI thread
        if threading.current_thread() is not threading.main_thread():
            # Schedule the call in the main GUI thread
            try:
                # Check if window exists before scheduling using widget check
                 if hasattr(self, 'output_log_text') and self.output_log_text.winfo_exists():
                      # Use after_idle for non-blocking update
                      self.after_idle(self._log_output, message)
                 # else: print(f"DEBUG (Thread): Window closed, discarding log: {message[:50]}...") # Debug log discard
            except Exception as e_schedule:
                 # Catch potential errors during scheduling itself (e.g., if app is shutting down)
                 print(f"ERROR scheduling log message: {e_schedule}", file=sys.stderr)
            return

        # --- This part runs only in the GUI thread ---
        if not hasattr(self, 'output_log_text') or not self.output_log_text.winfo_exists():
             # print(f"DEBUG (GUI): Widget destroyed, discarding log: {str(message)[:50]}...") # Debug log discard
             return
        try:
            msg = str(message) # Ensure string
            self.output_log_text.config(state=tk.NORMAL)
            self.output_log_text.insert(tk.END, msg + "\n")
            self.output_log_text.see(tk.END) # Auto-scroll
            self.output_log_text.config(state=tk.DISABLED)
            # self.update_idletasks() # Use sparingly, can impact performance heavily
        except tk.TclError: pass # Ignore error if widget destroyed during update
        except Exception as e_log:
             print(f"ERROR logging to GUI text widget: {e_log}", file=sys.stderr)


    def _quit(self):
        """Handles application exit gracefully."""
        quit_app = True # Assume we can quit unless user cancels

        if self.is_batch_running:
            if messagebox.askyesno("Exit Confirmation", "A batch test is currently running.\nStopping the batch may take a moment.\n\nDo you want to stop the batch and exit?", parent=self):
                self._log_output("--- EXIT REQUESTED - Stopping Batch ---")
                self.stop_batch_event.set()
                if hasattr(self, 'stop_batch_button') and self.stop_batch_button.winfo_exists():
                    self.stop_batch_button.config(state=tk.DISABLED) # Prevent further clicks
            else:
                quit_app = False # User cancelled exit

        # Check if manual test *might* be running
        manual_running_indicated = False
        if hasattr(self, 'run_manual_button') and self.run_manual_button.winfo_exists():
             manual_running_indicated = (self.run_manual_button['state'] == tk.DISABLED and not self.is_batch_running)

        if not self.is_batch_running and manual_running_indicated: # Only ask if manual *might* be running
             if not messagebox.askyesno("Exit Confirmation", "A manual test might be running in the background (no stop mechanism).\n\nExit anyway?", parent=self):
                  quit_app = False

        if quit_app:
             self._save_config() # Save paths on exit
             self.destroy() # Cleanly destroy the window

# --- Main Execution ---
if __name__ == "__main__":
    # Create the main application window FIRST
    app = ElevatorTesterApp()

    # NOW configure fonts, as the root window exists
    try:
        from tkinter import font
        root_font = font.nametofont("TkDefaultFont")
        root_font.configure(size=9) # Smaller default size
        # Configure specific fonts
        text_font = font.nametofont("TkTextFont")
        text_font.configure(family="Consolas", size=9) # Good for logs
        fixed_font = font.nametofont("TkFixedFont")
        fixed_font.configure(family="Consolas", size=9)
        # Apply explicitly if needed (safer)
        if hasattr(app, 'output_log_text') and app.output_log_text.winfo_exists():
             app.output_log_text.config(font=text_font)
        if hasattr(app, 'input_text') and app.input_text.winfo_exists():
             app.input_text.config(font=text_font)
        # print("Note: Default fonts configured.") # Optional confirmation
    except Exception as e:
        print(f"Note: Could not configure default fonts - {e}", file=sys.stderr)

    # Start the Tkinter event loop
    app.mainloop()