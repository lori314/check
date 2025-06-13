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
ELEVATOR_IDS = list(range(1, 7))
NUM_ELEVATORS = 6
PRIORITY_RANGE = (1, 20)
SCHEDULE_SPEEDS = [0.2, 0.3, 0.4, 0.5]
PROCESS_ENCODING = 'utf-8' # Or 'gbk' if needed for specific environments
STDIN_FILENAME = "stdin.txt"
FAILURES_DIR = "batch_failures"
MANUAL_FAILURES_DIR = "_Manual_Failures" # Subdir within FAILURES_DIR
DEFAULT_CONCURRENCY = max(1, os.cpu_count() // 2 if os.cpu_count() else 4) # Safer default concurrency
T_MAX_HUTEST = 220.0 # Max time for 互测 (peer testing)

# --- Validation Constants ---
DEFAULT_SPEED = 0.4 # s/floor
MIN_DOOR_TIME = 0.4 # s (>= 0.4s means minimal open duration is 0.4s)
CAPACITY = 6
SCHE_MAX_ARRIVES_BEFORE_BEGIN = 2
SCHE_MAX_COMPLETE_TIME = 6.0 # s (From ACCEPT to END)
# <<< FIX: Define allowed schedule target floors >>>
ALLOWED_SCHE_TARGET_FLOORS = ['B2', 'B1', 'F1', 'F2', 'F3', 'F4', 'F5']
TIMESTAMP_TOLERANCE = 0.005 # Small tolerance for float comparisons
SPEED_TOLERANCE_FACTOR_LOW = 0.85 # Allow slightly faster than expected
SPEED_TOLERANCE_FACTOR_HIGH = 1.6 # Allow significantly slower (e.g., first move)

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

# --- Assume these constants are defined globally ---
ELEVATOR_IDS = list(range(1, 7))
FLOORS = [f"B{i}" for i in range(4, 0, -1)] + [f"F{i}" for i in range(1, 8)]
PRIORITY_RANGE = (1, 20)
SCHEDULE_SPEEDS = [0.2, 0.3, 0.4, 0.5]
ALLOWED_SCHE_TARGET_FLOORS = ['B2', 'B1', 'F1', 'F2', 'F3', 'F4', 'F5']
SCHE_MAX_COMPLETE_TIME = 6.0 # Max duration from ACCEPT to END
TIMESTAMP_TOLERANCE = 0.005
# --- End Constants ---

def generate_test_data(num_passengers, num_schedules, max_time):
    """
    Generates test data with passenger and schedule requests.
    Ensures SCHE commands for the same elevator are spaced out sufficiently
    to allow the previous one to potentially complete.
    """
    requests_timed = []
    used_passenger_ids = set()
    # Track the earliest time the *next* SCHE command can be *sent* for each elevator
    elevator_next_sche_send_allowed_time = {eid: 0.0 for eid in ELEVATOR_IDS}

    ALL_SYSTEM_FLOORS = FLOORS
    AVAILABLE_ELEVATOR_IDS = ELEVATOR_IDS

    # --- Generate passenger requests ---
    # (This part remains the same as the previous good version)
    for _ in range(num_passengers):
        pid = random.randint(1, 9999)
        while pid in used_passenger_ids: pid = random.randint(1, 9999)
        used_passenger_ids.add(pid)

        from_fl = random.choice(ALL_SYSTEM_FLOORS)
        possible_to_fl = [f for f in ALL_SYSTEM_FLOORS if f != from_fl]
        if not possible_to_fl: continue
        to_fl = random.choice(possible_to_fl)

        priority = random.randint(PRIORITY_RANGE[0], PRIORITY_RANGE[1])
        # Ensure requests don't cluster too much at the very end, leave buffer
        request_max_send_time = max(0.6, max_time - (SCHE_MAX_COMPLETE_TIME + 5.0)) # Ensure passengers arrive before potential final SCHE ends
        send_time = round(random.uniform(0.1, request_max_send_time), 4)
        cmd = f"{pid}-PRI-{priority}-FROM-{from_fl}-TO-{to_fl}"
        requests_timed.append((send_time, cmd))

    # --- Generate schedule requests ---
    MIN_SCHE_INTERVAL = SCHE_MAX_COMPLETE_TIME + TIMESTAMP_TOLERANCE + 0.1 # Min time between end of one SCHE and *sending* the next for same elevator
    SCHE_SEND_BUFFER = 1.0 # Don't send SCHE commands right at the very end of max_time

    schedule_requests_generated = 0
    attempts = 0
    max_attempts = num_schedules * 5 # Try harder to generate requested schedules

    while schedule_requests_generated < num_schedules and attempts < max_attempts:
        attempts += 1
        if not ALLOWED_SCHE_TARGET_FLOORS: break # Stop if no valid targets

        eid = random.choice(AVAILABLE_ELEVATOR_IDS)
        speed = random.choice(SCHEDULE_SPEEDS)
        target_fl = random.choice(ALLOWED_SCHE_TARGET_FLOORS)

        # Determine the earliest possible send time for *this specific* elevator
        min_send_time_for_elevator = elevator_next_sche_send_allowed_time[eid]

        # Determine the latest possible send time globally
        # Ensure the command is sent early enough that it *could* complete within max_time
        # (Send time + Max completion time <= max_time - buffer)
        latest_global_send_time = max(min_send_time_for_elevator, max_time - SCHE_MAX_COMPLETE_TIME - SCHE_SEND_BUFFER)

        # Check if a valid time slot exists
        if min_send_time_for_elevator >= latest_global_send_time:
             # No valid time slot for this elevator within the constraints, try another elevator/attempt
             continue

        # Choose a random send time within the valid window
        send_time = round(random.uniform(min_send_time_for_elevator, latest_global_send_time), 4)

        # If somehow the chosen time is still too early (shouldn't happen with logic above), skip
        if send_time < elevator_next_sche_send_allowed_time[eid] - TIMESTAMP_TOLERANCE:
            continue

        # Add the schedule request
        cmd = f"SCHE-{eid}-{speed}-{target_fl}"
        requests_timed.append((send_time, cmd))
        schedule_requests_generated += 1

        # Update the next allowed send time for this elevator
        # Next send must be after this one *could* finish
        # Estimated finish time = send_time + SCHE_MAX_COMPLETE_TIME
        # Add a small buffer (MIN_SCHE_INTERVAL includes this)
        elevator_next_sche_send_allowed_time[eid] = send_time + MIN_SCHE_INTERVAL

    if schedule_requests_generated < num_schedules:
         print(f"WARNING: Could only generate {schedule_requests_generated}/{num_schedules} schedule requests due to timing constraints.")
         # This isn't necessarily an error, might just be tight timing.

    # --- Final Sort and Formatting ---
    requests_timed.sort(key=lambda x: x[0])
    formatted_lines = [f"[{t:.4f}]{cmd}" for t, cmd in requests_timed]
    return formatted_lines

# --- DETAILED VALIDATION FUNCTION ---
# Corrected version based on user's provided structure and latest logic refinements
def validate_output(output_log_lines, input_lines):
    """
    Performs detailed validation of elevator output log against rules.
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
    active_sche_commands = {} # Key: eid (int), Value: {'speed': float, 'target': str, 'sent_ts': float} - Track commands from input that might be accepted

    # Initialize elevator states
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
            'last_floor_ts': 0.0, # Timestamp of the last ARRIVE event
            'last_open_ts': 0.0,  # Timestamp of the last OPEN event
            'last_close_ts': 0.0, # Timestamp of the last CLOSE event
            # SCHE State
            'sche_active': False,       # Is a schedule command currently being processed?
            'sche_phase': None,         # 'ACCEPTED', 'BEGUN', None
            'sche_target_floor': None,  # Target floor for the current SCHE (str)
            'sche_speed': None,         # Speed for the current SCHE (float)
            'sche_accept_ts': -1.0,     # Timestamp of SCHE-ACCEPT
            'sche_begin_ts': -1.0,      # Timestamp of SCHE-BEGIN
            'sche_end_ts': -1.0,        # Timestamp of SCHE-END
            'sche_arrives_since_accept': 0, # Count of ARRIVEs between ACCEPT and BEGIN
            'sche_opened_at_target': False, # Flag: did the door OPEN at the SCHE target floor during the BEGUN phase?
        }

    # Initialize passenger states from input (basic info)
    for line in input_lines:
        ts_input = parse_output_timestamp(line) or 0.0 # Get input timestamp
        action_part_input = line.split(']', 1)[-1].strip() if ts_input > 0 else line.strip()

        match_pri = re.match(r'(\d+)-PRI-\d+-FROM-([BF]\d+)-TO-([BF]\d+)', action_part_input)
        match_sche = re.match(r'SCHE-(\d+)-([\d.]+)-([BF]\d+)', action_part_input)

        if match_pri:
            try: # Add safety for parsing
                pid = int(match_pri.group(1))
                from_f = match_pri.group(2)
                to_f = match_pri.group(3)
                if pid not in passengers: # Only add if not already seen
                     passengers[pid] = {
                        'id': pid, 'location': from_f, 'destination': to_f, 'state': 'WAITING',
                        'assigned_elevator': None, 'request_time': ts_input,
                     }
                     passenger_requests_from_input[pid] = {'from': from_f, 'to': to_f}
                # else: issues.append(f"INPUT WARNING: Duplicate passenger request ID {pid} in input.")
            except (ValueError, IndexError):
                 issues.append(f"INPUT WARNING: Malformed passenger request in input: {line}")


        elif match_sche:
             try: # Add safety for parsing
                 eid = int(match_sche.group(1))
                 speed = float(match_sche.group(2))
                 target_fl = match_sche.group(3)
                 if eid in ELEVATOR_IDS:
                      active_sche_commands[eid] = {'speed': speed, 'target': target_fl, 'sent_ts': ts_input}
                 # else: issues.append(f"INPUT WARNING: Invalid elevator ID {eid} in SCHE command: {line}")
             except (ValueError, IndexError):
                  issues.append(f"INPUT WARNING: Malformed SCHE command in input: {line}")


    # --- Regex Patterns (Compile for efficiency) ---
    patterns = {
        # Action-Floor-ElevatorID
        'ARRIVE': re.compile(r'ARRIVE-([BF]\d+)-(\d+)$'),
        'OPEN':   re.compile(r'OPEN-([BF]\d+)-(\d+)$'),
        'CLOSE':  re.compile(r'CLOSE-([BF]\d+)-(\d+)$'),
        # Action-PassengerID-Floor-ElevatorID
        'IN':     re.compile(r'IN-(\d+)-([BF]\d+)-(\d+)$'),
        'OUT_S':  re.compile(r'OUT-S-(\d+)-([BF]\d+)-(\d+)$'), # Successful arrival at DESTINATION
        'OUT_F':  re.compile(r'OUT-F-(\d+)-([BF]\d+)-(\d+)$'), # Forced/Finished at NON-DESTINATION
        # Action-PassengerID-ElevatorID
        'RECEIVE':re.compile(r'RECEIVE-(\d+)-(\d+)$'),
        # Action-ElevatorID-Speed-Floor (SCHE Commands)
        'ACCEPT': re.compile(r'SCHE-ACCEPT-(\d+)-([\d.]+)-([BF]\d+)$'),
        # Action-ElevatorID
        'BEGIN':  re.compile(r'SCHE-BEGIN-(\d+)$'),
        'END':    re.compile(r'SCHE-END-(\d+)$')
    }

    # --- Process Log Lines ---
    for line_num, raw_line in enumerate(output_log_lines, 1):
        line = raw_line.strip()
        if not line or line.startswith('#') or line.startswith('--'): # Skip empty lines and comments
            continue

        ts = parse_output_timestamp(line)
        action_part = line # Default to full line if no timestamp

        # Validate and update timestamp
        if ts is not None:
            if ts < last_ts - TIMESTAMP_TOLERANCE:
                issues.append(f"[L{line_num} @ {ts:.4f}] Timestamp decreased: {ts:.4f} is less than previous {last_ts:.4f} in: {raw_line}")
            if ts >= last_ts - TIMESTAMP_TOLERANCE:
                 current_time = ts
                 last_ts = max(ts, last_ts) # Ensure last_ts never decreases

            action_part = line.split(']', 1)[-1].strip() # Get part after timestamp
            if not action_part:
                 issues.append(f"[L{line_num} @ {current_time:.4f}] Line has timestamp but no action: {raw_line}")
                 continue
        else:
            # Only allow certain non-timestamped lines (like debug/stderr from student)
            # Core elevator actions MUST have timestamps
            if not any(action_part.startswith(kw) for kw in ['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL', 'Result:', '---', '[', '#', 'at ', '\tat ']): # Allow common stderr prefixes
                 issues.append(f"[L{line_num}] Malformed Line / Missing Timestamp for core action?: {raw_line}")
                 # Optionally add warning about using last known time for processing non-timestamped potentially relevant lines
                 # issues.append(f"[L{line_num} @ {current_time:.4f}] WARNING: Action processed using last known time due to missing timestamp: {raw_line}")


        # --- Match Action ---
        matched_action = None
        action_data = None
        for action, pattern in patterns.items():
            match = pattern.match(action_part)
            if match:
                matched_action = action
                action_data = match.groups()
                break

        if not matched_action:
            # Avoid flagging standard stderr/debug messages as unknown actions
            # Add more stderr prefixes if needed
            if not any(action_part.startswith(kw) for kw in ['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL', 'Result:', '---', '[', '#', 'at ', '\tat ']):
                 issues.append(f"[L{line_num} @ {current_time:.4f}] Unknown or Malformed Action Format: {action_part}")
            continue # Skip processing unknown actions or allowed non-actions

        # --- Get Elevator/Passenger State (handle potential errors) ---
        eid = None
        pid = None
        e = None
        p = None
        current_event_time = ts if ts is not None else current_time # Use actual event time if available

        try:
            # Extract IDs based on action type
            if matched_action in ['ARRIVE', 'OPEN', 'CLOSE', 'BEGIN', 'END']:
                eid = int(action_data[-1])
            elif matched_action in ['IN', 'OUT_S', 'OUT_F']:
                pid = int(action_data[0])
                eid = int(action_data[-1])
            elif matched_action == 'RECEIVE':
                pid = int(action_data[0])
                eid = int(action_data[1])
            elif matched_action == 'ACCEPT':
                eid = int(action_data[0])

            # Fetch state dictionaries
            if eid is not None:
                if eid not in elevators:
                    issues.append(f"[L{line_num} @ {current_event_time:.4f}] {matched_action}: Invalid elevator ID {eid}: {raw_line}")
                    continue # Cannot proceed without valid elevator
                e = elevators[eid]
                # Update last action time for this elevator if it's involved
                # Moved this update to *after* successful processing of the action for 'e'
                # e['last_action_ts'] = current_event_time

            if pid is not None:
                if pid not in passengers:
                    # Only raise error if the action requires a known passenger (IN, OUT, RECEIVE)
                    if matched_action in ['IN', 'OUT_S', 'OUT_F', 'RECEIVE']:
                        issues.append(f"[L{line_num} @ {current_event_time:.4f}] {matched_action}: Unknown passenger ID {pid}: {raw_line}")
                        # If IN/OUT, we might still need elevator state, so don't 'continue' yet if e is valid
                        if matched_action == 'RECEIVE': continue # RECEIVE needs known passenger
                else:
                    p = passengers[pid]
                # If passenger is unknown for IN/OUT, p will be None, handle checks accordingly later

        except (ValueError, IndexError) as id_err:
            issues.append(f"[L{line_num} @ {current_event_time:.4f}] Error parsing ID for action '{matched_action}': {id_err} in {raw_line}")
            continue

        # --- Process Matched Action ---
        try:
            # --- ARRIVE ---
            if matched_action == 'ARRIVE':
                floor = action_data[0]
                if e is None: continue # Should be caught above, but defensive check

                # Check 1: Door must be closed WHEN arriving
                if e['door'] != 'CLOSED':
                     issues.append(f"[L{line_num} @ {current_event_time:.4f}] E{eid} ARRIVE-{floor} door not CLOSED (was {e['door']}): {raw_line}")

                # Check 2: Floor validity and sequence
                previous_floor = e['floor'] # Get floor before state update
                if floor not in FLOOR_ORDER:
                     issues.append(f"[L{line_num} @ {current_event_time:.4f}] E{eid} ARRIVE invalid floor '{floor}': {raw_line}")
                     e['floor'] = floor; e['motion'] = 'IDLE'; e['last_floor_ts'] = current_event_time # Stop at invalid floor
                     e['last_action_ts'] = current_event_time # Update action time
                     continue

                diff = floor_diff(previous_floor, floor)
                if e['last_floor_ts'] > 0: # Check sequence only if not the first recorded move
                     if diff == 0:
                          issues.append(f"[L{line_num} @ {current_event_time:.4f}] WARN: E{eid} ARRIVE at same floor {floor} it was last at ({previous_floor}).")
                     elif diff != 1:
                          issues.append(f"[L{line_num} @ {current_event_time:.4f}] E{eid} ARRIVE floor jump: {previous_floor}->{floor} (diff {diff}): {raw_line}")

                # Check 3: Timing (approximate)
                current_speed = e['speed']
                time_elapsed = current_event_time - e['last_floor_ts']
                if e['last_floor_ts'] > 0 and diff == 1 and current_speed is not None and current_speed > 0:
                    expected_time = current_speed
                    # Adjusted tolerance factors slightly
                    low_bound = expected_time * 0.80 - TIMESTAMP_TOLERANCE # Allow a bit faster
                    high_bound = expected_time * 1.5 + TIMESTAMP_TOLERANCE # Allow more time for variability
                    if time_elapsed > 0.001 and not (low_bound <= time_elapsed <= high_bound): # Ignore near-zero elapsed time
                         issues.append(f"[L{line_num} @ {current_event_time:.4f}] WARN: E{eid} ARRIVE timing? {time_elapsed:.4f}s vs ~{expected_time:.4f}s ({previous_floor}->{floor}, speed={current_speed})")
                elif diff == 1 and (current_speed is None or current_speed <= 0):
                    issues.append(f"[L{line_num} @ {current_event_time:.4f}] WARN: E{eid} ARRIVE check invalid speed ({current_speed}) for timing calc.")


                # Check 4: SCHE Arrive Count (between ACCEPT and BEGIN)
                if e['sche_active'] and e['sche_phase'] == 'ACCEPTED':
                    e['sche_arrives_since_accept'] += 1
                    if e['sche_arrives_since_accept'] > SCHE_MAX_ARRIVES_BEFORE_BEGIN:
                        issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} had >{SCHE_MAX_ARRIVES_BEFORE_BEGIN} ARRIVEs after ACCEPT before BEGIN: {raw_line}")

                # Update state AFTER all checks
                e['floor'] = floor
                e['motion'] = 'IDLE' # Elevator is IDLE immediately upon arrival
                e['last_floor_ts'] = current_event_time # Record time of arrival
                e['last_action_ts'] = current_event_time # Update action time

            # --- OPEN ---
            elif matched_action == 'OPEN':
                floor = action_data[0]
                if e is None: continue

                # Check 1: Location consistency
                if e['floor'] != floor:
                    issues.append(f"[L{line_num} @ {current_event_time:.4f}] E{eid} OPEN at {floor} but is currently at {e['floor']}: {raw_line}")

                # Check 2: Door must be CLOSED to open
                if e['door'] != 'CLOSED':
                    issues.append(f"[L{line_num} @ {current_event_time:.4f}] E{eid} OPEN command received but door was already {e['door']}: {raw_line}")

                # Check 3: Motion state (should be IDLE based on new logic)
                if e['motion'] != 'IDLE':
                     issues.append(f"[L{line_num} @ {current_event_time:.4f}] WARN/INTERNAL?: E{eid} OPEN command received but motion state was {e['motion']} (Should be IDLE): {raw_line}")

                # Check 4: SCHE violation check (only during BEGUN phase before target)
                if e['sche_active'] and e['sche_phase'] == 'BEGUN' and e['floor'] != e['sche_target_floor']:
                    issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} OPEN at {floor} during SCHE move phase (target {e['sche_target_floor']}): {raw_line}")

                # Update state
                e['door'] = 'OPEN'
                e['last_open_ts'] = current_event_time
                e['motion'] = 'IDLE' # Opening always results in IDLE state
                e['last_action_ts'] = current_event_time # Update action time

                # Set flag if this OPEN meets the SCHE target requirement
                if e['sche_active'] and e['sche_phase'] == 'BEGUN' and e['floor'] == e['sche_target_floor']:
                     e['sche_opened_at_target'] = True

            # --- CLOSE ---
            elif matched_action == 'CLOSE':
                floor = action_data[0]
                if e is None: continue

                # Check 1: Location consistency
                if e['floor'] != floor:
                    issues.append(f"[L{line_num} @ {current_event_time:.4f}] E{eid} CLOSE at {floor} but is currently at {e['floor']}: {raw_line}")

                # Check 2: Door must be OPEN to close
                if e['door'] != 'OPEN':
                    issues.append(f"[L{line_num} @ {current_event_time:.4f}] E{eid} CLOSE command received but door was already {e['door']}: {raw_line}")

                # Check 3: Timing - Door must stay open for min duration
                time_since_open = current_event_time - e['last_open_ts']
                if e['last_open_ts'] > 0 and time_since_open < (MIN_DOOR_TIME - TIMESTAMP_TOLERANCE):
                    issues.append(f"[L{line_num} @ {current_event_time:.4f}] E{eid} CLOSE too fast at {floor} ({time_since_open:.4f}s < {MIN_DOOR_TIME}s): {raw_line}")

                # No specific SCHE phase checks needed here anymore based on refined rules.

                # Update state
                e['door'] = 'CLOSED'
                e['last_close_ts'] = current_event_time
                # Default to IDLE; movement is confirmed only by ARRIVE.
                e['motion'] = 'IDLE'
                e['last_action_ts'] = current_event_time # Update action time

            # --- IN ---
            elif matched_action == 'IN':
                pid_str, floor, eid_str = action_data; pid = int(pid_str)
                # Allow check even if p is None initially (will fail later if truly unknown)
                if e is None: continue

                # Perform checks, set err_in flag on failure
                err_in = False
                if e['floor'] != floor: issues.append(f"[L{line_num} @ {current_event_time:.4f}] P{pid} IN E{eid} at {floor}, E@ {e['floor']}"); err_in=True
                if e['door'] != 'OPEN': issues.append(f"[L{line_num} @ {current_event_time:.4f}] P{pid} IN E{eid} at {floor}, door {e['door']}"); err_in=True
                # Check passenger state only if passenger is known
                if p is None: issues.append(f"[L{line_num} @ {current_event_time:.4f}] P{pid} IN E{eid}, unknown passenger ID"); err_in=True
                else: # Passenger known, perform detailed checks
                    if p['location'] != floor: issues.append(f"[L{line_num} @ {current_event_time:.4f}] P{pid} IN E{eid} at {floor}, P.loc '{p['location']}'"); err_in=True
                    if p['state'] != 'WAITING': issues.append(f"[L{line_num} @ {current_event_time:.4f}] P{pid} IN E{eid}, state {p['state']} != WAITING"); err_in=True
                    assigned_eid = received_requests.get(pid)
                    if assigned_eid is None: issues.append(f"[L{line_num} @ {current_event_time:.4f}] P{pid} IN E{eid} without RECEIVE"); err_in=True
                    elif assigned_eid != eid: issues.append(f"[L{line_num} @ {current_event_time:.4f}] P{pid} IN E{eid}, but RECEIVE'd by E{assigned_eid}"); err_in=True
                # Check capacity and SCHE state regardless of passenger known status
                if len(e['passengers']) >= e['capacity']: issues.append(f"[L{line_num} @ {current_event_time:.4f}] E{eid} full (cap {e['capacity']}), cannot IN P{pid}"); err_in=True
                if e['sche_active'] and e['sche_phase'] == 'BEGUN': issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: P{pid} IN E{eid} during SCHE"); err_in=True

                # Update state only if no critical errors *and* passenger is known
                if not err_in and p is not None:
                    e['passengers'].add(pid); p['location'] = eid; p['state'] = 'TRANSIT'; p['assigned_elevator'] = eid
                e['last_action_ts'] = current_event_time # Update action time

            # --- OUT_S / OUT_F ---
            elif matched_action in ['OUT_S', 'OUT_F']:
                pid_str, floor, eid_str = action_data; pid = int(pid_str)
                # Allow check even if p is None initially (will fail later if truly unknown)
                if e is None: continue

                err_out = False
                if e['floor'] != floor: issues.append(f"[L{line_num} @ {current_event_time:.4f}] P{pid} {matched_action} E{eid} at {floor}, E@ {e['floor']}"); err_out = True
                if e['door'] != 'OPEN': issues.append(f"[L{line_num} @ {current_event_time:.4f}] P{pid} {matched_action} E{eid} at {floor}, door {e['door']}"); err_out = True
                if pid not in e['passengers']: issues.append(f"[L{line_num} @ {current_event_time:.4f}] P{pid} {matched_action} E{eid} at {floor}, P not inside"); err_out = True
                # Check passenger state only if known
                elif p is None: issues.append(f"[L{line_num} @ {current_event_time:.4f}] P{pid} {matched_action} E{eid}, unknown passenger ID"); err_out = True
                elif p['state'] != 'TRANSIT' or p['location'] != eid: issues.append(f"[L{line_num} @ {current_event_time:.4f}] P{pid} {matched_action} E{eid}, bad P state (st={p['state']}, loc={p['location']})"); err_out = True

                # SCHE Check remains the same
                is_sche_out_at_target = e['sche_active'] and e['sche_phase'] == 'BEGUN' and e['floor'] == e['sche_target_floor'] and e['door'] == 'OPEN'
                if e['sche_active'] and e['sche_phase'] == 'BEGUN' and not is_sche_out_at_target:
                     issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: P{pid} {matched_action} E{eid} during SCHE move/before target open"); err_out = True

                # Update elevator state regardless (remove passenger if they were inside)
                if pid in e['passengers']: e['passengers'].remove(pid)
                # Clear receive state regardless
                if pid in received_requests: del received_requests[pid]

                # Update passenger state only if known and basic checks passed (not strictly needed but cleaner)
                if p is not None: # Only update passenger state if passenger is known
                    p['location'] = floor
                    if p['assigned_elevator'] == eid: p['assigned_elevator'] = None # Clear assignment if it matches

                    # Update passenger state based on OUT type and destination match
                    is_destination = (floor == p['destination'])
                    if matched_action == 'OUT_S':
                        if is_destination: p['state'] = 'ARRIVED'
                        else: issues.append(f"[L{line_num} @ {current_event_time:.4f}] SEMANTIC VIOLATION: P{pid} OUT_S at {floor} != dest {p['destination']}."); p['state'] = 'WAITING'
                    elif matched_action == 'OUT_F':
                        if not is_destination: p['state'] = 'WAITING'
                        else: issues.append(f"[L{line_num} @ {current_event_time:.4f}] SEMANTIC VIOLATION: P{pid} OUT_F at destination {floor}."); p['state'] = 'ARRIVED'
                elif not err_out: # If passenger unknown but other checks okay, still log warning?
                    issues.append(f"[L{line_num} @ {current_event_time:.4f}] WARN: P{pid} {matched_action} E{eid}, unknown passenger ID, state not updated.")

                e['last_action_ts'] = current_event_time # Update action time

            # --- RECEIVE ---
            elif matched_action == 'RECEIVE':
                pid_str, eid_str = action_data; pid = int(pid_str)
                if p is None or e is None: continue # Must know both elevator and passenger

                err_receive = False
                if not isinstance(p['location'], str) or not re.match(r'[BF]\d+', p['location']): issues.append(f"[L{line_num} @ {current_event_time:.4f}] RECEIVE-{pid}-{eid} invalid: P{pid} not on floor (loc={p['location']})"); err_receive=True
                if p['state'] != 'WAITING': issues.append(f"[L{line_num} @ {current_event_time:.4f}] RECEIVE-{pid}-{eid} invalid: P{pid} state is {p['state']} (must be WAITING)"); err_receive=True
                current_assignment = received_requests.get(pid)
                if current_assignment is not None and current_assignment != eid: issues.append(f"[L{line_num} @ {current_event_time:.4f}] RECEIVE-{pid}-{eid} invalid: P{pid} already assigned to E{current_assignment}"); err_receive=True
                if e['sche_active'] and e['sche_phase'] == 'BEGUN': issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} RECEIVE P{pid} during SCHE move"); err_receive=True

                if not err_receive: # Update assignment only if checks pass
                     received_requests[pid] = eid; p['assigned_elevator'] = eid
                e['last_action_ts'] = current_event_time # Update action time

            # --- SCHE-ACCEPT ---
            elif matched_action == 'ACCEPT':
                eid_str, speed_str, target_floor = action_data
                if e is None: continue
                try: speed = float(speed_str)
                except ValueError: issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE-ACCEPT E{eid} invalid speed '{speed_str}'"); continue

                if target_floor not in FLOOR_ORDER: issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE-ACCEPT E{eid} invalid target floor '{target_floor}'"); continue
                if target_floor not in ALLOWED_SCHE_TARGET_FLOORS: issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} ACCEPT target floor '{target_floor}' not allowed.")
                if speed not in SCHEDULE_SPEEDS: issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} ACCEPT invalid speed '{speed}'")

                if e['sche_active']: issues.append(f"[L{line_num} @ {current_event_time:.4f}] WARN: E{eid} new SCHE-ACCEPT while active (Phase: {e['sche_phase']}). Overwriting.")

                e['sche_active'] = True; e['sche_phase'] = 'ACCEPTED'; e['sche_target_floor'] = target_floor
                e['sche_speed'] = speed; e['sche_accept_ts'] = current_event_time
                e['sche_begin_ts'] = -1.0; e['sche_end_ts'] = -1.0; e['sche_arrives_since_accept'] = 0; e['sche_opened_at_target'] = False
                e['last_action_ts'] = current_event_time # Update action time

            # --- SCHE-BEGIN ---
            elif matched_action == 'BEGIN':
                eid_str, = action_data
                if e is None: continue
                valid_begin = True
                if not e['sche_active'] or e['sche_phase'] != 'ACCEPTED': issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} BEGIN invalid state (active={e['sche_active']}, phase={e['sche_phase']})"); valid_begin = False
                if e['door'] != 'CLOSED': issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} BEGIN door not CLOSED ({e['door']})"); valid_begin = False
                if e['motion'] != 'IDLE': issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} BEGIN while {e['motion']}"); valid_begin = False
                if e['sche_arrives_since_accept'] > SCHE_MAX_ARRIVES_BEFORE_BEGIN: pass # Already logged

                if valid_begin:
                    e['sche_phase'] = 'BEGUN'; e['sche_begin_ts'] = current_event_time
                    if e['sche_speed'] is not None: e['speed'] = e['sche_speed']
                    else: issues.append(f"[L{line_num} @ {current_event_time:.4f}] INTERNAL ERR: E{eid} SCHE-BEGIN sche_speed is None."); e['speed'] = DEFAULT_SPEED
                    if e['floor'] != e['sche_target_floor']: e['motion'] = 'MOVING'; e['last_floor_ts'] = current_event_time
                    else: e['motion'] = 'IDLE'
                    # Cancel receives
                    pids_to_cancel = [pid_c for pid_c, assigned_eid in received_requests.items() if assigned_eid == eid]
                    for pid_cancel in pids_to_cancel:
                         del received_requests[pid_cancel]
                         if pid_cancel in passengers and passengers[pid_cancel]['state'] == 'WAITING': passengers[pid_cancel]['assigned_elevator'] = None
                else: issues.append(f"[L{line_num} @ {current_event_time:.4f}] E{eid} failed SCHE-BEGIN checks, state remains {e['sche_phase']}")
                e['last_action_ts'] = current_event_time # Update action time

            # --- SCHE-END ---
            elif matched_action == 'END':
                eid_str, = action_data
                if e is None: continue
                valid_end = True # Assume valid unless a check fails
                if not e['sche_active'] or e['sche_phase'] != 'BEGUN': issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} END invalid state (active={e['sche_active']}, phase={e['sche_phase']})"); valid_end = False
                if e['floor'] != e['sche_target_floor']: issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} END at {e['floor']} != target {e['sche_target_floor']}"); valid_end = False
                if e['door'] != 'CLOSED': issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} END door not CLOSED ({e['door']})"); valid_end = False
                if e['passengers']: issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} END with passengers: {list(e['passengers'])}"); valid_end = False
                if e['floor'] == e['sche_target_floor'] and not e['sche_opened_at_target']:
                     issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE VIOLATION: E{eid} END at target {e['floor']} without OPENING door during SCHE.")
                     valid_end = False # Opening at target is mandatory

                # Check Timing
                if e['sche_accept_ts'] <= 0: issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE INTERNAL ERR: E{eid} END check bad accept_ts ({e['sche_accept_ts']})."); valid_end = False
                else: # Can check timing only if accept_ts is valid
                    completion_time = current_event_time - e['sche_accept_ts']
                    if completion_time > (SCHE_MAX_COMPLETE_TIME + TIMESTAMP_TOLERANCE):
                        issues.append(f"[L{line_num} @ {current_event_time:.4f}] SCHE TIMING VIOLATION: E{eid} completion time {completion_time:.4f}s > {SCHE_MAX_COMPLETE_TIME}s"); valid_end = False

                # Always reset SCHE state
                e['sche_active'] = False; e['sche_phase'] = None; e['sche_target_floor'] = None; e['sche_speed'] = None
                e['sche_accept_ts'] = -1.0; e['sche_begin_ts'] = -1.0; e['sche_end_ts'] = current_event_time
                e['sche_arrives_since_accept'] = 0; e['sche_opened_at_target'] = False
                e['speed'] = DEFAULT_SPEED; e['motion'] = 'IDLE'
                e['last_action_ts'] = current_event_time # Update action time


        except (ValueError, IndexError, TypeError, KeyError) as e_state:
             issues.append(f"[L{line_num} @ {current_time:.4f}] STATE/PARSE ERROR action '{matched_action}': {type(e_state).__name__}: {e_state} in: {raw_line}")
             issues.append(traceback.format_exc()) # Add traceback for debugging state errors

    # --- Final Checks ---
    final_time = current_time
    if final_time > T_MAX_HUTEST: issues.append(f"FINAL CHECK: Time Limit Exceeded: {final_time:.4f}s > {T_MAX_HUTEST}s.")
    # Check passengers
    for pid, p_state in passengers.items():
        if p_state['state'] != 'ARRIVED':
             issues.append(f"FINAL CHECK: P{pid} not ARRIVED (State: {p_state['state']}, Loc: {p_state['location']}, Dest: {p_state['destination']}).")
        elif p_state['location'] != p_state['destination']:
             # This check might be redundant if ARRIVED state implies correct location, but good for sanity.
             issues.append(f"FINAL CHECK: P{pid} ARRIVED but final location '{p_state['location']}' != destination '{p_state['destination']}'.")
    # Check elevators
    for eid, e_state in elevators.items():
        if e_state['passengers']: issues.append(f"FINAL CHECK: E{eid} finished with passengers: {list(e_state['passengers'])}")
        if e_state['door'] != 'CLOSED': issues.append(f"FINAL CHECK: E{eid} finished door {e_state['door']}")
        if e_state['sche_active']: issues.append(f"FINAL CHECK: E{eid} finished SCHE active (Phase: {e_state['sche_phase']})")
    # Check pending receives
    passengers_assigned_pending = [p_id for p_id, assigned_eid in received_requests.items() if assigned_eid is not None]
    if passengers_assigned_pending: issues.append(f"FINAL CHECK: Finished with pending RECEIVE assignments for Ps: {passengers_assigned_pending}")
    return issues


# --- Task Function for Thread Pool ---
# Corrected version fixing 'action_part' NameError and ignoring WARN for success/fail
def _run_one_batch_test_task(test_id, input_lines, jar_path, datainput_exe_path, stop_event_ref):
    """
    Runs a single test case (datainput | java -jar ...) in a temporary directory.
    Determines success based on Java exit code 0 AND no validation VIOLATIONS/ERRORS (ignores WARNs).

    Args:
        test_id (str): Unique identifier for this test run.
        input_lines (list): List of strings for the input data (stdin.txt).
        jar_path (str): Absolute path to the elevator JAR file.
        datainput_exe_path (str): Absolute path to the datainput executable.
        stop_event_ref (threading.Event): Event to signal early termination.

    Returns:
        tuple: (test_id, success, output_log_string, validation_issues, input_lines_ref)
               - success (bool): True if Java ran successfully (exit code 0) AND no validation errors/violations.
               - validation_issues (list): List of ALL strings from validator (including warnings).
               - input_lines_ref (list): Reference to the input lines used.
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
            if p_datainput.stdout: p_datainput.stdout.close()

            # Check immediate termination
            if p_java.poll() is not None:
                 try: stderr_java_str = p_java.stderr.read() if p_java.stderr else ""
                 except Exception: pass
                 raise RuntimeError(f"Java terminated immediately. RC={p_java.returncode}. Stderr: {stderr_java_str[:500]}")
            if p_java.stdout is None: raise RuntimeError("Java failed to redirect stdout.")

            # --- Read Java Output Line-by-Line ---
            try:
                # Read stdout line by line as it comes in
                for line in iter(p_java.stdout.readline, ''):
                    if stop_event_ref.is_set():
                        # Attempt graceful termination
                        if p_java.poll() is None: p_java.terminate()
                        if p_datainput.poll() is None: p_datainput.terminate()
                        raise InterruptedError("Batch stop requested during output reading")
                    line = line.strip()
                    if line: output_log_lines.append(line)
            except IOError as e: # Catch direct IO errors on the stream e.g. reading after killed
                output_log_lines.append(f"WARNING: IOError reading Java stdout (process likely died): {e}")
            # Do NOT explicitly close p_java.stdout here; let communicate() handle it.

            # --- Wait for Processes and Get Results ---

            # --- Handle Java Process ---
            try:
                 # communicate() reads remaining stdout/stderr and waits
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
            if p_datainput.poll() is None: # Only communicate if still running
                try:
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
            if return_code_datainput > 0: output_log_lines.append(f"WARNING: datainput.exe exited code {return_code_datainput}")
            if return_code_java > 0: output_log_lines.append(f"WARNING: Java process exited code {return_code_java}")

            # --- Perform Detailed Validation ---
            # Assuming validate_output function exists and is correct
            if execution_finished_without_error:
                validation_start_time = time.monotonic()
                # Ensure validate_output is defined and accessible in this scope
                validation_issues = validate_output(output_log_lines, input_lines)
                validation_duration = time.monotonic() - validation_start_time
                if validation_duration > 10.0: output_log_lines.append(f"--- Validation took {validation_duration:.2f} seconds ---")
            elif return_code_java == -99: validation_issues = [f"Java process timed out ({java_wait_timeout}s). Validation skipped."]
            else: validation_issues = [f"Java process communication error (Code: {return_code_java}). Validation skipped."]

            # --- Determine Overall Success (Ignoring Warnings) ---
            java_ok = (return_code_java == 0)
            actual_errors_violations = []
            # Define prefixes/keywords indicating non-failure states or allowed messages
            allowed_prefixes_to_ignore = ("WARN:", "DEBUG:", "INFO:", "INPUT WARNING:")
            allowed_non_core_prefixes = ('DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL', 'Result:', '---', '[', '#', 'at ', '\tat ')
            # Define keywords indicating definite failure
            error_keywords = ("VIOLATION:", "ERROR:", "FATAL", "SEMANTIC VIOLATION:", "FINAL CHECK:", "INTERNAL ERR:")

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
        for p_name, p in [("Java", p_java), ("DataInput", p_datainput)]:
            if p and p.poll() is None:
                try: p.kill(); p.wait(timeout=0.5)
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
        self.title("Elevator Tester - Enhanced Batch Mode")
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

        # --- Generation Parameters ---
        self.gen_passengers = tk.IntVar(value=20) # Default more passengers
        self.gen_schedules = tk.IntVar(value=5)  # Default more schedules
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

        # --- Post-init Checks (like wexpect, though unused currently) ---
        # self._check_dependencies() # Example if needed

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

        # --- Parameters Frame (Combined Generation & Batch) ---
        params_frame = ttk.Frame(self, padding="5")
        params_frame.pack(fill=tk.X, padx=5, pady=5)

        gen_group = ttk.LabelFrame(params_frame, text="Data Generation Parameters", padding="5")
        gen_group.pack(side=tk.LEFT, padx=(0, 10), fill=tk.Y)
        ttk.Label(gen_group, text="Passengers:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(gen_group, textvariable=self.gen_passengers, width=6).grid(row=0, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(gen_group, text="Schedules:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(gen_group, textvariable=self.gen_schedules, width=6).grid(row=1, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(gen_group, text="Max Time (s):").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(gen_group, textvariable=self.gen_max_time, width=6).grid(row=2, column=1, padx=5, pady=2, sticky="w")

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
                    idx = self.test_listbox.get(0, tk.END).index(loaded_names[0])
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
            del self.test_data_sets[self.current_test_name]
            removed_name = self.current_test_name
            self.current_test_name = None # Clear current selection state
            self._update_test_listbox() # Update list (will clear preview/disable button)
            self._log_output(f"Removed manual test: {removed_name}")


    # --- Batch Testing Methods ---
    def _validate_batch_inputs(self):
        """Validates all inputs required for batch testing. Returns True if valid, False otherwise."""
        jar_folder = self.jar_folder_path.get()
        datainput_exe = self.datainput_path.get()
        num_tests = self.batch_num_tests.get()
        concurrency = self.batch_concurrency.get()
        num_p = self.gen_passengers.get()
        num_s = self.gen_schedules.get()
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
            if not (5.0 <= max_t <= T_MAX_HUTEST + 120.0): raise ValueError(f"Max Time must be between 5.0 and {T_MAX_HUTEST + 120.0} seconds.")
            if num_p == 0 and num_s == 0: raise ValueError("Number of passengers or schedules must be greater than 0.")
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

    def _start_batch_test(self):
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
        num_p = self.gen_passengers.get(); num_s = self.gen_schedules.get(); max_t = self.gen_max_time.get()

        # Log start info
        self._clear_output_log()
        self._log_output("--- Starting New Batch Test ---")
        self._log_output(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._log_output(f"JAR Folder: {self.jar_folder_path.get()} ({len(jar_paths)} JARs)")
        self._log_output(f"DataInput EXE: {datainput_exe}")
        self._log_output(f"Tests per JAR: {num_tests}, Concurrency: {concurrency}")
        self._log_output(f"Gen Params: Passengers={num_p}, Schedules={num_s}, MaxTime={max_t:.1f}s")
        self._log_output(f"Failure logs will be saved under: {self.base_failures_dir}")
        self._log_output("-" * 30)

        # Update GUI state
        self.is_batch_running = True
        self.start_batch_button.config(state=tk.DISABLED)
        self.stop_batch_button.config(state=tk.NORMAL)
        self.run_manual_button.config(state=tk.DISABLED) # Disable manual run
        for widget in self.winfo_children(): # Disable parameter entry during run? Optional.
             if isinstance(widget, (ttk.Entry, ttk.Button)) and widget not in [self.stop_batch_button]:
                 pass # Consider disabling parameter entry widgets here if desired
        self.batch_progress_label.config(text="Status: Initializing...")
        self.current_jar_being_tested.set("Preparing...")
        self.stop_batch_event.clear() # Ensure stop flag is reset

        # Start the manager thread
        self.batch_thread = threading.Thread(
            target=self._run_batch_manager,
            args=(jar_paths, num_tests, concurrency, datainput_exe, num_p, num_s, max_t),
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
            # Optionally, start a timer to force-quit if it takes too long? More complex.
        else:
            # Thread already finished or wasn't running, reset state just in case
            self._log_output("Stop requested, but batch thread is not active. Resetting GUI.")
            self._reset_gui_after_batch()


    def _reset_gui_after_batch(self):
        """Resets GUI controls to idle state after batch finishes or stops."""
        self.is_batch_running = False
        self.start_batch_button.config(state=tk.NORMAL)
        self.stop_batch_button.config(state=tk.DISABLED)
        # Re-enable manual run button based on current selection state
        self._on_test_select()
        # Re-enable any parameter widgets if they were disabled
        # ... (add code here if parameter widgets were disabled) ...
        self.batch_thread = None
        # Don't clear stop event here, let start clear it.

    def _run_batch_manager(self, jar_paths, num_tests_per_jar, concurrency, datainput_exe_path, num_p, num_s, max_t):
        """
        Manages the batch execution: generates data, runs tests per JAR, handles stop signals. (Worker Thread)
        Sends updates to the GUI via the message_queue.
        """
        overall_summary = {} # { jar_name: {'passed': p, 'failed': f, 'skipped': s} }
        total_jars = len(jar_paths)
        start_batch_time = time.monotonic()

        try:
            # 1. Generate all test data upfront
            self.message_queue.put(("progress", f"Status: Generating {num_tests_per_jar} test cases..."))
            self.message_queue.put(("current_jar", "Generating Data..."))
            all_test_inputs = []
            gen_start_time = time.monotonic()
            for i in range(num_tests_per_jar):
                if self.stop_batch_event.is_set():
                    raise InterruptedError("Stop requested during test data generation")
                input_lines = generate_test_data(num_p, num_s, max_t)
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
                # Sanitize JAR name for directory creation (remove .jar, replace weird chars)
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
                            _run_one_batch_test_task,
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
                                  # Log cancellation immediately? Maybe too verbose. Summary later.
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
                            # Sanity check: _tid_res should match test_id
                            if _tid_res != test_id:
                                 self.message_queue.put(("log", f"WARNING: Result ID mismatch! Expected {test_id}, got {_tid_res}"))

                            # Log result and update counters
                            if success:
                                jar_passed += 1
                                log_msg = f"  Result [{jar_processed}/{len(all_futures)}]: {test_id} -> PASSED"
                                self.message_queue.put(("update_counters", (jar_processed, jar_passed, jar_failed))) # Update counters
                                self.message_queue.put(("log", log_msg)) # Log concise pass message
                            else:
                                jar_failed += 1
                                log_msg = f"  Result [{jar_processed}/{len(all_futures)}]: {test_id} -> FAILED"
                                # Add first few issues to the log message for quick glance
                                if validation_issues: log_msg += f" ({validation_issues[0][:80]}{'...' if len(validation_issues[0])>80 else ''})"
                                self.message_queue.put(("update_counters", (jar_processed, jar_passed, jar_failed))) # Update counters
                                self.message_queue.put(("log", log_msg)) # Log concise fail message

                                # Save failure artifacts
                                try:
                                    # Use safe test_id for filename
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
                                    # Optional: Log where details were saved? Can get verbose.
                                    # self.message_queue.put(("log", f"    (Details saved in {os.path.basename(jar_failure_dir)})"))

                                except Exception as e_save:
                                     err_save_msg = f"    ERROR saving failure details for {test_id}: {e_save}"
                                     self.message_queue.put(("log", err_save_msg))

                        except Exception as exc: # Handle error *getting* the result (e.g., task raised unexpected exception)
                            jar_failed += 1
                            err_msg = f"  Result [{jar_processed}/{len(all_futures)}]: {test_id} -> ERROR (Task Exception: {exc})"
                            self.message_queue.put(("update_counters", (jar_processed, jar_passed, jar_failed)))
                            self.message_queue.put(("log", err_msg))
                            # Log traceback for debugging the task error itself
                            tb_lines = traceback.format_exc().splitlines()
                            self.message_queue.put(("log", f"    Task Traceback:\n      " + "\n      ".join(tb_lines)))
                            # Optionally save a minimal error log here as well
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

                        # Update overall progress label (do this per result processed)
                        current_progress_text = f"Status [JAR {jar_index+1}]: Processed {jar_processed}/{num_tests_per_jar} (P:{jar_passed}, F:{jar_failed})"
                        if self.stop_batch_event.is_set(): current_progress_text += " [Stopping...]"
                        self.message_queue.put(("progress", current_progress_text))
                        # time.sleep(0.005) # Tiny sleep to allow GUI updates if needed, likely not necessary

                # End of tests for current JAR (after processing all completed futures)
                self.message_queue.put(("log", f"--- Finished JAR: {jar_name} - Passed: {jar_passed}, Failed: {jar_failed} ---"))
                overall_summary[jar_name] = {'passed': jar_passed, 'failed': jar_failed, 'skipped': num_tests_per_jar - jar_processed}

        except InterruptedError: # Catch stop during generation or between JARs
             self.message_queue.put(("progress", "Status: Batch stopped by user."))
             self.message_queue.put(("log", "--- Batch Run Interrupted ---"))
             # Summary will reflect completed JARs and potentially skipped ones if stopped between JARs
        except Exception as e_manager:
             # Error during generation or manager setup/loop
             self.message_queue.put(("log", f"--- FATAL BATCH MANAGER ERROR ---"))
             self.message_queue.put(("log", f"Error: {e_manager}"))
             self.message_queue.put(("log", traceback.format_exc()))
             self.message_queue.put(("progress", "Status: FATAL ERROR"))
             # Try to mark remaining JARs as skipped? Difficult if error was early.
        finally:
             # --- Batch Finished or Stopped ---
             total_duration = time.monotonic() - start_batch_time
             self.message_queue.put(("log", f"\n--- Batch Run Finished in {total_duration:.2f} seconds ---"))
             # Signal GUI that batch is finished, sending final summary
             self.message_queue.put(("finished", overall_summary))


        # Corrected version addressing the AttributeError on StringVar.winfo_exists()
    def _process_batch_queue(self):
        """Processes messages from the batch manager thread. Runs in GUI Thread via self.after()."""
        try:
            while True: # Process all available messages in the queue currently
                if not self.winfo_exists(): return # Stop if window closed during processing

                msg_type, data = self.message_queue.get_nowait()

                if msg_type == "progress":
                    if self.batch_progress_label.winfo_exists(): self.batch_progress_label.config(text=data)
                elif msg_type == "log":
                    self._log_output(data) # _log_output handles widget check and scheduling
                elif msg_type == "current_jar":
                    # FIX: Removed winfo_exists() check for StringVar
                    self.current_jar_being_tested.set(data)
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
                    # FIX: Use get() for StringVar value, removed winfo_exists()
                    # Safely get JAR name part (handle potential format issues)
                    current_jar_full = self.current_jar_being_tested.get()
                    jar_name_match = re.match(r"([^\s\(]+)", current_jar_full) # Match chars until space or parenthesis
                    jar_name_str = jar_name_match.group(1) if jar_name_match else current_jar_full # Fallback to full string

                    prog_text = f"Status [{jar_name_str}]: " \
                                f"Processed {data[0]}/{total} (P:{data[1]}, F:{data[2]})"
                    if self.stop_batch_event.is_set(): prog_text += " [Stopping...]"
                    if self.batch_progress_label.winfo_exists(): self.batch_progress_label.config(text=prog_text)

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

                    if self.batch_progress_label.winfo_exists():
                        self.batch_progress_label.config(text=f"Status: Finished ({total_passed_all}P, {total_failed_all}F, {total_skipped_all}S)")
                    # FIX: Removed winfo_exists() check for StringVar
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
                if self.batch_progress_label.winfo_exists(): self.batch_progress_label.config(text="Status: Error processing results")
                # FIX: Removed winfo_exists() check for StringVar
                self.current_jar_being_tested.set("Error")
                self._reset_gui_after_batch() # Reset buttons etc.

        # Reschedule check ONLY if batch is still marked as running
        if self.is_batch_running:
            if self.batch_thread and self.batch_thread.is_alive():
                self.after(150, self._process_batch_queue) # Check again after 150ms
            else:
                # Thread died unexpectedly without sending "finished" or stop incomplete
                if self.is_batch_running:
                    self._log_output("--- WARNING: Batch thread ended unexpectedly or stop incomplete ---")
                    if self.batch_progress_label.winfo_exists(): self.batch_progress_label.config(text="Status: Ended Unexpectedly/Error")
                    # FIX: Removed winfo_exists() check for StringVar
                    self.current_jar_being_tested.set("Error/Ended")
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
        self.run_manual_button.config(state=tk.DISABLED)
        self.start_batch_button.config(state=tk.DISABLED) # Disable batch during manual run
        self.current_jar_being_tested.set(f"Manual: {os.path.basename(manual_jar_path)}")
        self.batch_progress_label.config(text="Status: Running Manual Test...")

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


        # Corrected version addressing the AttributeError on StringVar.winfo_exists()
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
                            if self.batch_progress_label.winfo_exists(): self.batch_progress_label.config(text="Status: Manual Test PASSED")
                        else:
                            self._log_output(f"Result: FAILED")
                            if self.batch_progress_label.winfo_exists(): self.batch_progress_label.config(text="Status: Manual Test FAILED")
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
                                # FIX: Use get() to retrieve value from StringVar for logging
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
                        log_chunk_size = 200
                        for i in range(0, len(log_lines_split), log_chunk_size):
                            chunk = "\n".join(log_lines_split[i:i+log_chunk_size])
                            self.after_idle(self._log_output, chunk)
                    else:
                        # Result tuple was None (error during task call)
                        self._log_output("--- Manual test finished with internal execution error (see logs above) ---")
                        if self.batch_progress_label.winfo_exists(): self.batch_progress_label.config(text="Status: Manual Test ERROR")

                    # --- Reset GUI state after manual run ---
                    if not self.is_batch_running:
                        self.start_batch_button.config(state=tk.NORMAL)
                        current_progress_text = self.batch_progress_label['text'] if self.batch_progress_label.winfo_exists() else ""
                        if self.batch_progress_label.winfo_exists() and "Manual Test" in current_progress_text:
                            # Only reset status if it was showing Manual Test status
                            if "PASSED" in current_progress_text:
                                self.batch_progress_label.config(text="Status: Idle")
                            # Otherwise leave FAILED/ERROR message displayed
                        # FIX: Removed winfo_exists() check for StringVar
                        self.current_jar_being_tested.set("N/A")
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
                self.start_batch_button.config(state=tk.NORMAL)
                if self.batch_progress_label.winfo_exists(): self.batch_progress_label.config(text="Status: Error processing results")
                # FIX: Removed winfo_exists() check for StringVar
                self.current_jar_being_tested.set("Error")
                self._on_test_select()

        # Reschedule check ONLY if manual run is still indicated as running
        if self.winfo_exists() and self.run_manual_button.winfo_exists() and self.run_manual_button['state'] == tk.DISABLED and not self.is_batch_running:
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
            # Check if window exists before scheduling
             if hasattr(self, 'output_log_text') and self.output_log_text.winfo_exists():
                  self.after_idle(self._log_output, message)
             # else: print(f"DEBUG: Window closed, discarding log: {message}") # Debug log discard
             return

        # --- This part runs only in the GUI thread ---
        if not hasattr(self, 'output_log_text') or not self.output_log_text.winfo_exists(): return
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
                self.stop_batch_button.config(state=tk.DISABLED) # Prevent further clicks
                # Give manager thread a chance to react? It should see the event soon.
                # We don't explicitly wait here, just signal and proceed to destroy.
                # Daemon thread will be terminated on exit anyway if it hangs.
            else:
                quit_app = False # User cancelled exit

        # Check if manual test *might* be running (indicated by disabled button when batch is NOT running)
        elif self.run_manual_button.winfo_exists() and self.run_manual_button['state'] == tk.DISABLED:
             if not messagebox.askyesno("Exit Confirmation", "A manual test might be running in the background (no stop mechanism).\n\nExit anyway?", parent=self):
                  quit_app = False

        if quit_app:
             self._save_config() # Save paths on exit
             # Cleanly destroy the window and let Tkinter handle main loop exit
             self.destroy()

# --- Main Execution ---
# Corrected version addressing the font initialization warning.
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
        # You might need to apply these fonts explicitly to widgets if default override doesn't work
        # e.g., app.output_log_text.config(font=text_font) inside _create_widgets
        print("Note: Default fonts configured.") # Optional confirmation
    except Exception as e:
        # This might still fail if font families aren't available, but it won't be 'too early'
        print(f"Note: Could not configure default fonts - {e}", file=sys.stderr)

    # Start the Tkinter event loop
    app.mainloop()
