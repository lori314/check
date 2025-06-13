import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import queue # Still used for GUI updates from manager thread
import time
import random
import os
import json
import re
import sys
import subprocess
import tempfile # For temporary directories
import shutil # For copying failed stdin.txt
from concurrent.futures import ThreadPoolExecutor, Future, as_completed # For concurrency

# --- Platform Specific Import ---
if sys.platform == 'win32':
    try: import wexpect as pexpect
    except ImportError: messagebox.showerror("Missing Dependency", "'wexpect' required. pip install wexpect"); sys.exit(1)
else: messagebox.showerror("Unsupported OS", "This script targets Windows."); sys.exit(1)


# --- Configuration ---
FLOORS = [f"B{i}" for i in range(4, 0, -1)] + [f"F{i}" for i in range(1, 8)]
ELEVATOR_IDS = list(range(1, 7)); PRIORITY_RANGE = (1, 20); SCHEDULE_SPEEDS = [0.2, 0.3, 0.4, 0.5]
PROCESS_ENCODING = 'utf-8'; STDIN_FILENAME = "stdin.txt"
FAILURES_DIR = "batch_failures"
DEFAULT_CONCURRENCY = 8

# --- Timestamp Helper ---
def parse_output_timestamp(line):
    match = re.match(r'^\s*\[\s*(\d+\.\d+)\s*\]', line)
    if match:
        try: return float(match.group(1))
        except ValueError: return None
    return None

# --- Data Generation (Ensure Schedules for Same Elevator are Spaced Out) ---
def generate_test_data(num_passengers, num_schedules, max_time):
    requests_timed = []
    used_passenger_ids = set()
    # Keep track of the estimated finish time of the last schedule for each elevator
    elevator_last_sche_finish_time = {} # {eid: estimated_finish_time}

    ALLOWED_FLOORS = ['B2', 'B1', 'F1', 'F2', 'F3', 'F4', 'F5']
    ALL_SYSTEM_FLOORS = [f"B{i}" for i in range(4, 0, -1)] + [f"F{i}" for i in range(1, 8)]
    AVAILABLE_ELEVATOR_IDS = list(range(1, 7)) # Use all elevators

    # Generate passenger requests (Ensuring destination is allowed)
    for i in range(num_passengers):
        pid = random.randint(1, 9999)
        while pid in used_passenger_ids: pid = random.randint(1, 9999)
        used_passenger_ids.add(pid)
        from_fl = random.choice(ALL_SYSTEM_FLOORS)
        to_fl = random.choice(ALLOWED_FLOORS)
        while from_fl == to_fl:
            to_fl = random.choice(ALLOWED_FLOORS)
            if from_fl != to_fl: break
        priority = random.randint(PRIORITY_RANGE[0], PRIORITY_RANGE[1])
        send_time = round(random.uniform(0.5, max_time - 1.0), 4)
        cmd = f"{pid}-PRI-{priority}-FROM-{from_fl}-TO-{to_fl}"
        requests_timed.append((send_time, cmd))

    # Generate schedule requests (spaced out per elevator)
    MAX_FLOOR_DIFF = 10 # Max possible floors to travel (B4 to F7)
    DOOR_CYCLE_TIME = 1.8 # open(0.4) + stop(1.0) + close(0.4)
    INTER_SCHEDULE_BUFFER = 0.1 # Add a small buffer between schedules

    for _ in range(num_schedules):
        # Randomly choose an elevator for this schedule
        eid = random.choice(AVAILABLE_ELEVATOR_IDS)
        speed = random.choice(SCHEDULE_SPEEDS)
        target_fl = random.choice(ALLOWED_FLOORS)

        # Generate an initial random send time
        send_time = round(random.uniform(1.0, max_time), 4)

        # Check if this elevator has a previous schedule
        last_finish_time = elevator_last_sche_finish_time.get(eid, 0.0) # Default to 0 if no previous

        # Ensure the new send_time is after the previous estimated finish time + buffer
        required_start_time = last_finish_time + INTER_SCHEDULE_BUFFER
        if send_time < required_start_time:
            # Adjust send_time if it's too early. Add some randomness to avoid piling up.
            send_time = round(random.uniform(required_start_time, required_start_time + 1.0), 4)
            # Clamp send_time if it exceeds max_time due to adjustment
            if send_time >= max_time:
                 # Option 1: Skip this schedule if it goes too late
                 # continue
                 # Option 2: Clamp it just before max_time (might make last requests unrealistic)
                 send_time = round(max_time - random.uniform(0.1, 0.5), 4)
                 if send_time <= last_finish_time: # Still too early after clamping? Skip.
                     continue

        # Calculate the estimated finish time for *this* schedule
        # Use worst-case travel time for conservative estimate
        estimated_travel_time = MAX_FLOOR_DIFF * speed
        estimated_finish_time = send_time + estimated_travel_time + DOOR_CYCLE_TIME

        # Update the last finish time for this elevator
        elevator_last_sche_finish_time[eid] = estimated_finish_time

        # Add the schedule request
        cmd = f"SCHE-{eid}-{speed}-{target_fl}"
        requests_timed.append((send_time, cmd))

    # Sort all requests by time
    requests_timed.sort(key=lambda x: x[0])
    formatted_lines = [f"[{t:.4f}]{cmd}" for t, cmd in requests_timed]
    return formatted_lines

# --- VALIDATION FUNCTION (REVISED LOGIC) ---
# --- VALIDATION FUNCTION (REVISED LOGIC AGAIN) ---
# --- VALIDATION FUNCTION (REVISED LOGIC YET AGAIN) ---
# --- VALIDATION FUNCTION (WITH DEBUGGING and Refined ID Parsing) ---
# --- VALIDATION FUNCTION (Corrected END Logic) ---
# --- VALIDATION FUNCTION (Corrected END Logic - Final Version) ---
# --- VALIDATION FUNCTION (SIMPLIFIED - Focus on Core Rules) ---
def validate_output(output_log_lines):
    """
    Performs simplified validation:
    1. Checks for timestamp decrease.
    2. Checks the "at most 2 ARRIVEs between SCHE-ACCEPT and SCHE-BEGIN" rule.
    Returns list of issue strings.
    """
    issues = []
    last_parsed_ts_overall = -1.0
    # Track state per elevator: {eid: {'accept_received': bool, 'arrive_count': int}}
    elevator_schedule_state = {}

    # Patterns focused on the rule
    pattern_arrive = re.compile(r"ARRIVE-(?:[A-Z0-9]+)-(\d+)$")
    pattern_accept = re.compile(r"SCHE-ACCEPT-(\d+)-.*")
    pattern_begin = re.compile(r"SCHE-BEGIN-(\d+)$")

    for line_num, raw_line in enumerate(output_log_lines, 1):
        line_content = raw_line.strip()
        parsed_ts = parse_output_timestamp(line_content)

        # 1. --- Overall Timestamp Check ---
        if parsed_ts is not None:
            if parsed_ts < last_parsed_ts_overall:
                issues.append(f"Timestamp decreased (Line {line_num}): {parsed_ts} after {last_parsed_ts_overall} in line: {raw_line}")
            last_parsed_ts_overall = parsed_ts
        # Allow malformed timestamps or other lines without failing validation here
        # else: # Optional: Add warning for malformed lines
        #    if line_content and not line_content.startswith('[') and line_content != 'null':
        #        issues.append(f"Non-timestamp line detected (Line {line_num}): {raw_line}")
        #    elif line_content and line_content != 'null':
        #        issues.append(f"Malformed timestamp? (Line {line_num}): {raw_line}")


        # 2. --- Check SCHE ARRIVE Rule ---
        action_part = line_content.split(']')[-1].strip()

        # Is it an ACCEPT line?
        match_accept = pattern_accept.match(action_part)
        if match_accept:
            try:
                elevator_id = int(match_accept.group(1))
                if 1 <= elevator_id <= 99:
                    # Record that ACCEPT was received for this elevator and reset count
                    elevator_schedule_state[elevator_id] = {'accept_received': True, 'arrive_count': 0}
                    # print(f"DEBUG (Line {line_num}): ACCEPT for {elevator_id}, State={elevator_schedule_state[elevator_id]}") # Optional Debug
            except (ValueError, IndexError): pass
            continue # Process next line after handling ACCEPT

        # Is it an ARRIVE line?
        match_arrive = pattern_arrive.match(action_part)
        if match_arrive:
            try:
                elevator_id = int(match_arrive.group(1)) # Corrected: Group 1 for ARRIVE ID
                if 1 <= elevator_id <= 99:
                    # Check if this elevator has a pending ACCEPT
                    if elevator_id in elevator_schedule_state and elevator_schedule_state[elevator_id]['accept_received']:
                        # Increment arrive count
                        elevator_schedule_state[elevator_id]['arrive_count'] += 1
                        # print(f"DEBUG (Line {line_num}): ARRIVE for {elevator_id}, Count={elevator_schedule_state[elevator_id]['arrive_count']}") # Optional Debug
                        # Check for violation *immediately*
                        if elevator_schedule_state[elevator_id]['arrive_count'] > 2:
                            issues.append(f"VIOLATION (Line {line_num}): Elevator {elevator_id} ARRIVED {elevator_schedule_state[elevator_id]['arrive_count']} times after SCHE-ACCEPT before SCHE-BEGIN. Line: {raw_line}")
                            # Optional: Reset state after violation to prevent repeated errors for the same SCHE?
                            # elevator_schedule_state[elevator_id]['accept_received'] = False
            except (ValueError, IndexError): pass
            continue # Process next line after handling ARRIVE

        # Is it a BEGIN line?
        match_begin = pattern_begin.match(action_part)
        if match_begin:
            try:
                elevator_id = int(match_begin.group(1))
                if 1 <= elevator_id <= 99:
                    # If BEGIN is received, clear the tracking state for this elevator
                    if elevator_id in elevator_schedule_state:
                         # print(f"DEBUG (Line {line_num}): BEGIN for {elevator_id}, Clearing state.") # Optional Debug
                         # Check for the warning condition (BEGIN without ACCEPT) just before clearing
                         if not elevator_schedule_state[elevator_id]['accept_received']:
                              issues.append(f"WARNING (Line {line_num}): Elevator {elevator_id} issued SCHE-BEGIN, but no preceding SCHE-ACCEPT was tracked correctly by validator. Line: {raw_line}")
                         # Clear state after BEGIN
                         elevator_schedule_state[elevator_id] = {'accept_received': False, 'arrive_count': 0}
                    # else: # Optional: Warning if BEGIN received for an unknown elevator?
                    #    issues.append(f"WARNING (Line {line_num}): Elevator {elevator_id} issued SCHE-BEGIN, but was not previously tracked. Line: {raw_line}")

            except (ValueError, IndexError): pass
            continue # Process next line after handling BEGIN

    # --- Final Checks ---
    # Check if any elevator ended with accept_received still True (never began)
    for eid, state in elevator_schedule_state.items():
        if state['accept_received']:
             issues.append(f"WARNING: Elevator {eid} received SCHE-ACCEPT but test ended before SCHE-BEGIN was issued.")

    return issues


# --- Task Function for Thread Pool ---
def _run_one_batch_test_task(test_id, input_lines, jar_path, datainput_exe_path, stop_event_ref):
    """
    Runs a single test using subprocess in a temporary directory.
    Returns tuple: (test_id, success, output_log_string, validation_issues, input_lines_ref)
    """
    output_log_lines = []
    success = False
    validation_issues = ["Task did not complete successfully."]
    return_code_java = -1
    return_code_datainput = -1
    p_datainput = None
    p_java = None
    execution_finished_without_error = False

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            stdin_path = os.path.join(temp_dir, STDIN_FILENAME)
            with open(stdin_path, 'w', encoding='utf-8') as f:
                for line in input_lines: f.write(line + '\n')

            datainput_cmd = [datainput_exe_path]
            java_cmd = ['java', '-jar', jar_path]

            p_datainput = subprocess.Popen(datainput_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=temp_dir, text=True, encoding=PROCESS_ENCODING, creationflags=subprocess.CREATE_NO_WINDOW)
            p_java = subprocess.Popen(java_cmd, stdin=p_datainput.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=temp_dir, text=True, encoding=PROCESS_ENCODING, creationflags=subprocess.CREATE_NO_WINDOW)
            p_datainput.stdout.close()

            for line in iter(p_java.stdout.readline, ''):
                if stop_event_ref.is_set(): raise InterruptedError("Batch stop requested")
                line = line.strip()
                if line: output_log_lines.append(line)

            stderr_java_str = ""; stderr_data_str = ""
            try:
                p_java.wait(timeout=25)
                return_code_java = p_java.returncode
                stderr_java_str = p_java.stderr.read().strip()
                p_datainput.wait(timeout=5)
                return_code_datainput = p_datainput.returncode
                stderr_data_str = p_datainput.stderr.read().strip()
            except subprocess.TimeoutExpired:
                 output_log_lines.append("ERROR: Timeout waiting for processes to finish.")
                 return_code_java = p_java.poll(); return_code_datainput = p_datainput.poll()
                 raise

            if stderr_data_str: output_log_lines.append(f"ERROR (datainput): {stderr_data_str}")
            if stderr_java_str: output_log_lines.append(f"ERROR (java): {stderr_java_str}")
            if return_code_datainput != 0: output_log_lines.append(f"WARNING: datainput.exe exited with code {return_code_datainput}")

            # --- Perform Detailed Validation ---
            validation_issues = validate_output(output_log_lines) # Call the detailed validator

            if return_code_java == 0 and not validation_issues:
                success = True
                output_log_lines.append(f"-- Java process finished: Code {return_code_java} (Validation PASSED) --")
            elif return_code_java == 0 and validation_issues:
                 success = False
                 output_log_lines.append(f"-- Java process finished: Code {return_code_java} (Validation FAILED) --")
            else:
                success = False
                if not any(f"exit code ({return_code_java})" in issue for issue in validation_issues):
                     validation_issues.append(f"Java process returned non-zero exit code ({return_code_java}).")
                output_log_lines.append(f"-- Java process finished: Code {return_code_java} (FAILED) --")

            execution_finished_without_error = True

    # --- Outer Exception Handlers ---
    except InterruptedError: # Catch custom stop signal
        success = False
        validation_issues = ["Batch run stopped by user."]
        # Message already added where raised
    except FileNotFoundError as e:
        output_log_lines.append(f"FATAL ERROR: Command not found - {e}")
        success = False
        validation_issues = [f"File not found: {e}"]
    except subprocess.TimeoutExpired: # Catch re-raised timeout
        # Message already added where caught initially
        success = False
        validation_issues = ["Timeout expired during process wait."]
    except Exception as e: # Catch other general errors
        output_log_lines.append(f"FATAL PYTHON ERROR during test execution: {e}")
        import traceback
        output_log_lines.append(traceback.format_exc())
        success = False
        validation_issues = ["Unexpected Python error during test."]

    # --- Finally block ALWAYS runs for cleanup ---
    finally:
        # Ensure processes are terminated (best effort cleanup)
        for p in [p_datainput, p_java]:
            if p and p.poll() is None:  # Check if process object exists AND is still running
                try:
                    p.kill()  # Use kill directly for faster cleanup in finally
                except Exception:
                    pass  # Ignore errors during final cleanup kill
        # Temporary directory is automatically cleaned up by 'with' statement exit

    # --- Determine final validation message AFTER finally block ---
    # This code needs to be OUTSIDE the finally block, at the same indentation level as try/except/finally
    if not execution_finished_without_error:
        # If the task failed before reaching validation (e.g., timeout, file not found, python error)
        # AND the validation_issues list still holds the default message, provide a better generic one.
        if validation_issues == ["Task did not complete successfully."]:
             validation_issues = ["Task failed during execution (check logs for errors)."]
        # If an except block *did* set a message, validation_issues will already be updated.

    # Return results - This line is also OUTSIDE the finally block
    return (test_id, success, "\n".join(output_log_lines), validation_issues, input_lines)

# --- Main Application Class ---
# ...(Rest of the code follows)...


# --- Main Application Class (Modified GUI and Logic) ---
class ElevatorTesterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Elevator Tester - Batch Mode")
        self.geometry("1000x750") # Wider

        # File Paths
        self.jar_path = tk.StringVar()
        self.datainput_path = tk.StringVar()

        # Test Data Storage (for manual tests)
        self.test_data_sets = {}
        self.current_test_name = None

        # Batch Processing State
        self.batch_thread = None
        self.stop_batch_event = threading.Event()
        self.message_queue = queue.Queue() # For progress updates from manager thread
        self.is_batch_running = False

        # Generation Parameters
        self.gen_passengers = tk.IntVar(value=10)
        self.gen_schedules = tk.IntVar(value=1)
        self.gen_max_time = tk.DoubleVar(value=30.0)
        self.batch_num_tests = tk.IntVar(value=10)
        self.batch_concurrency = tk.IntVar(value=DEFAULT_CONCURRENCY)

        self._create_widgets()
        self.protocol("WM_DELETE_WINDOW", self._quit)

    def _create_widgets(self):
        # --- Top Frame: File Selections ---
        top_frame = ttk.Frame(self, padding="5")
        top_frame.pack(fill=tk.X, side=tk.TOP, pady=(0, 5))
        jar_frame = ttk.Frame(top_frame)
        jar_frame.pack(fill=tk.X)
        ttk.Label(jar_frame, text="Elevator JAR:", width=15, anchor="w").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(jar_frame, textvariable=self.jar_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(jar_frame, text="Browse...", command=self._select_jar).pack(side=tk.LEFT, padx=(5, 0))
        datainput_frame = ttk.Frame(top_frame)
        datainput_frame.pack(fill=tk.X, pady=(5,0))
        ttk.Label(datainput_frame, text="datainput EXE:", width=15, anchor="w").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(datainput_frame, textvariable=self.datainput_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(datainput_frame, text="Browse...", command=self._select_datainput).pack(side=tk.LEFT, padx=(5, 0))

        # --- Generation Parameters Frame ---
        gen_frame = ttk.LabelFrame(self, text="Data Generation Parameters", padding="5")
        gen_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(gen_frame, text="Passengers:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(gen_frame, textvariable=self.gen_passengers, width=7).grid(row=0, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(gen_frame, text="Schedules:").grid(row=0, column=2, padx=5, pady=2, sticky="w")
        ttk.Entry(gen_frame, textvariable=self.gen_schedules, width=7).grid(row=0, column=3, padx=5, pady=2, sticky="w")
        ttk.Label(gen_frame, text="Max Time (s):").grid(row=0, column=4, padx=5, pady=2, sticky="w")
        ttk.Entry(gen_frame, textvariable=self.gen_max_time, width=7).grid(row=0, column=5, padx=5, pady=2, sticky="w")

        # --- Batch Controls Frame ---
        batch_frame = ttk.LabelFrame(self, text="Batch Testing", padding="5")
        batch_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(batch_frame, text="Number of Tests:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(batch_frame, textvariable=self.batch_num_tests, width=7).grid(row=0, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(batch_frame, text="Concurrency:").grid(row=0, column=2, padx=5, pady=2, sticky="w")
        ttk.Entry(batch_frame, textvariable=self.batch_concurrency, width=7).grid(row=0, column=3, padx=5, pady=2, sticky="w")
        self.start_batch_button = ttk.Button(batch_frame, text="Start Batch Test", command=self._start_batch_test)
        self.start_batch_button.grid(row=0, column=4, padx=10, pady=2)
        self.stop_batch_button = ttk.Button(batch_frame, text="Stop Batch Test", command=self._stop_batch_test, state=tk.DISABLED)
        self.stop_batch_button.grid(row=0, column=5, padx=5, pady=2)
        self.batch_progress_label = ttk.Label(batch_frame, text="Status: Idle")
        self.batch_progress_label.grid(row=1, column=0, columnspan=6, padx=5, pady=5, sticky="w")

        # --- Main Area: Paned Window ---
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Left Pane: Manual Test Management ---
        left_frame = ttk.Frame(main_pane, padding="5")
        main_pane.add(left_frame, weight=1)
        ttk.Label(left_frame, text="Manual Test Cases:", font="-weight bold").pack(anchor=tk.W)
        self.test_listbox = tk.Listbox(left_frame, height=6)
        self.test_listbox.pack(fill=tk.BOTH, expand=True, pady=(5, 5))
        self.test_listbox.bind('<<ListboxSelect>>', self._on_test_select)
        test_button_frame = ttk.Frame(left_frame)
        test_button_frame.pack(fill=tk.X)
        ttk.Button(test_button_frame, text="Load Manual...", command=self._load_tests).pack(side=tk.LEFT, padx=2)
        ttk.Button(test_button_frame, text="Save Selected...", command=self._save_selected_test).pack(side=tk.LEFT, padx=2)
        ttk.Button(test_button_frame, text="Remove Selected", command=self._remove_selected_test).pack(side=tk.LEFT, padx=2)
        ttk.Label(left_frame, text="Selected Input Preview:", font="-weight bold").pack(anchor=tk.W, pady=(10, 0))
        self.input_text = scrolledtext.ScrolledText(left_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.input_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        self.run_manual_button = ttk.Button(left_frame, text="Run Selected Manual Test", command=self._run_manual_selected_test)
        self.run_manual_button.pack(pady=5)

        # --- Right Pane: Output Log ---
        right_frame = ttk.Frame(main_pane, padding="5")
        main_pane.add(right_frame, weight=3)
        self.output_log_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.output_log_text.pack(fill=tk.BOTH, expand=True)


    # --- GUI Logic Methods ---
    # Methods for managing the manual test list (CORRECTED FORMATTING)

    def _select_jar(self):
        path = filedialog.askopenfilename(
            title="Select Elevator JAR File",
            filetypes=[("JAR files", "*.jar"), ("All files", "*.*")]
        )
        if path:
            self.jar_path.set(path)

    def _select_datainput(self):
        path = filedialog.askopenfilename(
            title="Select datainput Executable",
            filetypes=[("Executable files", "*.exe"), ("All files", "*.*")]
        )
        if path:
            self.datainput_path.set(path)

    def _update_test_listbox(self):
        self.test_listbox.delete(0, tk.END)
        # Sort keys for consistent order in listbox
        for name in sorted(self.test_data_sets.keys()):
            self.test_listbox.insert(tk.END, name)

    def _on_test_select(self, event=None):
        selected_indices = self.test_listbox.curselection()
        if not selected_indices:
            self.current_test_name = None
            self._display_input_data([]) # Clear display
            return
        # Get selected name
        self.current_test_name = self.test_listbox.get(selected_indices[0])
        # Display data if found
        if self.current_test_name in self.test_data_sets:
            self._display_input_data(self.test_data_sets[self.current_test_name])
        else:
             # Should not happen if listbox is synchronized, but handle defensively
             self._display_input_data([])

    def _display_input_data(self, data_lines):
        self.input_text.config(state=tk.NORMAL)
        self.input_text.delete('1.0', tk.END)
        if data_lines:
            # Format the display string nicely
            display_str = f"# Test: {self.current_test_name}\n"
            display_str += f"# {len(data_lines)} lines for {STDIN_FILENAME}\n\n"
            display_str += "\n".join(data_lines)
            self.input_text.insert('1.0', display_str)
        self.input_text.config(state=tk.DISABLED)

    def _load_tests(self):
        filepath = filedialog.askopenfilename(
            title="Load Manual Test Data (stdin format)",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath:
            return
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            # Basic validation of loaded structure
            if not isinstance(loaded_data, dict):
                raise ValueError("Expected a dictionary (test_name -> list of strings)")

            count_loaded = 0
            count_skipped = 0
            for name, data in loaded_data.items():
                # Validate data format for each test
                if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
                   # Log warning using the queue method, safer in GUI context
                   self.message_queue.put(("log", f"WARNING: Skipping test '{name}' due to invalid data format."))
                   count_skipped += 1
                   continue

                # Ask before overwriting existing tests
                if name in self.test_data_sets:
                    if not messagebox.askyesno("Confirm Overwrite", f"Manual test set '{name}' already exists. Overwrite?", parent=self):
                        count_skipped += 1
                        continue # Skip this one if user selects No
                # Add or overwrite test data
                self.test_data_sets[name] = data
                count_loaded += 1

            self._update_test_listbox() # Update listbox after processing all tests in file
            messagebox.showinfo("Load Complete", f"Loaded {count_loaded} test(s), skipped {count_skipped} from {os.path.basename(filepath)}", parent=self)

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load test data:\n{e}", parent=self)

    def _save_selected_test(self):
        if not self.current_test_name or self.current_test_name not in self.test_data_sets:
             messagebox.showwarning("Warning", "No manual test case selected to save.", parent=self)
             return

        filepath = filedialog.asksaveasfilename(
            title="Save Selected Manual Test Data (stdin format)",
            defaultextension=".json",
            initialfile=f"{self.current_test_name}.json", # Suggest filename based on test name
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath:
            return

        # Data to save is just the dictionary entry for the selected test
        data_to_save = {self.current_test_name: self.test_data_sets[self.current_test_name]}
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2) # Use indent for readable JSON
            messagebox.showinfo("Success", f"Saved test '{self.current_test_name}' to {os.path.basename(filepath)}", parent=self)
        except Exception as e:
             messagebox.showerror("Save Error", f"Failed to save test data:\n{e}", parent=self)

    def _remove_selected_test(self):
         if not self.current_test_name or self.current_test_name not in self.test_data_sets:
             messagebox.showwarning("Warning", "No manual test case selected to remove.", parent=self)
             return

         if messagebox.askyesno("Confirm Delete", f"Are you sure you want to remove manual test set '{self.current_test_name}'?", parent=self):
            del self.test_data_sets[self.current_test_name]
            self.current_test_name = None
            self._update_test_listbox()
            self._display_input_data([]) # Clear the preview display

    # --- Batch Testing Methods ---

    def _start_batch_test(self):
        # Validate paths and parameters
        jar = self.jar_path.get()
        datainput_exe = self.datainput_path.get()
        num_tests = self.batch_num_tests.get()
        concurrency = self.batch_concurrency.get()
        num_p = self.gen_passengers.get()
        num_s = self.gen_schedules.get()
        max_t = self.gen_max_time.get()

        # Input validation
        if not jar or not os.path.exists(jar):
            messagebox.showerror("Error", "Please select a valid Elevator JAR file.", parent=self)
            return
        if not datainput_exe or not os.path.exists(datainput_exe):
            messagebox.showerror("Error", "Please select the datainput executable.", parent=self)
            return
        if not (1 <= num_tests <= 10000): # Example limit
            messagebox.showerror("Error", "Number of tests must be between 1 and 10000.", parent=self)
            return
        if not (1 <= concurrency <= 32): # Example limit
            messagebox.showerror("Error", "Concurrency must be between 1 and 32.", parent=self)
            return
        # Add validation for generation params if needed

        # UI updates and logging
        self._clear_output_log()
        self._log_output("--- Starting Batch Test ---")
        self._log_output(f"JAR: {jar}")
        self._log_output(f"DataInput EXE: {datainput_exe}")
        self._log_output(f"Number of tests: {num_tests}, Concurrency: {concurrency}")
        self._log_output(f"Generation Params: Passengers={num_p}, Schedules={num_s}, MaxTime={max_t}s")

        # Ensure failures directory exists
        try:
            os.makedirs(FAILURES_DIR, exist_ok=True)
            self._log_output(f"Failed tests will be saved to: {os.path.abspath(FAILURES_DIR)}")
        except OSError as e:
            messagebox.showerror("Directory Error", f"Could not create failures directory '{FAILURES_DIR}':\n{e}", parent=self)
            return


        # Set state and disable controls
        self.is_batch_running = True
        self.start_batch_button.config(state=tk.DISABLED)
        self.stop_batch_button.config(state=tk.NORMAL)
        self.run_manual_button.config(state=tk.DISABLED)
        self.batch_progress_label.config(text="Status: Generating data...")
        self.stop_batch_event.clear()

        # Start the batch manager thread
        self.batch_thread = threading.Thread(
            target=self._run_batch_thread,
            args=(num_tests, concurrency, jar, datainput_exe, num_p, num_s, max_t),
            daemon=True
        )
        self.batch_thread.start()
        self.after(100, self._process_batch_queue) # Start checking queue

    def _stop_batch_test(self):
        if self.is_batch_running and self.batch_thread and self.batch_thread.is_alive():
            # Update GUI immediately
            self.message_queue.put(("progress", "Status: Stopping... (Finishing active tests)"))
            # Signal the thread
            self.stop_batch_event.set()
            # Disable button to prevent multiple clicks
            self.stop_batch_button.config(state=tk.DISABLED)
            # The manager thread handles final state reset via the queue

    def _run_batch_thread(self, num_tests, concurrency, jar_path, datainput_exe_path, num_p, num_s, max_t):
        # (Manager thread using ThreadPoolExecutor - Corrected logic)
        total_passed = 0
        total_failed = 0
        all_futures = []

        try:
            # 1. Generate all test data upfront
            all_test_inputs = []
            for i in range(num_tests):
                # Check stop event frequently during potentially long generation
                if self.stop_batch_event.is_set():
                    raise InterruptedError("Stop during generation")
                input_lines = generate_test_data(num_p, num_s, max_t)
                all_test_inputs.append(input_lines)
                # Update progress via queue periodically
                if (i + 1) % 10 == 0 or i == num_tests - 1:
                     self.message_queue.put_nowait(("progress", f"Status: Generated {i+1}/{num_tests} test cases..."))

            self.message_queue.put_nowait(("progress", f"Status: Data generation complete. Submitting {num_tests} tests..."))

            # 2. Run tests using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                # Submit all tasks
                for i in range(num_tests):
                    # Check stop event before submitting each task
                    if self.stop_batch_event.is_set():
                        self.message_queue.put_nowait(("progress", "Status: Stop requested. No more tests submitted."))
                        break # Stop submitting new tasks

                    test_id = f"BatchTest_{i+1:04d}" # e.g., BatchTest_0001
                    future = executor.submit(
                        _run_one_batch_test_task,
                        test_id,
                        all_test_inputs[i], # Get pre-generated data
                        jar_path,
                        datainput_exe_path,
                        self.stop_batch_event # Pass the shared event object
                    )
                    all_futures.append(future)

                # 3. Process results as they complete
                completed_count = 0
                for future in as_completed(all_futures):
                    # We still process results even if stop was requested,
                    # to log results of tests that already finished/failed.
                    completed_count += 1
                    try:
                        # future.result() will raise exception if the task raised one
                        test_id, success, output_log, validation_issues, input_lines = future.result()

                        # Log PASSED/FAILED status
                        if success:
                            total_passed += 1
                            log_msg = f"Result: {test_id} PASSED"
                        else:
                            total_failed += 1
                            log_msg = f"Result: {test_id} FAILED - Issues: {'; '.join(validation_issues)}"
                            # Save artifacts for failed tests
                            try:
                                failure_stdin_path = os.path.join(FAILURES_DIR, f"stdin_{test_id}.txt")
                                failure_log_path = os.path.join(FAILURES_DIR, f"output_{test_id}.log")
                                # Write failure details
                                with open(failure_stdin_path, 'w', encoding='utf-8') as f_in:
                                    f_in.write("\n".join(input_lines))
                                with open(failure_log_path, 'w', encoding='utf-8') as f_log:
                                    f_log.write(f"# Test ID: {test_id}\n")
                                    f_log.write(f"# Issues: {'; '.join(validation_issues)}\n")
                                    f_log.write(f"# --- Input ({len(input_lines)} lines) ---\n")
                                    f_log.write("\n".join(input_lines))
                                    f_log.write("\n# --- Output Log ---\n")
                                    f_log.write(output_log)
                                log_msg += f" (Details saved to {FAILURES_DIR})"
                            except Exception as e:
                                log_msg += f" (ERROR saving failure details: {e})"

                        # Update progress label via queue
                        progress_text = f"Status: Completed {completed_count}/{len(all_futures)} ({total_passed} Passed, {total_failed} Failed)"
                        if self.stop_batch_event.is_set():
                            progress_text += " [Stopping...]"
                        self.message_queue.put_nowait(("progress", progress_text))
                        # Log individual result summary via queue
                        self.message_queue.put_nowait(("log", log_msg))

                    except Exception as exc:
                        # Handle error fetching result (e.g., task cancelled, unexpected error in task)
                        total_failed += 1
                        err_msg = f"ERROR processing result for a task: {exc}"
                        # Log the error via queue
                        self.message_queue.put_nowait(("log", err_msg))
                        # Update progress reflecting the error
                        progress_text = f"Status: Completed {completed_count}/{len(all_futures)} ({total_passed} Passed, {total_failed} Failed) [ERROR]"
                        if self.stop_batch_event.is_set():
                            progress_text += " [Stopping...]"
                        self.message_queue.put_nowait(("progress", progress_text))

        except InterruptedError: # Catch stop during generation/submission
             self.message_queue.put_nowait(("progress", "Status: Batch stopped by user during setup."))
             # Ensure failure count reflects reality if some tasks ran
             total_failed = len(all_futures) - total_passed # Assume non-passed are failed if stopped early
        except Exception as e:
             # Error during generation or executor setup
             self.message_queue.put_nowait(("log", f"FATAL BATCH ERROR: {e}"))
             import traceback
             self.message_queue.put_nowait(("log", traceback.format_exc()))
             total_failed = len(all_futures) - total_passed
        finally:
             # Signal GUI that batch is finished, sending final counts
             self.message_queue.put_nowait(("finished", (total_passed, total_failed)))


    # ... (Rest of the methods: _process_batch_queue, _run_manual_selected_test, _process_manual_queue, _clear_output_log, _log_output, _quit remain the same as the corrected versions) ...
    def _process_batch_queue(self):
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait();
                if not self.winfo_exists(): return
                if msg_type == "progress": self.batch_progress_label.config(text=data)
                elif msg_type == "log": self._log_output(data)
                elif msg_type == "finished":
                    total_passed, total_failed = data; final_msg = f"--- Batch Finished: {total_passed} Passed, {total_failed} Failed ---"; self._log_output(final_msg); self.batch_progress_label.config(text=f"Status: Finished ({total_passed} Passed, {total_failed} Failed)")
                    self.is_batch_running = False; self.start_batch_button.config(state=tk.NORMAL); self.stop_batch_button.config(state=tk.DISABLED); self.run_manual_button.config(state=tk.NORMAL); self.batch_thread = None
                    # --- Cleanup stdin.txt (if applicable, though temp dirs handle it now) ---
                    # stdin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), STDIN_FILENAME)
                    # if os.path.exists(stdin_path):
                    #     try: os.remove(stdin_path); self._log_output(f"INFO: Cleaned up {stdin_path}")
                    #     except Exception as e: self._log_output(f"WARNING: Failed to delete {stdin_path}: {e}")
                    return # Stop polling queue
        except queue.Empty: pass
        except Exception as e: print(f"Error processing queue: {e}") # Debug
        if self.is_batch_running and self.batch_thread and self.batch_thread.is_alive(): self.after(200, self._process_batch_queue)
        elif self.is_batch_running: self._log_output("--- Batch ended unexpectedly ---"); self.batch_progress_label.config(text="Status: Error/Ended Unexpectedly"); self.is_batch_running = False; self.start_batch_button.config(state=tk.NORMAL); self.stop_batch_button.config(state=tk.DISABLED); self.run_manual_button.config(state=tk.NORMAL); self.batch_thread = None

    def _run_manual_selected_test(self):
        if self.is_batch_running: messagebox.showwarning("Busy", "Batch is running.", parent=self); return
        jar = self.jar_path.get(); datainput_exe = self.datainput_path.get()
        if not jar or not os.path.exists(jar): messagebox.showerror("Error", "Select JAR.", parent=self); return
        if not datainput_exe or not os.path.exists(datainput_exe): messagebox.showerror("Error", "Select datainput EXE.", parent=self); return
        if not self.current_test_name or self.current_test_name not in self.test_data_sets: messagebox.showerror("Error", "Select a manual test.", parent=self); return
        test_data_lines = self.test_data_sets[self.current_test_name]
        if not test_data_lines: messagebox.showwarning("Warning", "Test case empty.", parent=self); return
        self._clear_output_log(); self._log_output(f"--- Starting Manual Test: {self.current_test_name} ---")
        self.run_manual_button.config(state=tk.DISABLED); self.start_batch_button.config(state=tk.DISABLED)
        def manual_run_thread():
            dummy_stop = threading.Event(); test_id = f"Manual_{self.current_test_name}"
            result = _run_one_batch_test_task(test_id, test_data_lines, jar, datainput_exe, dummy_stop)
            self.message_queue.put(("log", f"--- Manual Test {self.current_test_name} Finished ---")); self.message_queue.put(("log", f"Success: {result[1]}")); self.message_queue.put(("log", f"Validation Issues: {'; '.join(result[3])}")); self.message_queue.put(("log", f"--- Full Output Log ---")); self.message_queue.put(("log", result[2])); self.message_queue.put(("manual_finished", None))
        threading.Thread(target=manual_run_thread, daemon=True).start()
        self.after(100, self._process_manual_queue)

    def _process_manual_queue(self):
         try:
            while True:
                msg_type, data = self.message_queue.get_nowait();
                if not self.winfo_exists(): return
                if msg_type == "log": self._log_output(data)
                elif msg_type == "manual_finished": self.run_manual_button.config(state=tk.NORMAL); self.start_batch_button.config(state=tk.NORMAL); return
         except queue.Empty: pass
         except Exception as e: print(f"Error manual queue: {e}") # Debug
         if self.run_manual_button['state'] == tk.DISABLED: self.after(100, self._process_manual_queue)

    def _clear_output_log(self):
        if not self.output_log_text.winfo_exists(): return
        self.output_log_text.config(state=tk.NORMAL); self.output_log_text.delete('1.0', tk.END); self.output_log_text.config(state=tk.DISABLED)

    def _log_output(self, message):
        if not self.output_log_text.winfo_exists(): return
        try:
            self.output_log_text.config(state=tk.NORMAL)
            self.output_log_text.insert(tk.END, str(message) + "\n")
            self.output_log_text.see(tk.END)
            self.output_log_text.config(state=tk.DISABLED)
        except tk.TclError: # Handle cases where widget might be destroyed during update
            pass

    def _quit(self):
        if self.is_batch_running:
            if messagebox.askyesno("Exit", "Batch test running. Stop and exit?", parent=self): self.stop_batch_event.set()
            else: return
        self.destroy()


# --- Main Execution ---
if __name__ == "__main__":
    app = ElevatorTesterApp()
    app.mainloop()