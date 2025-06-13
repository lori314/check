#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import subprocess
import threading
import queue # Used for internal thread comms in run_single_test, NOT inter-process
import multiprocessing as mp # Use multiprocessing for parallelism
from multiprocessing import Pool, Manager, cpu_count, TimeoutError as MPTimeoutError
import os
import sys
import re
import platform
import random
from decimal import Decimal
from pathlib import Path
import time
import math # For math.nextafter (Python 3.9+) or alternative
import shutil # For copying files and removing directory tree
import tempfile # For creating temporary directories
import signal # For process group termination on Unix

# --- Attempt to import checker logic and config ---
try:
    from checker import Checker
    from config import (
        int_to_floor as config_int_to_floor, # Rename to avoid conflict
        INPUT_PATTERN, OUTPUT_PATTERN, FLOORS, NUM_ELEVATORS, EPSILON,
        MIN_FLOOR, MAX_FLOOR, INVALID_FLOOR
    )
    CHECKER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import 'checker' or 'config'. Validation disabled. Error: {e}", file=sys.stderr)
    CHECKER_AVAILABLE = False
    # Define fallback values if config is missing essential parts
    MIN_FLOOR = -3
    MAX_FLOOR = 10
    INVALID_FLOOR = 0
    NUM_ELEVATORS = 6

# --- Default Configurable Values ---
DEFAULT_JAR_DIR = '.'
DEFAULT_PARALLELISM = max(1, cpu_count() - 1)
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_REQUESTS = 30
TARGET_ELEVATOR_ID = 1
OTHER_ELEVATOR_PROBABILITY = 0.05
MAX_TIME_DELTA = 0.2
MAX_BURST_SIZE = 4
TOTAL_ELEVATORS = NUM_ELEVATORS if CHECKER_AVAILABLE else 6

# --- Datainput Executable Selection ---
def get_datainput_executable():
    system = platform.system().lower()
    arch = platform.machine().lower()
    if system == "windows": return "datainput_student_win64.exe"
    elif system == "linux": return "datainput_student_linux_x86_64"
    elif system == "darwin":
        m1_binary = "datainput_student_darwin_m1"
        intel_binary = "datainput_student_darwin_x86_64" # Example if you have both
        # Prioritize M1 if exists, otherwise try Intel or generic
        if os.path.exists(m1_binary): return m1_binary
        # elif os.path.exists(intel_binary): return intel_binary # Add if needed
        else: print("Warning: M1 datainput binary not found. Check file name/path."); return "datainput_student_darwin"
    else: raise OSError(f"Unsupported OS: {system}")

try:
    # Determine absolute path relative to script location if needed
    script_dir = Path(__file__).parent.resolve()
    DATAINPUT_EXE_NAME = get_datainput_executable()
    DATAINPUT_EXE_PATH = script_dir / DATAINPUT_EXE_NAME # Assume it's in the same dir as script
    DATAINPUT_EXISTS = DATAINPUT_EXE_PATH.exists()
    if not DATAINPUT_EXISTS:
        # Fallback: check current working directory
        if Path(DATAINPUT_EXE_NAME).exists():
            DATAINPUT_EXE_PATH = Path(DATAINPUT_EXE_NAME).resolve()
            DATAINPUT_EXISTS = True

except OSError as e:
    print(f"Error detecting datainput executable: {e}", file=sys.stderr)
    DATAINPUT_EXE_PATH = Path("datainput_not_found")
    DATAINPUT_EXISTS = False
except NameError: # Handle if __file__ is not defined (e.g., interactive session)
     DATAINPUT_EXE_PATH = Path(get_datainput_executable()) # Check CWD
     DATAINPUT_EXISTS = DATAINPUT_EXE_PATH.exists()


# --- Helper Functions ---
def generate_floor():
    while True:
        floor = random.randint(MIN_FLOOR, MAX_FLOOR)
        if floor != INVALID_FLOOR: return floor

def translate_to_bf(floor):
    try: return config_int_to_floor(floor)
    except NameError:
        if floor > 0: return f"F{floor}"
        else: return f"B{-floor + 1}"
    except ValueError: return "INVALID"

def generate_dense_data(num_requests, target_elevator_id, other_prob, max_delta, max_burst, total_elevators):
    requests = []
    current_time = 0.0
    passenger_id_counter = 1
    while len(requests) < num_requests:
        burst_size = random.randint(1, max_burst)
        for _ in range(burst_size):
            if len(requests) >= num_requests: break
            passenger_id = passenger_id_counter; passenger_id_counter += 1
            priority = random.randint(1, 20); from_floor = generate_floor(); to_floor = generate_floor()
            while from_floor == to_floor: to_floor = generate_floor()
            elevator_id = target_elevator_id
            if random.random() < other_prob and total_elevators > 1:
                while True:
                    other_id = random.randint(1, total_elevators);
                    if other_id != target_elevator_id: elevator_id = other_id; break
            request_str = f"[{current_time:.1f}]{passenger_id}-PRI-{priority}-FROM-{translate_to_bf(from_floor)}-TO-{translate_to_bf(to_floor)}-BY-{elevator_id}"
            requests.append((current_time, request_str)) # Keep time info if needed by checker later
        time_increment = random.uniform(0, max_delta); current_time += time_increment
        try: current_time = math.nextafter(current_time, float('inf'))
        except AttributeError: current_time += sys.float_info.epsilon
    return [req[1] for req in requests] # Return only strings for stdin.txt

# --- I/O Reader Thread (used within run_single_test) ---
def read_output(pipe, output_queue):
    try:
        for line in iter(pipe.readline, b''): output_queue.put(line.decode('utf-8', errors='ignore').strip())
    except Exception: pass
    finally:
        try: pipe.close()
        except (IOError, OSError): pass

# --- WORKER FUNCTION (MUST BE TOP-LEVEL for multiprocessing) ---
def run_single_test(args_tuple):
    """
    Runs the test for a single JAR file in its own temporary directory.
    Designed to be called by pool.map (accepts a single tuple argument).
    Puts the result dictionary onto the provided multiprocessing queue.
    """
    # --- 1. Unpack Arguments ---
    jar_path_orig, input_lines, timeout_sec, results_queue, datainput_exe_path_str = args_tuple # Added datainput path
    jar_name = jar_path_orig.name
    pid = os.getpid()
    # print(f"[{jar_name}/{pid}] Starting test...") # Less verbose for GUI

    # --- 2. Initialize State ---
    start_real_time = time.time()
    stdout_lines = []; stderr_lines = []
    message = "Timeout"; passed = False
    checker_results = ["Checker not run"]
    process = None; return_code = None
    temp_dir_path = None

    try:
        # --- 3. Create Unique Temporary Directory ---
        temp_dir_path = Path(tempfile.mkdtemp(prefix=f"p1test_{jar_name}_{pid}_"))

        # --- 4. Prepare Files in Temp Directory ---
        datainput_exe_path = Path(datainput_exe_path_str) # Convert string back to Path
        datainput_dest = temp_dir_path / datainput_exe_path.name
        jar_dest = temp_dir_path / jar_name
        stdin_dest = temp_dir_path / "stdin.txt"

        # Copy datainput executable
        if not datainput_exe_path.exists(): raise FileNotFoundError(f"Datainput executable not found at source: {datainput_exe_path}")
        shutil.copy2(datainput_exe_path, datainput_dest)

        # Copy student JAR
        jar_path_abs = jar_path_orig.resolve()
        if not jar_path_abs.exists(): raise FileNotFoundError(f"Student JAR not found at source: {jar_path_abs}")
        shutil.copy2(jar_path_abs, jar_dest)

        # Write generated input to stdin.txt
        with open(stdin_dest, "w", encoding='utf-8') as f:
            for line in input_lines: f.write(line + "\n")

        # --- 5. Set Permissions (Non-Windows) ---
        if platform.system() != "Windows":
            try: os.chmod(datainput_dest, 0o755)
            except OSError: pass # Ignore if fails

        # --- 6. Construct Command ---
        command = f'"{datainput_dest}" | java -jar "{jar_dest}"'

        # --- 7. Run the Piped Process ---
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            shell=True, cwd=temp_dir_path, text=False,
            # Process group handling for cleaner termination on Unix
            preexec_fn=os.setsid if platform.system() != "Windows" else None
        )

        # --- 8. Monitor Process and Read Output ---
        stdout_queue_local = queue.Queue()
        stderr_queue_local = queue.Queue()
        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, stdout_queue_local), daemon=True)
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, stderr_queue_local), daemon=True)
        stdout_thread.start(); stderr_thread.start()

        try:
            return_code = process.wait(timeout=timeout_sec)
            message = "OK"
        except subprocess.TimeoutExpired:
            print(f"[{jar_name}/{pid}] Timeout ({timeout_sec}s). Terminating.", file=sys.stderr)
            message = "Timeout"; passed = False
            try:
                # Terminate the process group on Unix, process tree on Windows is harder
                if platform.system() != "Windows":
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM) # Send TERM to group
                else:
                    # Try to terminate parent, might not get children spawned by shell pipe
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(0.5)
                if process.poll() is None: process.kill() # Force kill if still alive
            except Exception: pass # Ignore errors during termination

        stdout_thread.join(timeout=1.0); stderr_thread.join(timeout=1.0)
        while not stdout_queue_local.empty(): stdout_lines.append(stdout_queue_local.get_nowait())
        while not stderr_queue_local.empty(): stderr_lines.append(stderr_queue_local.get_nowait())

        # --- 9. Validation ---
        if message == "OK":
            if return_code != 0: message = f"Runtime Error (Code: {return_code})"; passed = False;
            else:
                if CHECKER_AVAILABLE:
                    # print(f"[{jar_name}/{pid}] Process OK. Running checker...") # Verbose
                    try:
                        checker = Checker(); checker_results = checker.check(stdout_lines, input_lines)
                        if "Failed" in checker_results[0]: message = "Check Failed"; passed = False
                        elif "Passed" in checker_results[0]: message = "Check Passed"; passed = True
                        else: message = "Check Ambiguous"; passed = False
                    except Exception as check_e: message = f"Checker Error: {type(check_e).__name__}"; passed = False; print(f"[{jar_name}/{pid}] Checker Error: {check_e}", file=sys.stderr)
                else:
                    if not stderr_lines: message = "Ran (No Checker)"; passed = True
                    else: message = "Ran (No Checker, +Stderr)"; passed = False

    # --- Error Handling ---
    except FileNotFoundError as e: message = f"Setup Error: File Not Found ({e})"; passed = False; print(f"[{jar_name}/{pid}] {message}", file=sys.stderr)
    except RuntimeError as e: message = f"Setup Error: {e}"; passed = False; print(f"[{jar_name}/{pid}] {message}", file=sys.stderr)
    except Exception as e: message = f"Tester Error: {type(e).__name__}"; passed = False; print(f"[{jar_name}/{pid}] Error in test worker: {e}", file=sys.stderr, exc_info=False)
    finally: # --- Cleanup ---
        if process and process.poll() is None: # Ensure process is dead
             try: process.kill()
             except Exception: pass
        if temp_dir_path and temp_dir_path.exists(): # Clean up temp dir
            try: shutil.rmtree(temp_dir_path, ignore_errors=True)
            except Exception as clean_e: print(f"Warning [{jar_name}/{pid}]: Failed cleanup {temp_dir_path}: {clean_e}", file=sys.stderr)

        # --- Send result back via Queue ---
        end_real_time = time.time(); duration = end_real_time - start_real_time
        result_dict = { "jar_name": jar_name, "passed": passed, "message": message, "duration_sec": duration, "stdout": stdout_lines, "stderr": stderr_lines, "checker_log": checker_results }
        try: results_queue.put(result_dict)
        except Exception as q_err: print(f"Error [{jar_name}/{pid}] putting result on queue: {q_err}", file=sys.stderr)
    # print(f"[{jar_name}/{pid}] Test function finished.") # Verbose debug

# --- Main GUI Application Class ---
class CheckerApp:
    def __init__(self, master):
        self.master = master
        master.title("Parallel Elevator Checker GUI")
        master.geometry("1000x750")

        self.jar_dir_path = tk.StringVar(value=DEFAULT_JAR_DIR)
        self.num_requests = tk.IntVar(value=DEFAULT_REQUESTS)
        self.parallelism = tk.IntVar(value=DEFAULT_PARALLELISM)
        self.timeout = tk.IntVar(value=DEFAULT_TIMEOUT_SECONDS)
        self.target_elevator = tk.IntVar(value=TARGET_ELEVATOR_ID)

        self.test_runner_thread = None
        self.pool = None
        self.manager = None
        self.results_queue = None
        self.running_tests = False
        self.jar_results = {}

        self._setup_gui()

        if not DATAINPUT_EXISTS:
             messagebox.showwarning("Warning", f"Datainput executable '{DATAINPUT_EXE_PATH}' not found. Testing will fail.")
        if not CHECKER_AVAILABLE:
             messagebox.showwarning("Warning", f"checker.py or config.py not found/importable. Validation will be disabled.")

    def _setup_gui(self):
        top_frame = ttk.Frame(self.master, padding="10")
        top_frame.pack(fill=tk.X)
        ttk.Label(top_frame, text="JAR Dir:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(top_frame, textvariable=self.jar_dir_path, width=40).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(top_frame, text="Browse...", command=self.browse_dir).pack(side=tk.LEFT)
        ttk.Label(top_frame, text="Parallel:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Spinbox(top_frame, from_=1, to=cpu_count()*2, textvariable=self.parallelism, width=4).pack(side=tk.LEFT, padx=5) # Allow oversubscription
        ttk.Label(top_frame, text="Timeout(s):").pack(side=tk.LEFT)
        ttk.Spinbox(top_frame, from_=10, to=600, textvariable=self.timeout, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(top_frame, text="Requests:").pack(side=tk.LEFT)
        ttk.Spinbox(top_frame, from_=1, to=200, textvariable=self.num_requests, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(top_frame, text="Target EID:").pack(side=tk.LEFT)
        ttk.Spinbox(top_frame, from_=1, to=TOTAL_ELEVATORS, textvariable=self.target_elevator, width=3).pack(side=tk.LEFT, padx=5)

        mid_frame = ttk.Frame(self.master, padding="5 10")
        mid_frame.pack(fill=tk.X)
        self.run_button = ttk.Button(mid_frame, text="Run All Tests", command=self.start_tests)
        self.run_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(mid_frame, text="Stop All Tests", command=self.stop_tests, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.status_label = ttk.Label(mid_frame, text="Status: Idle")
        self.status_label.pack(side=tk.RIGHT, padx=10)

        bottom_frame = ttk.Frame(self.master, padding="10")
        bottom_frame.pack(fill=tk.BOTH, expand=True)
        bottom_frame.rowconfigure(1, weight=1); bottom_frame.rowconfigure(3, weight=2); bottom_frame.columnconfigure(0, weight=1)

        ttk.Label(bottom_frame, text="Test Results Summary:").grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.results_tree = ttk.Treeview(bottom_frame, columns=('Status', 'Time', 'Info'), show='headings')
        self.results_tree.heading('#0', text='JAR Name') # Use column #0 for JAR name if text isn't reliable
        self.results_tree.heading('Status', text='Status'); self.results_tree.heading('Time', text='Time (s)'); self.results_tree.heading('Info', text='Info')
        self.results_tree.column('#0', width=200); self.results_tree.column('Status', width=80, anchor='center'); self.results_tree.column('Time', width=80, anchor='e'); self.results_tree.column('Info', width=400)
        self.results_tree.grid(row=1, column=0, sticky="nsew", pady=5)
        tree_scroll = ttk.Scrollbar(bottom_frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=tree_scroll.set); tree_scroll.grid(row=1, column=1, sticky='ns', pady=5)
        self.results_tree.bind('<<TreeviewSelect>>', self.show_details)

        ttk.Label(bottom_frame, text="Details for Selected JAR:").grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 2))
        self.details_text = scrolledtext.ScrolledText(bottom_frame, wrap=tk.NONE, height=15, width=80) # Use tk.NONE for wrap
        self.details_text.grid(row=3, column=0, columnspan=2, sticky="nsew")
        self.details_text.configure(state='disabled')

    def browse_dir(self):
        dirpath = filedialog.askdirectory(title="Select Directory Containing JARs", initialdir=self.jar_dir_path.get())
        if dirpath: self.jar_dir_path.set(dirpath)

    def set_status(self, message):
        self.status_label.config(text=f"Status: {message}")

    def show_details(self, event=None):
        selected_items = self.results_tree.selection()
        self.details_text.configure(state='normal')
        self.details_text.delete('1.0', tk.END)
        if not selected_items:
            self.details_text.insert('1.0', "Select a JAR from the results table.")
        else:
            item_id = selected_items[0]
            jar_name = self.results_tree.item(item_id, 'text') # Get JAR name from Treeview's internal text property

            if jar_name in self.jar_results:
                result = self.jar_results[jar_name]
                details = f"--- Details for {jar_name} ---\n"
                details += f"Status: {'PASS' if result['passed'] else 'FAIL'}\n"
                details += f"Duration: {result['duration_sec']:.2f}s\n"
                details += f"Message: {result['message']}\n"
                details += "\n--- Checker Log ---\n"
                details += "\n".join(result.get('checker_log', ["(Log Unavailable)"]))
                details += "\n\n--- Standard Output ---\n"
                details += "\n".join(result.get('stdout', ["(No stdout)"]))
                details += "\n\n--- Standard Error ---\n"
                details += "\n".join(result.get('stderr', ["(No stderr)"]))
                self.details_text.insert('1.0', details)
            else:
                self.details_text.insert('1.0', f"Details not yet available or lost for {jar_name}.")
        self.details_text.configure(state='disabled')

       # Inside CheckerApp class

    def start_tests(self):
        if self.running_tests:
            messagebox.showwarning("Busy", "Tests are already running.")
            return

        if not DATAINPUT_EXISTS:
            messagebox.showerror("Error", f"Datainput executable '{DATAINPUT_EXE_PATH}' not found.")
            return

        # Get config values from GUI variables
        jar_dir = Path(self.jar_dir_path.get())
        parallelism = self.parallelism.get()
        timeout = self.timeout.get()
        num_req = self.num_requests.get()
        target_eid = self.target_elevator.get()

        # Validate JAR directory
        if not jar_dir.is_dir():
            messagebox.showerror("Error", f"Invalid directory: {jar_dir}")
            return

        # Find JAR files
        jar_files = list(jar_dir.glob('*.jar'))
        if not jar_files:
            messagebox.showinfo("No JARs", f"No .jar files found in {jar_dir}")
            return

        # Reset GUI state for a new run
        self.set_status(f"Preparing {len(jar_files)} tests...")
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.results_tree.delete(*self.results_tree.get_children()) # Clear results table
        self.details_text.configure(state='normal'); self.details_text.delete('1.0', tk.END); self.details_text.configure(state='disabled')
        self.jar_results = {} # Clear stored results
        self.running_tests = True

        # Prepare args for worker processes
        print("Generating input data for all tests...")
        test_args_list = []
        self.manager = Manager() # Start the multiprocessing manager
        self.results_queue = self.manager.Queue() # Create the shared queue

        for jar_path in jar_files:
            # Insert placeholder row in Treeview
            # Use resolved path string as unique ID to handle potential duplicate filenames
            iid = str(jar_path.resolve())
            self.results_tree.insert('', tk.END, text=jar_path.name, values=('Queued', '-', 'Waiting...'), iid=iid)

            # --- CORRECTED DATA GENERATION ASSIGNMENT ---
            # generate_dense_data now returns only the list of input strings
            input_lines = generate_dense_data(
                num_req, target_eid, OTHER_ELEVATOR_PROBABILITY,
                MAX_TIME_DELTA, MAX_BURST_SIZE, TOTAL_ELEVATORS
            )
            # --- ---

            # Add the arguments tuple for this test to the list
            # Ensure DATAINPUT_EXE_PATH is passed as a string for pickling
            test_args_list.append((jar_path, input_lines, timeout, self.results_queue, str(DATAINPUT_EXE_PATH)))

        print("Input data generated.")
        self.set_status(f"Running {len(jar_files)} tests...")

        # Run the Pool management in a separate thread to avoid blocking GUI
        self.test_runner_thread = threading.Thread(
            target=self._run_parallel_tests_thread,
            args=(test_args_list, parallelism),
            daemon=True # Allow program to exit even if this thread hangs (shouldn't happen ideally)
        )
        self.test_runner_thread.start()

        # Start polling the results queue periodically from the GUI thread
        self.master.after(200, self.check_results_queue)

    def _run_parallel_tests_thread(self, test_args_list, parallelism):
        all_tasks_processed = False
        try:
            # Need to handle context for pool on some platforms
            ctx = mp.get_context('spawn') if platform.system() == 'Darwin' or platform.system() == 'Windows' else mp.get_context()
            with ctx.Pool(processes=parallelism) as self.pool:
                # Map blocks this thread until all results are implicitly collected (or error)
                self.pool.map(run_single_test, test_args_list)
            all_tasks_processed = True # Map finished
            print("Worker pool finished processing.")
            self.results_queue.put(("DONE", None)) # Signal completion
        except Exception as e:
            print(f"Error in parallel test execution thread: {e}", file=sys.stderr)
            if not all_tasks_processed: self.results_queue.put(("ERROR", f"Pool execution failed: {e}"))
            # If error during map, results might be partial
        finally:
            self.pool = None # Clear pool reference


    def check_results_queue(self):
        if not self.running_tests and not (self.manager and self.results_queue): return # Stop if told to or manager gone

        gui_updated = False
        try:
            while True: # Process all available messages
                msg = self.results_queue.get_nowait()
                gui_updated = True # We got something

                if isinstance(msg, tuple) and len(msg) == 2:
                    msg_type, msg_data = msg
                    if msg_type == "DONE":
                        self.set_status("All tests complete."); self.running_tests = False
                        self.run_button.config(state=tk.NORMAL); self.stop_button.config(state=tk.DISABLED)
                        if self.manager: self.manager.shutdown(); self.manager = None; self.results_queue = None
                        print("Polling stopped: DONE received.")
                        return # Stop polling loop
                    elif msg_type == "ERROR":
                         self.set_status(f"Error: {msg_data}"); self.running_tests = False
                         self.run_button.config(state=tk.NORMAL); self.stop_button.config(state=tk.DISABLED)
                         if self.manager: self.manager.shutdown(); self.manager = None; self.results_queue = None
                         print(f"Polling stopped: ERROR received: {msg_data}")
                         return # Stop polling loop
                    else:
                         print(f"Warning: Unknown tuple message type from queue: {msg_type}")

                elif isinstance(msg, dict) and "jar_name" in msg: # Received a result dictionary
                    result = msg
                    jar_name = result["jar_name"]
                    # Find corresponding iid using original path if stored, otherwise fallback to name
                    iid = None
                    for item_iid in self.results_tree.get_children():
                         if self.results_tree.item(item_iid, 'text') == jar_name:
                              iid = item_iid
                              break
                    if iid is None: # Fallback if iid wasn't path or item removed?
                         iid = jar_name # Use name as last resort iid

                    self.jar_results[jar_name] = result # Store full result using name as key

                    status = "PASS" if result["passed"] else "FAIL"
                    duration = f"{result['duration_sec']:.2f}"
                    info = result['message'].splitlines()[0] if result['message'] else 'N/A'
                    if self.results_tree.exists(iid):
                         self.results_tree.item(iid, values=(status, duration, info))
                         # Optionally change tag/color based on status
                         self.results_tree.tag_configure('FAIL', foreground='red')
                         self.results_tree.tag_configure('PASS', foreground='green')
                         self.results_tree.item(iid, tags=(status,))

                    else:
                         print(f"Warning: Treeview item ID '{iid}' for JAR '{jar_name}' not found for update.")

                else:
                    print(f"Warning: Unknown message format from queue: {type(msg)}")

        except queue.Empty:
            pass # No new results right now
        except Exception as e:
            print(f"Error processing results queue: {e}", file=sys.stderr)
            # Decide whether to stop polling or continue

        # Reschedule check only if tests are marked as running
        if self.running_tests:
            self.master.after(200, self.check_results_queue)
        # else: Stop rescheduling if self.running_tests became false

    def stop_tests(self):
        if not self.running_tests: return
        self.set_status("Stopping tests...")
        self.running_tests = False # Signal queue checker to stop

        # Important: Terminate the pool *first* to stop workers
        if self.pool:
            try:
                print("Terminating worker pool...")
                self.pool.terminate() # SIGTERM workers
                self.pool.join(timeout=5) # Wait briefly
                print("Worker pool terminated.")
            except Exception as e: print(f"Error terminating pool: {e}", file=sys.stderr)
            finally: self.pool = None

        # Then shutdown the manager (which owns the queue)
        if self.manager:
             try:
                 print("Shutting down manager...")
                 # Drain queue quickly before shutdown? Might lose messages.
                 # while not self.results_queue.empty():
                 #    try: self.results_queue.get_nowait()
                 #    except queue.Empty: break
                 self.manager.shutdown()
             except Exception as e: print(f"Error shutting down manager: {e}", file=sys.stderr)
             finally: self.manager = None; self.results_queue = None

        # Check the managing thread (it should exit after pool finishes/errors)
        if self.test_runner_thread and self.test_runner_thread.is_alive():
             print("Warning: Test runner thread still alive after stop request.")
             # It might exit shortly if pool.map unblocks due to termination

        self.set_status("Tests stopped.")
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)


    def on_closing(self):
        if self.running_tests:
            if messagebox.askokcancel("Quit", "Tests are running. Stop them and quit?"):
                self.stop_tests() # Attempt clean stop first
                self.master.destroy()
        else:
            if self.manager: # Ensure manager cleanup if exited abnormally
                try: self.manager.shutdown()
                except Exception: pass
            self.master.destroy()

# --- Run Application ---
if __name__ == "__main__":
    mp.freeze_support() # Necessary for Windows frozen executables
    root = tk.Tk()
    app = CheckerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()