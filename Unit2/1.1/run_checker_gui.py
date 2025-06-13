# run_checker_gui.py
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import subprocess
import threading
import queue
import os
import sys
import re
import platform
from decimal import Decimal

from datagen import generate_data
from checker import Checker

# --- Datainput Executable Selection ---
def get_datainput_executable():
    system = platform.system().lower()
    arch = platform.machine().lower() # For potential M1 mac differentiation if needed

    if system == "windows":
        return "datainput_student_win64.exe"
    elif system == "linux":
        # Assuming x86_64, add checks for other architectures if necessary
        return "datainput_student_linux_x86_64"
    elif system == "darwin":
        # Assuming M1/ARM, provide the specific binary name
        # Check if the M1 binary exists, otherwise fallback?
        m1_binary = "datainput_student_darwin_m1"
        # intel_binary = "datainput_student_darwin_x86_64" # If you had one
        if os.path.exists(m1_binary): # Simple check
             return m1_binary
        else:
             # Fallback or error - adjust as needed
             print("Warning: M1 datainput binary not found. Trying generic name.")
             return "datainput_student_darwin" # Or raise error
    else:
        raise OSError(f"Unsupported operating system: {system}")

DATAINPUT_EXE = get_datainput_executable()
INPUT_FILENAME = "stdin.txt"
OUTPUT_FILENAME = "stdout.txt" # To save output if needed
JAR_FILENAME = "code.jar" # Default name for the target jar

class CheckerApp:
    def __init__(self, master):
        self.master = master
        master.title("BUAA OO HW5+ Checker")
        master.geometry("900x700")

        self.jar_path = tk.StringVar(value=f"./{JAR_FILENAME}") # Default to current dir
        self.num_requests = tk.IntVar(value=10)
        self.is_hack_mode = tk.BooleanVar(value=False)
        self.output_queue = queue.Queue()
        self.running_process = None

        # --- Top Frame: JAR Selection ---
        top_frame = ttk.Frame(master, padding="10")
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="Student .jar File:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(top_frame, textvariable=self.jar_path, width=60).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(top_frame, text="Browse...", command=self.browse_jar).pack(side=tk.LEFT)
        # ttk.Button(top_frame, text="Create run.bat", command=self.create_bat).pack(side=tk.LEFT, padx=5) # Optional BAT generation

        # --- Middle Frame: Data Generation & Run ---
        mid_frame = ttk.Frame(master, padding="10")
        mid_frame.pack(fill=tk.X)

        ttk.Label(mid_frame, text="Num Requests:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Spinbox(mid_frame, from_=1, to=100, textvariable=self.num_requests, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(mid_frame, text="Hack Mode Data", variable=self.is_hack_mode).pack(side=tk.LEFT, padx=5)
        ttk.Button(mid_frame, text="Generate Data", command=self.generate).pack(side=tk.LEFT, padx=5)
        ttk.Button(mid_frame, text="Run Test", command=self.run_test).pack(side=tk.LEFT, padx=5)
        ttk.Button(mid_frame, text="Stop Process", command=self.stop_process).pack(side=tk.LEFT, padx=5)
        self.status_label = ttk.Label(mid_frame, text="Status: Idle")
        self.status_label.pack(side=tk.RIGHT, padx=10)


        # --- Bottom Frame: Text Areas ---
        bottom_frame = ttk.Frame(master, padding="10")
        bottom_frame.pack(fill=tk.BOTH, expand=True)
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.columnconfigure(1, weight=2)
        bottom_frame.columnconfigure(2, weight=2)
        bottom_frame.rowconfigure(1, weight=1)

        ttk.Label(bottom_frame, text="Input (stdin.txt)").grid(row=0, column=0, sticky="w")
        self.input_text = scrolledtext.ScrolledText(bottom_frame, wrap=tk.WORD, height=10, width=30)
        self.input_text.grid(row=1, column=0, sticky="nsew", padx=(0, 5))

        ttk.Label(bottom_frame, text="Output (stdout)").grid(row=0, column=1, sticky="w")
        self.output_text = scrolledtext.ScrolledText(bottom_frame, wrap=tk.WORD, height=10, width=50)
        self.output_text.grid(row=1, column=1, sticky="nsew", padx=5)

        ttk.Label(bottom_frame, text="Checker Result").grid(row=0, column=2, sticky="w")
        self.result_text = scrolledtext.ScrolledText(bottom_frame, wrap=tk.WORD, height=10, width=50)
        self.result_text.grid(row=1, column=2, sticky="nsew", padx=(5, 0))

        # Check for datainput executable on startup
        if not os.path.exists(DATAINPUT_EXE):
             messagebox.showwarning("Warning", f"Datainput executable '{DATAINPUT_EXE}' not found in the current directory. Please place it here.")


    def browse_jar(self):
        filepath = filedialog.askopenfilename(
            title="Select Student .jar File",
            filetypes=[("JAR files", "*.jar"), ("All files", "*.*")]
        )
        if filepath:
            self.jar_path.set(filepath)
            # Optionally copy/rename to code.jar if needed by specific scripts
            # try:
            #     import shutil
            #     shutil.copy(filepath, JAR_FILENAME)
            #     self.jar_path.set(f"./{JAR_FILENAME}") # Update path if copied
            #     print(f"Copied selected file to {JAR_FILENAME}")
            # except Exception as e:
            #     messagebox.showerror("Error", f"Failed to copy JAR file: {e}")

    # def create_bat(self): # Optional BAT generation
    #     jar = self.jar_path.get()
    #     if not jar or not os.path.exists(jar):
    #         messagebox.showerror("Error", "Please select a valid .jar file first.")
    #         return
    #     bat_content = f"@echo off\njava -jar \"{os.path.basename(jar)}\"\npause"
    #     try:
    #         with open("run.bat", "w") as f:
    #             f.write(bat_content)
    #         messagebox.showinfo("Success", f"Created run.bat for {os.path.basename(jar)}")
    #     except Exception as e:
    #          messagebox.showerror("Error", f"Failed to create run.bat: {e}")

    def generate(self):
        self.input_text.delete('1.0', tk.END)
        self.output_text.delete('1.0', tk.END)
        self.result_text.delete('1.0', tk.END)
        self.set_status("Generating data...")
        try:
            num_req = self.num_requests.get()
            hack_mode = self.is_hack_mode.get()
            generated_lines = generate_data(num_req, INPUT_FILENAME, hack_mode)
            self.input_text.insert('1.0', "\n".join(generated_lines))
            self.set_status(f"Generated {len(generated_lines)} requests into {INPUT_FILENAME}")
        except ValueError as e:
            messagebox.showerror("Data Generation Error", str(e))
            self.set_status("Data generation failed.")
        except Exception as e:
             messagebox.showerror("Error", f"An unexpected error occurred during generation: {e}")
             self.set_status("Data generation failed.")

    def run_test(self):
        jar = self.jar_path.get()
        if not jar or not os.path.exists(jar):
            messagebox.showerror("Error", "Student .jar file not found or not selected.")
            return
        if not os.path.exists(INPUT_FILENAME):
             messagebox.showerror("Error", f"Input file '{INPUT_FILENAME}' not found. Please generate data first.")
             return
        if not os.path.exists(DATAINPUT_EXE):
              messagebox.showerror("Error", f"Datainput executable '{DATAINPUT_EXE}' not found.")
              return

        # Clear previous results
        self.output_text.delete('1.0', tk.END)
        self.result_text.delete('1.0', tk.END)
        self.set_status("Running test...")

        # Run in a separate thread
        self.thread = threading.Thread(target=self._run_process, args=(jar,), daemon=True)
        self.thread.start()
        self.master.after(100, self.check_queue) # Start checking queue for updates

    def _run_process(self, jar_filepath):
        try:
            # Ensure datainput has execute permissions (Linux/Mac)
            if platform.system() != "Windows":
                try:
                    os.chmod(DATAINPUT_EXE, 0o755) # rwxr-xr-x
                except OSError as e:
                    self.output_queue.put(("status", f"Warning: Could not chmod {DATAINPUT_EXE}: {e}"))
                    # Proceed anyway, it might already have permissions

            # Command: ./datainput | java -jar code.jar
            # Need absolute path for jar? Or ensure cwd is correct.
            # Let's assume datainput and jar are accessible from cwd.
            # Use absolute path for jar for robustness.
            jar_abs_path = os.path.abspath(jar_filepath)
            command = f"\"{os.path.abspath(DATAINPUT_EXE)}\" | java -jar \"{jar_abs_path}\""

            self.output_queue.put(("status", f"Executing: {command}"))

            # Using subprocess.Popen for more control (like termination)
            # Note: shell=True can be a security risk if command is user-influenced,
            # but needed here for the pipe operator '|'. Be cautious.
            # Alternatively, manage two processes and pipe manually.
            self.running_process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                # cwd=os.path.dirname(sys.executable) # Or specify CWD if needed
            )

            # Determine timeout (Tmax + buffer)
            is_hack = self.is_hack_mode.get()
            timeout_sec = 130.0 if is_hack else None # 120s + buffer for hack, None initially for normal
            if not is_hack:
                try:
                    with open(INPUT_FILENAME, 'r') as f:
                        lines = f.readlines()
                    last_line = lines[-1].strip() if lines else ""
                    match = re.match(r"\[\s*(\d+\.\d+)\s*\]", last_line)
                    if match:
                        t_std = Decimal(match.group(1))
                        t_max = max(t_std + 10, t_std * Decimal("1.15"))
                        timeout_sec = float(t_max + 15) # Tmax + 15s buffer
                        self.output_queue.put(("status", f"Calculated T_max={t_max:.2f}s, Timeout={timeout_sec:.1f}s"))
                    else:
                        timeout_sec = 180.0 # Default large timeout if Tstd parse fails
                        self.output_queue.put(("status", f"Could not parse T_std, using default timeout {timeout_sec}s"))
                except Exception as e:
                    timeout_sec = 180.0
                    self.output_queue.put(("status", f"Error calculating T_max, using default timeout {timeout_sec}s: {e}"))

            try:
                stdout_data, stderr_data = self.running_process.communicate(timeout=timeout_sec)
                self.output_queue.put(("stdout", stdout_data))
                if stderr_data:
                    self.output_queue.put(("stderr", stderr_data))
                self.output_queue.put(("status", "Process finished."))
                self.output_queue.put(("run_complete", None))

            except subprocess.TimeoutExpired:
                self.output_queue.put(("status", f"Process timed out after {timeout_sec} seconds!"))
                self.output_queue.put(("stderr", f"TIMEOUT: Process exceeded {timeout_sec}s limit."))
                self.stop_process() # Ensure it's killed
                # Try to get any output produced so far
                # stdout_data, stderr_data = self.running_process.communicate() # This might hang? Risky.
                # self.output_queue.put(("stdout", stdout_data))
                # if stderr_data: self.output_queue.put(("stderr", stderr_data))
                self.output_queue.put(("run_complete", "TIMEOUT")) # Signal completion (with timeout)

            except Exception as e:
                 self.output_queue.put(("status", f"Error during execution: {e}"))
                 self.output_queue.put(("stderr", f"EXECUTION ERROR: {e}"))
                 self.output_queue.put(("run_complete", "ERROR")) # Signal completion (with error)


        finally:
            self.running_process = None # Clear the process handle

    def stop_process(self):
        if self.running_process:
            self.set_status("Attempting to stop process...")
            try:
                # Terminate politely first, then kill if necessary
                self.running_process.terminate()
                try:
                    # Wait a short time for termination
                    self.running_process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.set_status("Process did not terminate, killing...")
                    self.running_process.kill()
                self.set_status("Process stopped.")
                self.output_queue.put(("stderr", "\n--- PROCESS MANUALLY STOPPED ---"))

            except Exception as e:
                self.set_status(f"Error stopping process: {e}")
            finally:
                self.running_process = None
        else:
            self.set_status("No process running.")


    def check_queue(self):
            """ Check the queue for updates from the worker thread """
            try:
                while True: # Drain the queue completely in this polling cycle
                    msg_type, msg_data = self.output_queue.get_nowait()

                    if msg_type == "stdout":
                        self.output_text.insert(tk.END, msg_data)
                        self.output_text.see(tk.END) # Scroll to end
                    elif msg_type == "stderr":
                        # Ensure stderr messages are always visible in the result area
                        self.result_text.insert(tk.END, f"--- STDERR ---\n{msg_data}\n--------------\n")
                        self.result_text.see(tk.END)
                    elif msg_type == "status":
                        self.set_status(msg_data)
                    elif msg_type == "run_complete":
                        # --- This is the crucial signal that the process ended ---
                        self.set_status("Run finished. Starting checker...")

                        # Define the header message based on how the run completed
                        result_header = ""
                        if msg_data == "TIMEOUT":
                            result_header = "\n--- CHECKER RESULT (after Timeout) ---\n"
                        elif msg_data == "ERROR":
                            result_header = "\n--- CHECKER RESULT (after Error) ---\n"
                        # Note: Normal completion doesn't add a special header here,
                        # perform_check will add the standard "--- CHECKER RESULT ---"

                        # Insert the header *before* running the potentially slow check
                        if result_header:
                            self.result_text.insert(tk.END, result_header)
                            self.result_text.see(tk.END)

                        # Now, perform the correctness check
                        self.perform_check() # This updates status label itself upon completion

                        # Re-enable the run button now that everything is done
                        self.enable_run_button() # Need to implement this helper

                        # Since run is complete, no need to reschedule polling from here
                        # The thread is dead. Exit the check_queue cycle.
                        return # Stop polling

            except queue.Empty:
                # Queue is empty for this moment.
                # Check if the background thread is still working.
                if hasattr(self, 'thread') and self.thread.is_alive():
                    # If the thread is alive, schedule this check again
                    self.master.after(100, self.check_queue)
                # else:
                #    If the thread is dead, we assume it sent 'run_complete'
                #    or terminated unexpectedly. The loop above should have
                #    processed any final messages. We don't reschedule.
                #    If 'run_complete' wasn't received, something else went wrong.


    def perform_check(self):
        """ Reads input/output and runs the checker logic """
        input_data = self.input_text.get('1.0', tk.END).strip().splitlines()
        output_data = self.output_text.get('1.0', tk.END).strip().splitlines()

        if not input_data:
                self.result_text.insert(tk.END, "Checker Error: No input data found in text area.\n")
                self.set_status("Checker failed (no input).")
                return
        # Output might be empty if program crashes instantly
        # if not output_data:
        #      self.result_text.insert(tk.END, "Checker Warning: No output data found in text area.\n")
                # Continue checking, maybe final state check reveals errors


        self.set_status("Running checker...")
        checker = Checker()
        results = checker.check(output_data, input_data, self.is_hack_mode.get())

        self.result_text.insert(tk.END, "\n--- CHECKER RESULT ---\n")
        for line in results:
            self.result_text.insert(tk.END, line + "\n")
        self.result_text.see(tk.END)

        if "Failed" in results[0]:
                self.set_status("Check Failed.")
        elif "Passed" in results[0]:
                self.set_status("Check Passed.")
        else:
                self.set_status("Checker finished.") # Ambiguous result?


    def set_status(self, message):
        self.status_label.config(text=f"Status: {message}")

    # Add this method inside the CheckerApp class definition:
    def on_closing(self):
        """Handles the event when the user tries to close the window."""
        if self.running_process:
                # Ask for confirmation if a process is active
                if messagebox.askokcancel("Quit", "A test process is currently running. Stop it and quit?"):
                    self.stop_process() # Attempt to stop the process first
                    self.master.destroy() # Then destroy the window
                # else: If user clicks Cancel, do nothing - window stays open
        else:
                # No process running, close immediately
                self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CheckerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing) # Handle closing while running
    root.mainloop()