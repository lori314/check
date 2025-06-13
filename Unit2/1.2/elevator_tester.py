import pexpect
import time
import sys
import os
import re # Regular expressions for parsing output (optional but useful)

# --- Configuration ---
# IMPORTANT: Replace with the actual path to your JAR file!
ELEVATOR_JAR_PATH = "path/to/your/ElevatorSim.jar"
# Command to run the Java JAR file
JAVA_COMMAND = f"java -jar {ELEVATOR_JAR_PATH}"

# Maximum time to wait for the program to finish after last input (seconds)
PROCESS_TIMEOUT = 20
# Time to wait between checking for output when expecting input timing (seconds)
OUTPUT_CHECK_INTERVAL = 0.02 # Check frequently

# --- Test Case Definition ---
# List of tuples: (time_to_send, input_string)
# time_to_send is the simulation time (seconds) when the input should be sent.
# input_string is the exact command the Java program expects (without the evaluator's timestamp).
# The official I/O package handles timestamping the output based on *its* internal clock.
# Your script controls *when* to send the input.
test_inputs = [
    # Example based on HW6 sample (Timestamps control *when* python sends)
    (0.8, "SCHE-6-0.2-F1"),
    (1.2, "417-PRI-15-FROM-B2-TO-B4"),
    (5.1, "SCHE-3-0.4-B1"), # Note: HW PDF shows [5.1], we send at 5.1s
    # Add more test inputs based on your scenarios
    (6.0, "100-PRI-10-FROM-F1-TO-F7"),
    (6.5, "101-PRI-5-FROM-F3-TO-B1"),
    (15.0, "200-PRI-20-FROM-B4-TO-F5"),
    # Example: An input sent much later
    # (45.0, "999-PRI-1-FROM-F1-TO-F2"),
]

# --- Helper Function to Parse Timestamps from Output (Example) ---
def parse_output_timestamp(line):
    """Attempts to parse the [ timestamp ] prefix from a line."""
    match = re.match(r'^\s*\[\s*(\d+\.\d+)\s*\]', line)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

# --- Main Test Execution Function ---
def run_test(command, inputs):
    """Runs the elevator simulation and feeds it inputs based on timing."""

    if not os.path.exists(ELEVATOR_JAR_PATH) or "path/to/your" in ELEVATOR_JAR_PATH:
        print(f"ERROR: Elevator JAR file not found or path not updated: {ELEVATOR_JAR_PATH}", file=sys.stderr)
        print("Please update the ELEVATOR_JAR_PATH variable in the script.", file=sys.stderr)
        return False, []

    print(f"Starting process: {command}")
    # Spawn the process. Use encoding='utf-8' for text.
    # logfile=sys.stdout dumps all interaction to the console for debugging.
    try:
        child = pexpect.spawn(command, encoding='utf-8', timeout=10) # Default expect timeout
        # child.logfile_read = sys.stdout # Uncomment for verbose debugging
    except pexpect.exceptions.ExceptionPexpect as e:
        print(f"ERROR: Failed to spawn process: {e}", file=sys.stderr)
        print("Check if Java is installed and the JAR path is correct.", file=sys.stderr)
        return False, []

    start_real_time = time.time()
    current_input_index = 0
    all_output_lines = []
    last_input_sent_time = 0.0

    print("--- Test Started ---")

    try:
        while current_input_index < len(inputs) or child.isalive():
            now_real_time = time.time()
            elapsed_sim_time = now_real_time - start_real_time

            # 1. Check if it's time to send the next input
            next_input_to_send = None
            if current_input_index < len(inputs):
                target_send_time, input_str = inputs[current_input_index]
                if elapsed_sim_time >= target_send_time:
                    next_input_to_send = input_str
                    last_input_sent_time = elapsed_sim_time

            # 2. Send input if ready
            if next_input_to_send:
                print(f"[{elapsed_sim_time:.4f}s] >> Sending: {next_input_to_send}")
                child.sendline(next_input_to_send)
                current_input_index += 1
                # After sending, immediately try to read any quick response
                time.sleep(0.01) # Small delay to allow processing

            # 3. Read available output (non-blocking)
            # We do this frequently in the loop
            try:
                # Use expect with a very small timeout to check for output without blocking
                # Match a full line ending in \n or the EOF
                # Use a non-greedy match for the line content: r'(.*?)\r?\n'
                index = child.expect([r'(.*?)\r?\n', pexpect.EOF], timeout=OUTPUT_CHECK_INTERVAL)

                if index == 0: # Matched a line
                    line = child.match.group(1).strip() # Get the captured group, strip whitespace
                    output_sim_time = time.time() - start_real_time # Timestamp when received
                    if line: # Avoid printing empty lines if any occur
                        print(f"[{output_sim_time:.4f}s] << Received: {line}")
                        all_output_lines.append((output_sim_time, line))
                elif index == 1: # EOF reached
                    print(f"[{elapsed_sim_time:.4f}s] -- Process terminated (EOF detected).")
                    break # Exit the main loop

            except pexpect.TIMEOUT:
                # This is expected when no output is ready - just continue the loop
                pass
            except pexpect.EOF:
                 # Should be caught by index==1, but handle defensively
                 print(f"[{elapsed_sim_time:.4f}s] -- Process terminated (EOF Exception).")
                 break # Exit the main loop

            # 4. Check for overall timeout if all input is sent
            if current_input_index >= len(inputs):
                 if elapsed_sim_time - last_input_sent_time > PROCESS_TIMEOUT:
                      print(f"[{elapsed_sim_time:.4f}s] -- Timeout waiting for process to finish after last input.", file=sys.stderr)
                      child.terminate(force=True) # Force kill
                      break

            # 5. Small sleep to prevent hogging CPU if nothing happens
            if not next_input_to_send: # Only sleep if we weren't busy sending
                 time.sleep(OUTPUT_CHECK_INTERVAL / 2) # Sleep less than the check interval

    except pexpect.exceptions.ExceptionPexpect as e:
        print(f"\nERROR: pexpect exception during interaction: {e}", file=sys.stderr)
        return False, all_output_lines
    except Exception as e:
         print(f"\nERROR: Unexpected Python error during interaction: {e}", file=sys.stderr)
         return False, all_output_lines
    finally:
        if child.isalive():
            print("--- Forcing process termination ---")
            child.terminate(force=True)
        print("--- Test Interaction Finished ---")
        if child.exitstatus is not None:
             print(f"Process Exit Code: {child.exitstatus}")
        elif child.signalstatus is not None:
             print(f"Process Terminated by Signal: {child.signalstatus}")


    # Basic Validation Example (can be greatly expanded)
    print("\n--- Basic Validation ---")
    # Check if any output was produced
    if not all_output_lines:
        print("VALIDATION FAILED: No output received from the simulation.", file=sys.stderr)
        return False, all_output_lines

    # Check if timestamps in output are non-decreasing (a common requirement)
    last_parsed_ts = -1.0
    timestamps_ok = True
    for _, line in all_output_lines:
        parsed_ts = parse_output_timestamp(line)
        if parsed_ts is not None:
            if parsed_ts < last_parsed_ts:
                print(f"VALIDATION WARNING: Timestamp decreased: {parsed_ts} after {last_parsed_ts} in line: {line}", file=sys.stderr)
                timestamps_ok = False # Mark as warning or failure based on rules
            last_parsed_ts = parsed_ts

    if timestamps_ok:
         print("Timestamp non-decreasing check: PASSED (or no timestamps found)")

    # Add more checks here:
    # - Check for specific required outputs (e.g., OUT-S for every IN)
    # - Check timing constraints (e.g., door open duration)
    # - Check passenger delivery
    # - Check for error messages from the simulation

    print("------------------------")

    return True, all_output_lines

# --- Script Entry Point ---
if __name__ == "__main__":
    # Sort inputs by time just in case they weren't defined in order
    test_inputs.sort(key=lambda x: x[0])

    success, output = run_test(JAVA_COMMAND, test_inputs)

    print("\n--- Final Collected Output ---")
    for sim_time, line_text in output:
        print(f"[{sim_time:.4f}s] {line_text}")
    print("---------------------------\n")

    if success:
        print("SCRIPT: Test run completed. Further validation might be needed.")
        # sys.exit(0) # Exit with success code
    else:
        print("SCRIPT: Test run failed or encountered errors.", file=sys.stderr)
        # sys.exit(1) # Exit with failure code