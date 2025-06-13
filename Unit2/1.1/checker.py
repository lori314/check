# checker.py
import sys
import re
from decimal import Decimal, InvalidOperation
from collections import defaultdict, deque
from config import (
    floor_to_int, int_to_floor, INPUT_PATTERN, # OUTPUT_PATTERN needs replacement
    FLOORS, MIN_FLOOR, MAX_FLOOR, MOVE_TIME_PER_FLOOR, DOOR_OPEN_CLOSE_TIME,
    MAX_CAPACITY, NUM_ELEVATORS, EPSILON
)

# --- Revised Output Pattern ---
# Handles ARRIVE/OPEN/CLOSE: Action-Floor-ElevatorID
# Handles IN/OUT: Action-PassengerID-Floor-ElevatorID
OUTPUT_PATTERN = re.compile(
    r"\[\s*(\d+\.\d+)\s*\]"  # 1: Timestamp
    r"(?:"                     # Start non-capturing group for alternatives
        r"(ARRIVE|OPEN|CLOSE)" # 2: Action A/O/C
        r"-([BF]\d+)"          # 3: Floor for A/O/C
        r"-([1-6])"            # 4: Elevator ID for A/O/C
    r"|"                       # OR
        r"(IN|OUT)"            # 5: Action I/O
        r"-(\d+)"              # 6: Passenger ID for I/O
        r"-([BF]\d+)"          # 7: Floor for I/O
        r"-([1-6])"            # 8: Elevator ID for I/O
    r")"                       # End non-capturing group
)

class Passenger:
    # (Keep the Passenger class definition as before)
    def __init__(self, id, priority, from_floor, to_floor, assigned_elevator, request_time):
        self.id = id
        self.priority = priority
        self.from_floor = from_floor
        self.to_floor = to_floor
        self.assigned_elevator = assigned_elevator
        self.request_time = request_time
        self.state = "WAITING" # WAITING, INSIDE, ARRIVED
        self.location = from_floor

class Elevator:
    def __init__(self, id):
        self.id = id
        self.current_floor = 1 # Starts at F1
        self.door_state = "CLOSED" # CLOSED, OPEN
        # --- Timing and State Tracking ---
        self.last_action_finish_time = Decimal("-1.0") # Time the last relevant action (ARRIVE, OPEN, CLOSE) finished
        self.last_open_time = Decimal("-1.0")       # Specifically track when the door finished opening
        self.last_arrive_time = Decimal("-1.0")     # Specifically track when the elevator last arrived
        self.last_move_start_time = Decimal("-1.0") # Track when the last move (after CLOSE) started
        # --- ---
        self.passengers = set() # Set of passenger IDs inside

    def __repr__(self):
         return f"E[{self.id}]@{self.current_floor} {self.door_state} ({len(self.passengers)}ppl)"


class Checker:
    def __init__(self):
        self.errors = []
        self.last_output_time = Decimal("-1.0")
        self.elevators = {} # Will be populated in reset
        self.passengers = {} # Will be populated in reset
        self.raw_input_requests = []
        self.t_std = Decimal("0.0")
        self.t_max = Decimal("Infinity")
        self._reset() # Initialize state

    def _reset(self):
        self.errors = []
        self.last_output_time = Decimal("-1.0")
        self.elevators = {i: Elevator(i) for i in range(1, NUM_ELEVATORS + 1)}
        self.passengers = {}
        self.raw_input_requests = []
        self.t_std = Decimal("0.0")
        self.t_max = Decimal("Infinity")

    def _add_error(self, timestamp, message):
        # Avoid duplicate messages for the exact same condition at the same time
        error_msg = f"Error at T={timestamp:.4f}: {message}"
        if not self.errors or self.errors[-1] != error_msg:
             self.errors.append(error_msg)

    def _parse_input(self, input_lines):
        # (Keep _parse_input mostly as before, just ensure it clears self.passengers)
        self.passengers = {} # Clear previous passengers
        self.raw_input_requests = input_lines
        if not input_lines:
            self._add_error(Decimal("0.0"), "No input requests found.")
            return False
        last_time = Decimal("-1.0")
        has_input_errors = False
        for i, line in enumerate(input_lines):
            line = line.strip()
            if not line: continue
            match = INPUT_PATTERN.match(line)
            if not match:
                self._add_error(Decimal("0.0"), f"Input line {i+1} format error: '{line}'")
                has_input_errors = True
                continue

            try:
                t, pid, pri, f_from, f_to, eid = match.groups()
                timestamp = Decimal(t)
                passenger_id = int(pid)
                priority = int(pri) # Keep priority info if needed later
                from_floor = floor_to_int(f_from)
                to_floor = floor_to_int(f_to)
                elevator_id = int(eid)

                if timestamp < last_time - EPSILON: # Allow slightly decreasing due to precision? Better strict.
                     self._add_error(timestamp, f"Input timestamp decreased: {timestamp} < {last_time}")
                     has_input_errors = True
                last_time = timestamp

                if passenger_id in self.passengers:
                     self._add_error(timestamp, f"Duplicate passenger ID: {passenger_id}")
                     has_input_errors = True
                if from_floor == to_floor:
                    self._add_error(timestamp, f"Passenger {passenger_id}: From floor == To floor ({f_from})")
                    has_input_errors = True
                if from_floor not in FLOORS or to_floor not in FLOORS:
                    self._add_error(timestamp, f"Passenger {passenger_id}: Invalid floor ({f_from} or {f_to})")
                    has_input_errors = True
                if not (1 <= elevator_id <= NUM_ELEVATORS):
                     self._add_error(timestamp, f"Passenger {passenger_id}: Invalid elevator ID {elevator_id}")
                     has_input_errors = True
                # Priority check removed as it's for performance, not correctness here

                self.passengers[passenger_id] = Passenger(
                    passenger_id, priority, from_floor, to_floor, elevator_id, timestamp
                )
                self.t_std = max(self.t_std, timestamp)

            except (ValueError, InvalidOperation) as e:
                self._add_error(Decimal("0.0"), f"Input line {i+1} data error: '{line}' - {e}")
                has_input_errors = True

        return not has_input_errors


    def check(self, output_lines, input_lines, is_hack_data=False):
        self._reset() # Ensure clean state

        if not self._parse_input(input_lines):
            return ["Input check failed:"] + self.errors # Stop if input is broken

        self.t_max = Decimal("120.0") if is_hack_data else max(self.t_std + 10, self.t_std * Decimal("1.15"))

        current_sim_time = Decimal("0.0")

        for i, line in enumerate(output_lines):
            line = line.strip()
            if not line: continue

            match = OUTPUT_PATTERN.match(line)
            if not match:
                self._add_error(self.last_output_time, f"Output line {i+1} format error: '{line}'")
                continue # Skip this line, proceed with others

            try:
                groups = match.groups()
                timestamp = Decimal(groups[0])

                # --- Global Time Checks ---
                if timestamp < self.last_output_time - EPSILON:
                    self._add_error(timestamp, f"Timestamp decreased: {timestamp} < {self.last_output_time}")
                self.last_output_time = timestamp
                current_sim_time = timestamp

                if current_sim_time > self.t_max + EPSILON:
                    self._add_error(timestamp, f"Exceeded T_max ({self.t_max:.4f}): current time is {current_sim_time:.4f}")
                    # Don't stop, report other errors too

                # --- Dispatch based on matched groups ---
                action_aoc = groups[1]
                action_io = groups[4]

                if action_aoc: # Matched ARRIVE, OPEN, or CLOSE
                    action = action_aoc
                    floor_str = groups[2]
                    elevator_id = int(groups[3])
                    try:
                         floor = floor_to_int(floor_str)
                    except ValueError:
                         self._add_error(timestamp, f"Invalid floor string '{floor_str}' in {action}")
                         continue
                    if elevator_id not in self.elevators:
                        self._add_error(timestamp, f"Invalid elevator ID {elevator_id} in {action}")
                        continue
                    elevator = self.elevators[elevator_id]

                    if action == "ARRIVE":
                        self._check_arrive(timestamp, elevator, floor)
                    elif action == "OPEN":
                        self._check_open(timestamp, elevator, floor)
                    elif action == "CLOSE":
                        self._check_close(timestamp, elevator, floor)

                elif action_io: # Matched IN or OUT
                    action = action_io
                    passenger_id = int(groups[5])
                    floor_str = groups[6]
                    elevator_id = int(groups[7])
                    try:
                         floor = floor_to_int(floor_str)
                    except ValueError:
                         self._add_error(timestamp, f"Invalid floor string '{floor_str}' in {action}")
                         continue
                    if elevator_id not in self.elevators:
                        self._add_error(timestamp, f"Invalid elevator ID {elevator_id} in {action}")
                        continue
                    if passenger_id not in self.passengers:
                        self._add_error(timestamp, f"Unknown passenger ID {passenger_id} in {action}")
                        continue

                    elevator = self.elevators[elevator_id]
                    passenger = self.passengers[passenger_id]

                    if action == "IN":
                        self._check_in(timestamp, elevator, passenger, floor)
                    elif action == "OUT":
                        self._check_out(timestamp, elevator, passenger, floor)
                else:
                    # This case shouldn't happen if regex is correct and matches
                     self._add_error(timestamp, f"Internal Error: Regex matched but no action group found for line: '{line}'")

            except (ValueError, InvalidOperation, IndexError) as e:
                 self._add_error(self.last_output_time, f"Output line {i+1} processing error: '{line}' - {type(e).__name__}: {e}")
            except KeyError as e:
                 self._add_error(self.last_output_time, f"Output line {i+1}: Invalid ID lookup - {e} in '{line}'")


        # --- Final State Checks ---
        final_time = self.last_output_time
        all_passengers_arrived = True
        for pid, p in self.passengers.items():
            if p.state != "ARRIVED":
                all_passengers_arrived = False
                self._add_error(final_time, f"Passenger {pid} did not arrive at destination (state: {p.state}, location: {p.location}, dest: {p.to_floor})")

        for eid, e in self.elevators.items():
            if e.passengers:
                self._add_error(final_time, f"Elevator {eid} finished with passengers inside: {e.passengers}")
            if e.door_state != "CLOSED":
                 self._add_error(final_time, f"Elevator {eid} did not finish with doors closed (state: {e.door_state})")

        if not self.errors and not all_passengers_arrived:
             # If no other errors, but passengers didn't arrive, add a general failure message
             self._add_error(final_time, "Execution finished but not all passengers reached their destination.")

        if not self.errors:
             return ["Correctness Check Passed.", f"Finished at T={final_time:.4f} (T_max={self.t_max:.4f})"]
        else:
            # Return unique errors
            unique_errors = []
            seen_errors = set()
            for err in self.errors:
                 if err not in seen_errors:
                      unique_errors.append(err)
                      seen_errors.add(err)
            return ["Correctness Check Failed:"] + unique_errors


    def _check_arrive(self, timestamp, elevator, current_floor):
        if elevator.door_state != "CLOSED":
            self._add_error(timestamp, f"Elevator {elevator.id} ARRIVE while doors not CLOSED (state: {elevator.door_state})")
            return # Invalid state for arrival

        if current_floor not in FLOORS:
             self._add_error(timestamp, f"Elevator {elevator.id} ARRIVE at invalid floor: {current_floor}")
             return # Invalid floor

        # --- Time Check (>= 0.4s after last move started) ---
        # last_move_start_time should be updated on CLOSE
        if timestamp < elevator.last_move_start_time + MOVE_TIME_PER_FLOOR - EPSILON:
            self._add_error(timestamp, f"Elevator {elevator.id} ARRIVE too fast. Moved from {elevator.current_floor} to {current_floor}. "
                                      f"Move started ~{elevator.last_move_start_time:.4f}, expected >= {elevator.last_move_start_time + MOVE_TIME_PER_FLOOR:.4f}, got {timestamp:.4f}")

        # --- Floor Sequence Check ---
        floor_diff = current_floor - elevator.current_floor
        is_b1_f1_move = (elevator.current_floor == -1 and current_floor == 1) or \
                        (elevator.current_floor == 1 and current_floor == -1)

        if is_b1_f1_move:
            if abs(floor_diff) != 2: # Should be exactly 2 for B1<->F1
                 self._add_error(timestamp, f"Elevator {elevator.id} invalid floor sequence around B1/F1: {elevator.current_floor} -> {current_floor}")
        elif abs(floor_diff) != 1:
             self._add_error(timestamp, f"Elevator {elevator.id} moved incorrect number of floors: {elevator.current_floor} -> {current_floor} (diff: {floor_diff})")

        # Update state AFTER checks
        elevator.current_floor = current_floor
        elevator.last_arrive_time = timestamp
        # ARRIVE completion *is* the action finish time before potential OPEN
        elevator.last_action_finish_time = timestamp


    def _check_open(self, timestamp, elevator, floor):
        if elevator.current_floor != floor:
             self._add_error(timestamp, f"Elevator {elevator.id} OPEN at wrong floor. Is at {elevator.current_floor}, tried to open at {floor}")
             # Don't return, check other conditions too

        if elevator.door_state != "CLOSED":
            self._add_error(timestamp, f"Elevator {elevator.id} OPEN while not CLOSED (state: {elevator.door_state})")
            return # Invalid state transition

        # OPEN should happen at or after arrival at this floor
        if timestamp < elevator.last_arrive_time - EPSILON:
             self._add_error(timestamp, f"Elevator {elevator.id} OPEN before arriving at floor {floor}. Arrived at {elevator.last_arrive_time:.4f}, opened at {timestamp:.4f}")

        # Update state AFTER checks
        elevator.door_state = "OPEN"
        elevator.last_open_time = timestamp # Record when opening *finished* (assuming instant for now)
        elevator.last_action_finish_time = timestamp


    def _check_close(self, timestamp, elevator, floor):
        if elevator.current_floor != floor:
             self._add_error(timestamp, f"Elevator {elevator.id} CLOSE at wrong floor. Is at {elevator.current_floor}, tried to close at {floor}")
             # Don't return, check other conditions

        if elevator.door_state != "OPEN":
             self._add_error(timestamp, f"Elevator {elevator.id} CLOSE while not OPEN (state: {elevator.door_state})")
             return # Invalid state transition

        # --- Time Check (>= 0.4s after OPEN finished) ---
        if timestamp < elevator.last_open_time + DOOR_OPEN_CLOSE_TIME - EPSILON:
             self._add_error(timestamp, f"Elevator {elevator.id} CLOSE too fast. Doors must stay open >= {DOOR_OPEN_CLOSE_TIME}s. "
                                       f"Opened at {elevator.last_open_time:.4f}, tried to close at {timestamp:.4f}")

        # Update state AFTER checks
        elevator.door_state = "CLOSED"
        elevator.last_action_finish_time = timestamp
        # Record when the next move can potentially start
        elevator.last_move_start_time = timestamp


    def _check_in(self, timestamp, elevator, passenger, floor):
        if elevator.door_state != "OPEN":
             self._add_error(timestamp, f"Passenger {passenger.id} IN Elevator {elevator.id} while doors not OPEN (state: {elevator.door_state})")
             return # Cannot enter if doors not open

        if elevator.current_floor != floor:
             self._add_error(timestamp, f"Passenger {passenger.id} IN Elevator {elevator.id} at wrong floor. Elevator at {elevator.current_floor}, IN at {floor}")
             # Return because location check depends on correct elevator floor

        if passenger.state != "WAITING":
             self._add_error(timestamp, f"Passenger {passenger.id} IN Elevator {elevator.id} but not WAITING (state: {passenger.state})")
             # Don't return yet, check other things

        if passenger.location != floor:
             self._add_error(timestamp, f"Passenger {passenger.id} IN Elevator {elevator.id} at {floor}, but passenger location is {passenger.location}")
             # Can't enter if not at the right floor

        if passenger.assigned_elevator != elevator.id:
             self._add_error(timestamp, f"Passenger {passenger.id} IN Elevator {elevator.id}, but assigned to {passenger.assigned_elevator}")
             # Trying to enter wrong elevator

        # Check capacity *before* adding
        if len(elevator.passengers) >= MAX_CAPACITY:
             self._add_error(timestamp, f"Elevator {elevator.id} OVERLOAD. Cannot accept passenger {passenger.id}. Current occupants: {len(elevator.passengers)}")
             return # Cannot enter if full

        # Check timing (must be during OPEN state, i.e., >= last_open_time)
        if timestamp < elevator.last_open_time - EPSILON:
              self._add_error(timestamp, f"Passenger {passenger.id} IN Elevator {elevator.id} before doors finished opening. Doors opened at {elevator.last_open_time:.4f}, IN at {timestamp:.4f}")

        # Update state only if checks pass (especially state and location)
        if passenger.state == "WAITING" and passenger.location == floor:
            passenger.state = "INSIDE"
            passenger.location = None
            elevator.passengers.add(passenger.id)
        # Note: IN/OUT don't update elevator's last_action_finish_time or last_move_start_time


    def _check_out(self, timestamp, elevator, passenger, floor):
        if elevator.door_state != "OPEN":
             self._add_error(timestamp, f"Passenger {passenger.id} OUT Elevator {elevator.id} while doors not OPEN (state: {elevator.door_state})")
             return # Cannot exit if doors not open

        if elevator.current_floor != floor:
             self._add_error(timestamp, f"Passenger {passenger.id} OUT Elevator {elevator.id} at wrong floor. Elevator at {elevator.current_floor}, OUT at {floor}")
             return # Cannot exit at wrong floor

        if passenger.state != "INSIDE":
             self._add_error(timestamp, f"Passenger {passenger.id} OUT Elevator {elevator.id} but not INSIDE (state: {passenger.state})")
             # Might be redundant if already arrived, but check anyway

        if passenger.id not in elevator.passengers:
              self._add_error(timestamp, f"Passenger {passenger.id} OUT Elevator {elevator.id}, but passenger was not recorded inside {elevator.passengers}")
              return # Cannot exit if not inside

        if floor != passenger.to_floor:
             self._add_error(timestamp, f"Passenger {passenger.id} OUT Elevator {elevator.id} at floor {floor}, but destination is {passenger.to_floor}")
             # Exiting at wrong destination

        # Check timing (must be during OPEN state, i.e., >= last_open_time)
        if timestamp < elevator.last_open_time - EPSILON:
              self._add_error(timestamp, f"Passenger {passenger.id} OUT Elevator {elevator.id} before doors finished opening. Doors opened at {elevator.last_open_time:.4f}, OUT at {timestamp:.4f}")

        # Update state only if checks pass
        if passenger.state == "INSIDE" and passenger.id in elevator.passengers and floor == passenger.to_floor:
            passenger.state = "ARRIVED"
            passenger.location = floor
            elevator.passengers.remove(passenger.id)
        # Note: IN/OUT don't update elevator's times