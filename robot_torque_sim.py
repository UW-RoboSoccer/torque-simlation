"""
Robot Torque Simulation

This module provides a simulation framework for calculating maximum static torques
in robotic joints, considering the mass and length of segments and their
hierarchical structure. It allows for defining different scenarios with varying
segment properties and outputs the results in a CSV file.

Usage:
    python robot_torque_sim.py [options]

Options:
    -h, --help          Show this help message and exit
    -v, --version       Show version number and exit
    -o, --output FILE   Specify output CSV file (default: results.csv)
    -s, --scenario NAME Run simulation for specific scenario (default: all)
"""

import math
import copy
import argparse
import csv

# --- Constants ---
G = 9.81  # Acceleration due to gravity (m/s^2)

# --- Robot Definition ---

# Define the segments (links) of the robot
# Keys: Unique segment names
# Values: Dictionary with 'mass' (kg) and 'length' (m)
# Adjust these values for your specific robot design range
BASE_SEGMENT_DEFINITIONS = {
    # Torso/Pelvis - Often the root or split reference
    "Pelvis": {"mass": 10.0, "length": 0.3},  # Acts as base for legs/upper body
    "Torso": {"mass": 15.0, "length": 0.5},  # Connects to Pelvis

    # Head
    "Head": {"mass": 4.0, "length": 0.25},  # Connects to Torso

    # Arms (L/R) - Assuming symmetry for base values
    "UpperArm_L": {"mass": 2.5, "length": 0.3},
    "Forearm_L": {"mass": 1.8, "length": 0.28},  # Includes hand mass estimate
    "UpperArm_R": {"mass": 2.5, "length": 0.3},
    "Forearm_R": {"mass": 1.8, "length": 0.28},  # Includes hand mass estimate

    # Legs (L/R) - Assuming symmetry for base values
    "Thigh_L": {"mass": 8.0, "length": 0.45},
    "Shin_L": {"mass": 4.5, "length": 0.4},  # Includes foot mass estimate
    "Foot_L": {"mass": 1.5, "length": 0.2},  # Foot length often more about contact, but needed for CoM calc if separate
    "Thigh_R": {"mass": 8.0, "length": 0.45},
    "Shin_R": {"mass": 4.5, "length": 0.4},  # Includes foot mass estimate
    "Foot_R": {"mass": 1.5, "length": 0.2},
}

# Define the Joint Hierarchy and DOF
# Keys: Unique joint names
# Values: Dictionary with 'parent_segment' and 'child_segment'
# This defines the kinematic chain structure.
JOINT_HIERARCHY = {
    # Torso -> Head connection
    "Neck_Yaw": {"parent_segment": "Torso", "child_segment": "Head"},
    "Neck_Pitch": {"parent_segment": "Torso", "child_segment": "Head"},  # Shares segments, analysis is per-joint axis

    # Torso -> Arms connections
    "Shoulder_Pitch_L": {"parent_segment": "Torso", "child_segment": "UpperArm_L"},
    "Shoulder_Roll_L": {"parent_segment": "Torso", "child_segment": "UpperArm_L"},
    "Shoulder_Yaw_L": {"parent_segment": "Torso", "child_segment": "UpperArm_L"},
    "Elbow_Pitch_L": {"parent_segment": "UpperArm_L", "child_segment": "Forearm_L"},

    "Shoulder_Pitch_R": {"parent_segment": "Torso", "child_segment": "UpperArm_R"},
    "Shoulder_Roll_R": {"parent_segment": "Torso", "child_segment": "UpperArm_R"},
    "Shoulder_Yaw_R": {"parent_segment": "Torso", "child_segment": "UpperArm_R"},
    "Elbow_Pitch_R": {"parent_segment": "UpperArm_R", "child_segment": "Forearm_R"},

    # Pelvis -> Torso connection
    "Waist_Joint": {"parent_segment": "Pelvis", "child_segment": "Torso"},  # Simplification

    # Pelvis -> Legs connections
    "Hip_Yaw_L": {"parent_segment": "Pelvis", "child_segment": "Thigh_L"},
    "Hip_Pitch_L": {"parent_segment": "Pelvis", "child_segment": "Thigh_L"},
    "Hip_Roll_L": {"parent_segment": "Pelvis", "child_segment": "Thigh_L"},
    "Knee_Pitch_L": {"parent_segment": "Thigh_L", "child_segment": "Shin_L"},
    "Ankle_Pitch_L": {"parent_segment": "Shin_L", "child_segment": "Foot_L"},
    "Ankle_Roll_L": {"parent_segment": "Shin_L", "child_segment": "Foot_L"},

    "Hip_Yaw_R": {"parent_segment": "Pelvis", "child_segment": "Thigh_R"},
    "Hip_Pitch_R": {"parent_segment": "Pelvis", "child_segment": "Thigh_R"},
    "Hip_Roll_R": {"parent_segment": "Pelvis", "child_segment": "Thigh_R"},
    "Knee_Pitch_R": {"parent_segment": "Thigh_R", "child_segment": "Shin_R"},
    "Ankle_Pitch_R": {"parent_segment": "Shin_R", "child_segment": "Foot_R"},
    "Ankle_Roll_R": {"parent_segment": "Shin_R", "child_segment": "Foot_R"},
}

# --- Classes and Calculation Functions ---

from typing import Dict, List, Optional, Set

class Robot:
    """
    Represents a robot with segments and joint hierarchy.
    """

    def __init__(self, segments: Dict[str, Dict[str, float]], joint_hierarchy: Dict[str, Dict[str, str]]):
        self.segments = segments
        self.joint_hierarchy = joint_hierarchy

    def get_distal_segments(self, start_segment: str, _depth: int = 0, _visited_recursion: Optional[Set[str]] = None) -> List[str]:
        """
        Recursively find all segments distal to (further down the chain from) the start_segment.
        """
        if _visited_recursion is None:
            _visited_recursion = set()
        if start_segment in _visited_recursion:
            print(f"  {'  ' * _depth}ERROR: Cycle detected! Already visited '{start_segment}' in this recursion path. Aborting branch.")
            return []
        if _depth > len(self.segments) * 2:
            print(f"  {'  ' * _depth}ERROR: Max recursion depth exceeded for '{start_segment}'. Check hierarchy for cycles or extreme depth.")
            return []
        _visited_recursion.add(start_segment)
        distal_segments_set = set()
        for joint_name, links in self.joint_hierarchy.items():
            if links['parent_segment'] == start_segment:
                child = links['child_segment']
                distal_segments_set.add(child)
                distal_segments_set.update(self.get_distal_segments(child, _depth=_depth + 1, _visited_recursion=_visited_recursion.copy()))
        return list(distal_segments_set)

    def calculate_max_static_torques(self) -> Dict[str, float]:
        """
        Calculates the maximum static holding torque for each joint.
        Assumes worst-case horizontal extension of all distal segments.
        """
        max_torques = {}
        for joint_name, links in self.joint_hierarchy.items():
            parent = links['parent_segment']
            child = links['child_segment']
            distal_segments = self.get_distal_segments(child)
            # The torque at this joint is the sum of torques from all distal segments
            torque = 0.0
            for segment in [child] + distal_segments:
                seg_def = self.segments.get(segment)
                if seg_def is None:
                    print(f"  WARNING: Segment '{segment}' not found in definitions.")
                    continue
                mass = seg_def['mass']
                length = seg_def['length']
                # Assume center of mass at half the segment length
                torque += mass * G * length / 2
            max_torques[joint_name] = torque
        return max_torques

class Simulation:
    """
    Handles running torque simulations for multiple robot configurations.
    """

    def __init__(self, base_segments: Dict[str, Dict[str, float]], joint_hierarchy: Dict[str, Dict[str, str]], variations: Dict[str, Dict[str, Dict[str, float]]]):
        self.base_segments = base_segments
        self.joint_hierarchy = joint_hierarchy
        self.variations = variations
        self.results = {}

    def run(self, scenario: Optional[str] = None):
        scenarios = [scenario] if scenario else list(self.variations.keys())
        for name in scenarios:
            print(f"\nRunning simulation for variation: {name}")
            variation = self.variations[name]
            # Deep copy base segments and apply variation
            current_segments = copy.deepcopy(self.base_segments)
            for seg, props in variation.items():
                if seg not in current_segments:
                    print(f"  WARNING: Segment '{seg}' not found in base definitions.")
                    continue
                current_segments[seg].update(props)
            robot = Robot(current_segments, self.joint_hierarchy)
            max_torques = robot.calculate_max_static_torques()
            self.results[name] = max_torques
            print(f"  Finished torque calculation for variation '{name}'.")

    def save_results_csv(self, filename: str):
        """
        Save simulation results to a CSV file.
        """
        all_joints = set()
        for torques in self.results.values():
            all_joints.update(torques.keys())
        all_joints = sorted(all_joints)
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['Scenario'] + all_joints
            writer.writerow(header)
            for scenario, torques in self.results.items():
                row = [scenario] + [f"{torques.get(j, ''):.2f}" if j in torques else '' for j in all_joints]
                writer.writerow(row)

        segments_to_consider = [child_segment] + distal_segments_list
        # Ensure all segments are valid before proceeding (should be guaranteed by get_distal_segments check now)
        segments_to_consider = list(set([s for s in segments_to_consider if s in segment_definitions])) # Use set for uniqueness, then list
        print(f"        DEBUG: Full list of unique segments contributing torque: {segments_to_consider}") # DEBUG

        total_torque = 0.0

        # --- Loop through contributing segments ---
        for distal_idx, distal_seg_name in enumerate(segments_to_consider):
            print(f"        DEBUG: --- Calculating contribution of segment {distal_idx+1}/{len(segments_to_consider)}: '{distal_seg_name}' ---") # DEBUG
            if distal_seg_name not in segment_definitions:
                 print(f"        ERROR: Distal segment '{distal_seg_name}' is in the consider list but not in definitions. Skipping.")
                 continue # Should not happen if filtering above works
            segment_props = segment_definitions[distal_seg_name]

            # --- Path finding using BFS ---
            print(f"          DEBUG: Starting BFS to find path from '{child_segment}' to '{distal_seg_name}'.") # DEBUG
            visited_bfs = set()
            # Queue stores tuples: (current_segment_name, path_list_to_current)
            queue_bfs = [(child_segment, [child_segment])]
            path_to_distal = []
            bfs_steps = 0
            max_bfs_steps = len(segment_definitions) * 5 # Generous limit based on segment count

            while queue_bfs:
                bfs_steps += 1
                if bfs_steps > max_bfs_steps:
                    print(f"          ERROR: BFS exceeded max steps ({max_bfs_steps}) searching for '{distal_seg_name}'. Aborting BFS. Check hierarchy.")
                    path_to_distal = [] # Mark as failed
                    break # Exit BFS while loop

                # --- BFS DEQUEUE ---
                # Check if queue is empty before popping
                if not queue_bfs:
                     print(f"          ERROR: BFS queue became empty before finding '{distal_seg_name}'. Path not found.")
                     path_to_distal = []
                     break
                current_node, path = queue_bfs.pop(0) # FIFO for BFS
                # print(f"          DEBUG BFS Step {bfs_steps}: Dequeued '{current_node}', Path: {'->'.join(path)}") # Optional Verbose BFS trace

                # --- Target Found ---
                if current_node == distal_seg_name:
                    path_to_distal = path
                    print(f"          DEBUG: BFS Success! Path found in {bfs_steps} steps: {' -> '.join(path_to_distal)}") # DEBUG
                    break # Exit BFS while loop

                # --- Cycle/Visited Check ---
                # Check if visited *before* finding children to optimize
                if current_node in visited_bfs:
                    # print(f"          DEBUG BFS: Node '{current_node}' already visited. Skipping.") # Optional Verbose BFS trace
                    continue
                visited_bfs.add(current_node)

                # --- Find Children for BFS ---
                children_of_current = []
                # print(f"          DEBUG BFS: Finding children of '{current_node}'") # Optional Verbose BFS trace
                for j_name_bfs, j_links_bfs in joint_hierarchy.items():
                     if j_links_bfs['parent_segment'] == current_node:
                         child_seg_name_bfs = j_links_bfs['child_segment']
                         # Add child if it's valid and hasn't been visited/processed yet
                         if child_seg_name_bfs in segment_definitions:
                             # Check if already visited before adding to children list
                             if child_seg_name_bfs not in visited_bfs:
                                # Avoid adding duplicates to the list of children *to be processed* (unique needed for next step)
                                if child_seg_name_bfs not in children_of_current:
                                     children_of_current.append(child_seg_name_bfs)
                                     # print(f"            Found valid child: {child_seg_name_bfs}") # Optional Verbose BFS trace
                         # else: # Optional: Warn if child in hierarchy isn't in definitions
                         #    print(f"          WARNING BFS: Child '{child_seg_name_bfs}' of '{current_node}' not in segment_definitions.")


                # --- Enqueue Children ---
                # print(f"          DEBUG BFS: Enqueuing children: {children_of_current}") # Optional Verbose BFS trace
                for child in children_of_current:
                    # Check again if visited before enqueueing (belt-and-suspenders for complex graphs, maybe overkill here)
                    if child not in visited_bfs:
                        new_path = list(path) # Create a new path list for this child
                        new_path.append(child)
                        queue_bfs.append((child, new_path))
                        # print(f"            Enqueued '{child}' with path: {'->'.join(new_path)}") # Optional Verbose BFS trace

            # --- Check if BFS failed ---
            if not path_to_distal:
                 print(f"        ERROR: BFS FAILED to find path from '{child_segment}' to '{distal_seg_name}' for joint '{joint_name}'. Skipping torque contribution.")
                 continue # Skip this segment's contribution

            # --- Calculate Lever Arm based on found path ---
            current_lever = 0.0
            path_str = " -> ".join(path_to_distal) # DEBUG
            print(f"          DEBUG: Calculating lever arm for path: {path_str}") # DEBUG
            for i_path, seg_name_in_path in enumerate(path_to_distal):
                # Check segment exists before accessing length (should always exist if path found correctly)
                if seg_name_in_path not in segment_definitions:
                    print(f"          ERROR: Segment '{seg_name_in_path}' from BFS path not in definitions! Skipping its length.")
                    continue

                seg_props_in_path = segment_definitions[seg_name_in_path]
                segment_length = seg_props_in_path['length']

                if seg_name_in_path == distal_seg_name:
                    # It's the target segment, add distance to its CoM (L/2)
                    current_lever += segment_length / 2.0
                    print(f"            + {segment_length / 2.0:.3f} (CoM of target {seg_name_in_path})") # DEBUG
                else:
                    # It's an intermediate segment connecting joint to target, add its full length
                    current_lever += segment_length
                    print(f"            + {segment_length:.3f} (Full length of intermediate {seg_name_in_path})") # DEBUG

            # --- Calculate Torque Contribution ---
            mass = segment_props['mass']
            force = mass * G
            torque_contribution = force * current_lever
            total_torque += torque_contribution
            print(f"          DEBUG: Segment '{distal_seg_name}': Mass={mass:.2f}kg, LeverArm={current_lever:.3f}m -> Torque Contrib={torque_contribution:.2f} Nm") # DEBUG

        # --- Store final torque for the joint ---
        joint_torques[joint_name] = total_torque
        print(f"      DEBUG: === Finished Joint {joint_name}. Total Max Static Torque = {total_torque:.2f} Nm ===") # DEBUG
        processed_joints_count += 1 # DEBUG

    print(f"\n    DEBUG: Finished calculate_max_static_torques. Processed {processed_joints_count} joints.") # DEBUG
    if processed_joints_count != len(joint_hierarchy):
        print(f"    WARNING: Processed {processed_joints_count} joints, but {len(joint_hierarchy)} defined in hierarchy.")
    return joint_torques


def run_simulation_range(base_segments, joint_hierarchy, variations):
    """ Runs the torque calculation for different segment property variations. """
    results = {} # Store results keyed by variation name

    for name, changes in variations.items():
        print(f"\n--- Running Simulation: {name} ---")
        current_segments = copy.deepcopy(base_segments) # Start with base values

        # Apply changes for this variation
        if changes: # Apply changes only if the 'changes' dictionary is not empty
            print(f"  Applying changes for variation '{name}':")
            for segment, props_to_change in changes.items():
                if segment in current_segments:
                    print(f"    Modifying segment '{segment}':")
                    for prop, value in props_to_change.items():
                        if prop in current_segments[segment]:
                            print(f"      Setting {prop} = {value} (was {current_segments[segment][prop]})")
                            current_segments[segment][prop] = value
                        else:
                            print(f"    WARNING: Property '{prop}' not found for segment '{segment}' in base definition for variation '{name}'.")
                else:
                    print(f"    WARNING: Segment '{segment}' specified in variation '{name}' not found in base definition.")
        else:
            print("  Using base segment definitions (no changes).")

        # Calculate torques for this configuration
        print("  Starting torque calculation...")
        max_torques = calculate_max_static_torques(current_segments, joint_hierarchy)
        results[name] = max_torques
        print(f"  Finished torque calculation for variation '{name}'.")

        # Print results for this variation
        print("\nCalculated Maximum Static Torques (Nm) for this variation:")
        # Sort by joint name for consistent output
        sorted_joints = sorted(max_torques.keys())
        for joint in sorted_joints:
            print(f"  {joint:<18}: {max_torques[joint]:.2f}")

    return results


# --- Simulation Setup ---

# Define different scenarios (variations from the base)
# Format: { "Variation Name": {"Segment Name": {"property": new_value, ...}, ...} }
SIMULATION_VARIATIONS = {
    "BaseCase": {
        # No changes, use BASE_SEGMENT_DEFINITIONS directly
    },
    "HeavierLimbs": {
        "UpperArm_L": {"mass": 3.5}, "Forearm_L": {"mass": 2.5},
        "UpperArm_R": {"mass": 3.5}, "Forearm_R": {"mass": 2.5},
        "Thigh_L": {"mass": 10.0}, "Shin_L": {"mass": 5.5}, "Foot_L": {"mass": 2.0},
        "Thigh_R": {"mass": 10.0}, "Shin_R": {"mass": 5.5}, "Foot_R": {"mass": 2.0},
    },
    "LongerLimbs": {
         "UpperArm_L": {"length": 0.35}, "Forearm_L": {"length": 0.32},
         "UpperArm_R": {"length": 0.35}, "Forearm_R": {"length": 0.32},
         "Thigh_L": {"length": 0.50}, "Shin_L": {"length": 0.45}, "Foot_L": {"length": 0.22},
         "Thigh_R": {"length": 0.50}, "Shin_R": {"length": 0.45}, "Foot_R": {"length": 0.22},
    },
     "HeavierAndLonger": {
        "UpperArm_L": {"mass": 3.5, "length": 0.35}, "Forearm_L": {"mass": 2.5, "length": 0.32},
        "UpperArm_R": {"mass": 3.5, "length": 0.35}, "Forearm_R": {"mass": 2.5, "length": 0.32},
        "Thigh_L": {"mass": 10.0, "length": 0.50}, "Shin_L": {"mass": 5.5, "length": 0.45}, "Foot_L": {"mass": 2.0, "length": 0.22},
        "Thigh_R": {"mass": 10.0, "length": 0.50}, "Shin_R": {"mass": 5.5, "length": 0.45}, "Foot_R": {"mass": 2.0, "length": 0.22},
    },
    # Add more variations as needed (e.g., "LightweightDesign", "PayloadCarry")
    # Example for payload: Increase Forearm/Hand mass
     "CarryingPayload_5kg_Left": {
        # Add payload mass to the segment representing the hand/forearm
        "Forearm_L": {"mass": BASE_SEGMENT_DEFINITIONS["Forearm_L"]["mass"] + 5.0},
     }
}


# --- CLI and Main Execution ---
def main():
    parser = argparse.ArgumentParser(description='Robot Joint Torque Simulation')
    parser.add_argument('--scenario', type=str, help='Name of the scenario to run (default: all)')
    parser.add_argument('--csv', type=str, help='Output CSV file for results (optional)')
    args = parser.parse_args()

    print("Starting Simulation Runs...")
    sim = Simulation(BASE_SEGMENT_DEFINITIONS, JOINT_HIERARCHY, SIMULATION_VARIATIONS)
    sim.run(args.scenario)
    print("\nFinished All Simulation Runs.")

    # --- Post-Processing (Torque Ranges) ---
    print("\n--- Torque Ranges Across Simulations ---")
    all_torques_by_joint = {joint: [] for joint in JOINT_HIERARCHY.keys()}
    for variation_name, torques in sim.results.items():
        for joint_name, torque_value in torques.items():
            if joint_name in all_torques_by_joint:
                all_torques_by_joint[joint_name].append(torque_value)
            else:
                print(f"WARNING: Joint '{joint_name}' found in results of '{variation_name}' but not in base hierarchy keys during range calculation. Adding it.")
                all_torques_by_joint[joint_name] = [torque_value]
    print("\nMax Static Torque Ranges (Min - Max) Nm:")
    sorted_joints_final = sorted(all_torques_by_joint.keys())
    for joint_name in sorted_joints_final:
        torques_list = all_torques_by_joint[joint_name]
        if torques_list:
            min_torque = min(torques_list)
            max_torque = max(torques_list)
            print(f"  {joint_name:<18}: {min_torque:.2f} - {max_torque:.2f}")
        else:
            print(f"  {joint_name:<18}: No valid data calculated across variations.")
    print("\nSimulation Complete.")
    if args.csv:
        sim.save_results_csv(args.csv)
        print(f"Results saved to {args.csv}")

if __name__ == "__main__":
    main()