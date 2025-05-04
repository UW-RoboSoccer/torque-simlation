import math
import copy # To avoid modifying original definitions

# --- Constants ---
G = 9.81  # Acceleration due to gravity (m/s^2)

# --- Robot Definition ---

# Define the segments (links) of the robot
# Keys: Unique segment names
# Values: Dictionary with 'mass' (kg) and 'length' (m)
# Adjust these values for your specific robot design range
BASE_SEGMENT_DEFINITIONS = {
    # Torso/Pelvis - Often the root or split reference
    "Pelvis": {"mass": 10.0, "length": 0.3}, # Acts as base for legs/upper body
    "Torso":  {"mass": 15.0, "length": 0.5}, # Connects to Pelvis

    # Head
    "Head":   {"mass": 4.0, "length": 0.25}, # Connects to Torso

    # Arms (L/R) - Assuming symmetry for base values
    "UpperArm_L": {"mass": 2.5, "length": 0.3},
    "Forearm_L":  {"mass": 1.8, "length": 0.28}, # Includes hand mass estimate
    "UpperArm_R": {"mass": 2.5, "length": 0.3},
    "Forearm_R":  {"mass": 1.8, "length": 0.28}, # Includes hand mass estimate

    # Legs (L/R) - Assuming symmetry for base values
    "Thigh_L": {"mass": 8.0, "length": 0.45},
    "Shin_L":  {"mass": 4.5, "length": 0.4}, # Includes foot mass estimate
    "Foot_L": {"mass": 1.5, "length": 0.2}, # Foot length often more about contact, but needed for CoM calc if separate
    "Thigh_R": {"mass": 8.0, "length": 0.45},
    "Shin_R":  {"mass": 4.5, "length": 0.4}, # Includes foot mass estimate
    "Foot_R": {"mass": 1.5, "length": 0.2},
}

# Define the Joint Hierarchy and DOF
# Keys: Unique joint names
# Values: Dictionary with 'parent_segment' and 'child_segment'
# This defines the kinematic chain structure.
JOINT_HIERARCHY = {
    # Torso -> Head connection
    "Neck_Yaw":   {"parent_segment": "Torso", "child_segment": "Head"},
    "Neck_Pitch": {"parent_segment": "Torso", "child_segment": "Head"}, # Shares segments, analysis is per-joint axis

    # Torso -> Arms connections
    "Shoulder_Pitch_L": {"parent_segment": "Torso", "child_segment": "UpperArm_L"},
    "Shoulder_Roll_L":  {"parent_segment": "Torso", "child_segment": "UpperArm_L"},
    "Shoulder_Yaw_L":   {"parent_segment": "Torso", "child_segment": "UpperArm_L"},
    "Elbow_Pitch_L":    {"parent_segment": "UpperArm_L", "child_segment": "Forearm_L"},

    "Shoulder_Pitch_R": {"parent_segment": "Torso", "child_segment": "UpperArm_R"},
    "Shoulder_Roll_R":  {"parent_segment": "Torso", "child_segment": "UpperArm_R"},
    "Shoulder_Yaw_R":   {"parent_segment": "Torso", "child_segment": "UpperArm_R"},
    "Elbow_Pitch_R":    {"parent_segment": "UpperArm_R", "child_segment": "Forearm_R"},

    # Pelvis -> Torso connection
    "Waist_Joint":      {"parent_segment": "Pelvis", "child_segment": "Torso"}, # Simplification

    # Pelvis -> Legs connections
    "Hip_Yaw_L":    {"parent_segment": "Pelvis", "child_segment": "Thigh_L"},
    "Hip_Pitch_L":  {"parent_segment": "Pelvis", "child_segment": "Thigh_L"},
    "Hip_Roll_L":   {"parent_segment": "Pelvis", "child_segment": "Thigh_L"},
    "Knee_Pitch_L": {"parent_segment": "Thigh_L", "child_segment": "Shin_L"},
    "Ankle_Pitch_L":{"parent_segment": "Shin_L", "child_segment": "Foot_L"},
    "Ankle_Roll_L": {"parent_segment": "Shin_L", "child_segment": "Foot_L"},

    "Hip_Yaw_R":    {"parent_segment": "Pelvis", "child_segment": "Thigh_R"},
    "Hip_Pitch_R":  {"parent_segment": "Pelvis", "child_segment": "Thigh_R"},
    "Hip_Roll_R":   {"parent_segment": "Pelvis", "child_segment": "Thigh_R"},
    "Knee_Pitch_R": {"parent_segment": "Thigh_R", "child_segment": "Shin_R"},
    "Ankle_Pitch_R":{"parent_segment": "Shin_R", "child_segment": "Foot_R"},
    "Ankle_Roll_R": {"parent_segment": "Shin_R", "child_segment": "Foot_R"},
}


# --- Calculation Functions ---

# MODIFIED: Added depth parameter and print statements for debugging recursion
def get_distal_segments(start_segment, segments_def, joint_hier, _depth=0, _visited_recursion=None):
    """ Recursively find all segments distal to (further down the chain from) the start_segment. """
    # Initialize visited set at the top level call
    if _visited_recursion is None:
        _visited_recursion = set()

    # print(f"{'  '*_depth}DEBUG: get_distal_segments(start='{start_segment}', depth={_depth})") # Optional: Verbose trace

    # --- Cycle Detection ---
    if start_segment in _visited_recursion:
        print(f"{'  '*_depth}ERROR: Cycle detected! Already visited '{start_segment}' in this recursion path. Aborting branch.")
        return [] # Return empty list to prevent infinite loop
    if _depth > len(segments_def) * 2: # Safety break for excessive depth
        print(f"{'  '*_depth}ERROR: Max recursion depth exceeded for '{start_segment}'. Check hierarchy for cycles or extreme depth.")
        return []

    _visited_recursion.add(start_segment) # Mark current segment as visited for this path

    distal_segments_set = set()
    child_joints_segments = []
    for joint_name, links in joint_hier.items():
        if links['parent_segment'] == start_segment:
            # Ensure child exists before adding
            if links['child_segment'] in segments_def:
                child_joints_segments.append(links['child_segment'])
            # else: # Optional: Warn about undefined child segments in hierarchy
            #     print(f"WARNING: Child segment '{links['child_segment']}' for parent '{start_segment}' (joint '{joint_name}') not in segment definitions.")

    # Find unique children for this segment
    unique_child_segments = list(set(child_joints_segments))

    for child_seg in unique_child_segments:
        # Add the direct child
        distal_segments_set.add(child_seg)
        # Recursively find segments distal to this child
        # Pass a copy of the visited set for the recursive call specific to this branch
        recursive_distal = get_distal_segments(child_seg, segments_def, joint_hier, _depth + 1, _visited_recursion.copy())
        distal_segments_set.update(recursive_distal)

    # Remove current segment from visited *after* exploring children (backtrack)
    # Note: This might not be strictly necessary if we pass copies, but good practice.
    # We passed copies, so modifying the original set isn't needed here.

    # print(f"{'  '*_depth}DEBUG: Returning for '{start_segment}': {list(distal_segments_set)}") # Optional: Verbose trace
    return list(distal_segments_set)

# MODIFIED: Added extensive print statements for debugging progress and potential hangs
def calculate_max_static_torques(segment_definitions, joint_hierarchy):
    """
    Calculates the maximum static holding torque for each joint.
    Assumes worst-case horizontal extension of all distal segments.
    """
    print("    DEBUG: Entered calculate_max_static_torques.") # DEBUG
    joint_torques = {}
    processed_joints_count = 0 # DEBUG
    joint_names_list = list(joint_hierarchy.keys()) # Get a list to track progress

    for i, joint_name in enumerate(joint_names_list):
        links = joint_hierarchy[joint_name]
        print(f"\n      DEBUG: === Processing Joint {i+1}/{len(joint_names_list)}: {joint_name} ===") # DEBUG
        parent_segment = links['parent_segment']
        child_segment = links['child_segment']

        # --- Basic Sanity Checks ---
        # Check if child segment exists (Critical!)
        if child_segment not in segment_definitions:
             print(f"      ERROR: Child segment '{child_segment}' for joint '{joint_name}' not in segment definitions. Skipping this joint.")
             joint_torques[joint_name] = 0.0 # Assign 0 or handle error appropriately
             continue # Skip this joint calculation

        print(f"        DEBUG: Parent='{parent_segment}', Child='{child_segment}'. Finding distal segments for child.") # DEBUG
        # Call the modified get_distal_segments
        distal_segments_list = get_distal_segments(child_segment, segment_definitions, joint_hierarchy)
        print(f"        DEBUG: Distal segments (relative to child '{child_segment}') found: {distal_segments_list}") # DEBUG

        # Segments whose CoM contributes to torque at this joint (child + all distal to child)
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


# --- Run Simulation ---
print("Starting Simulation Runs...")
simulation_results = run_simulation_range(
    BASE_SEGMENT_DEFINITIONS,
    JOINT_HIERARCHY,
    SIMULATION_VARIATIONS
)
print("\nFinished All Simulation Runs.")

# --- Post-Processing (Example: Find Torque Ranges) ---
print("\n--- Torque Ranges Across Simulations ---")
all_torques_by_joint = {}

# Initialize with all joints from the hierarchy
for joint_name in JOINT_HIERARCHY.keys():
    all_torques_by_joint[joint_name] = []

# Populate with results
for variation_name, torques in simulation_results.items():
    for joint_name, torque_value in torques.items():
        if joint_name in all_torques_by_joint:
             all_torques_by_joint[joint_name].append(torque_value)
        else:
             # This might happen if a joint calculation was skipped due to an error earlier
             print(f"WARNING: Joint '{joint_name}' found in results of '{variation_name}' but not in base hierarchy keys during range calculation. Adding it.")
             all_torques_by_joint[joint_name] = [torque_value] # Initialize it here if missed


print("\nMax Static Torque Ranges (Min - Max) Nm:")
# Sort for consistent output
sorted_joints_final = sorted(all_torques_by_joint.keys())
for joint_name in sorted_joints_final:
    torques_list = all_torques_by_joint[joint_name]
    if torques_list:
        min_torque = min(torques_list)
        max_torque = max(torques_list)
        print(f"  {joint_name:<18}: {min_torque:.2f} - {max_torque:.2f}")
    else:
        # This might happen if a joint consistently errored out in all variations
        print(f"  {joint_name:<18}: No valid data calculated across variations.")

print("\nSimulation Complete.")