"""
Robot Torque Sim

Assumptions:
1. static analysis only
2. worst case scenario - all segments are fully extended horizontally
3. joint friction is negligible
4. links are rigid bodies with fixed CoM positions
5. Joint limits are not considered in torque calculations
6. No consideration of mechanical advantage/leverage from joint positions

Input:
1. BASE_SEGMENT_DEFINITIONS: Define your robot's segments with:
   - mass (kg)
   - length (m)
   - com_offset (m) - distance from segment start to center of mass
2. JOINT_HIERARCHY: Define joint connections with:
   - parent_segment: segment name
   - child_segment: segment name
3. SIMULATION_VARIATIONS: Define test scenarios with:
   - segment name: {property: new_value}
"""

import math
import copy
import argparse
import csv
from typing import Dict, List, Optional, Set

G = 9.81

BASE_SEGMENT_DEFINITIONS = {
    "torso": {"mass": 1.604816123528635563, "length": 0.14, "com_offset": 0.061076268675456742152},
    "arm_roll_right": {"mass": 0.074284128952064418239, "length": 0.035, "com_offset": 0.038860581164318078184},
    "arm_connector_right": {"mass": 0.013804448279315588433, "length": 0.02, "com_offset": 0.0068346953424079006506},
    "elbow": {"mass": 0.11054057778322769201, "length": 0.088, "com_offset": 0.04651408986213597524},
    "arm_roll": {"mass": 0.074284128952064432116, "length": 0.035, "com_offset": 0.038860581164318071246},
    "arm_connector": {"mass": 0.013804448279315581494, "length": 0.02, "com_offset": 0.0068346953424078989159},
    "elbow_2": {"mass": 0.11054057778322769201, "length": 0.088, "com_offset": 0.046514089862135968301},
    "neck_pitch": {"mass": 0.044275119779519978014, "length": 0.03, "com_offset": 0.017325336317445198808},
    "camera": {"mass": 0.095539524348764226502, "length": 0.03, "com_offset": 0.013807352396023634192},
    "part_1": {"mass": 0.059562985501351084494, "length": 0.02, "com_offset": 0.021592443941690239034},
    "hip_yaw": {"mass": 0.061443350158479730083, "length": 0.062, "com_offset": 0.021655385352294762025},
    "knee_hip_connector": {"mass": 0.010094408194960852176, "length": 0.015, "com_offset": 0.016327850096648170025},
    "shin": {"mass": 0.094218308738066483543, "length": 0.124, "com_offset": 0.013860790149757070897},
    "feet": {"mass": 0.086230150028128074724, "length": 0.03, "com_offset": 0.016553739061179448266},
    "hip_roll": {"mass": 0.059562985501351077555, "length": 0.02, "com_offset": 0.021592443941690239034},
    "hip_yaw_2": {"mass": 0.061443350158479730083, "length": 0.062, "com_offset": 0.021655385352294758555},
    "knee_hip_connector_2": {"mass": 0.010094408194960852176, "length": 0.015, "com_offset": 0.016327850096648170025},
    "shin_2": {"mass": 0.094218308738066483543, "length": 0.124, "com_offset": 0.013860790149757070897},
    "feet_2": {"mass": 0.086230150028128074724, "length": 0.03, "com_offset": 0.016553739061179462144}
}

JOINT_HIERARCHY = {
    "right_elbow": {"parent_segment": "arm_connector_right", "child_segment": "elbow"},
    "right_shoulder_roll": {"parent_segment": "arm_roll_right", "child_segment": "arm_connector_right"},
    "right_shoulder_pitch": {"parent_segment": "torso", "child_segment": "arm_roll_right"},
    "left_elbow": {"parent_segment": "arm_connector", "child_segment": "elbow_2"},
    "left_shoulder_roll": {"parent_segment": "arm_roll", "child_segment": "arm_connector"},
    "left_shoulder_pitch": {"parent_segment": "torso", "child_segment": "arm_roll"},
    "head_pitch": {"parent_segment": "neck_pitch", "child_segment": "camera"},
    "head_yaw": {"parent_segment": "torso", "child_segment": "neck_pitch"},
    "right_ankle_pitch": {"parent_segment": "shin", "child_segment": "feet"},
    "right_knee": {"parent_segment": "knee_hip_connector", "child_segment": "shin"},
    "right_hip_yaw": {"parent_segment": "hip_yaw", "child_segment": "knee_hip_connector"},
    "right_hip_roll": {"parent_segment": "part_1", "child_segment": "hip_yaw"},
    "right_hip_pitch": {"parent_segment": "torso", "child_segment": "part_1"},
    "left_ankle_pitch": {"parent_segment": "shin_2", "child_segment": "feet_2"},
    "left_knee": {"parent_segment": "knee_hip_connector_2", "child_segment": "shin_2"},
    "left_hip_yaw": {"parent_segment": "hip_yaw_2", "child_segment": "knee_hip_connector_2"},
    "left_hip_roll": {"parent_segment": "hip_roll", "child_segment": "hip_yaw_2"},
    "left_hip_pitch": {"parent_segment": "torso", "child_segment": "hip_roll"}
}

class Robot:
    def __init__(self, segments: Dict[str, Dict[str, float]], joint_hierarchy: Dict[str, Dict[str, str]]):
        self.segments = segments
        self.joint_hierarchy = joint_hierarchy

    def get_distal_segments(self, start_segment: str, _visited: Optional[Set[str]] = None) -> List[str]:
        if _visited is None:
            _visited = set()
        if start_segment in _visited:
            return []
        _visited.add(start_segment)
        distal_segments = set()
        for joint_name, links in self.joint_hierarchy.items():
            if links['parent_segment'] == start_segment:
                child = links['child_segment']
                distal_segments.add(child)
                distal_segments.update(self.get_distal_segments(child, _visited.copy()))
        return list(distal_segments)

    def calculate_max_static_torques(self) -> Dict[str, float]:
        max_torques = {}
        for joint_name, links in self.joint_hierarchy.items():
            parent = links['parent_segment']
            child = links['child_segment']
            distal_segments = self.get_distal_segments(child)
            torque_kgcm = 0.0
            
            for segment in [child] + distal_segments:
                seg_def = self.segments.get(segment)
                if seg_def is None:
                    continue
                    
                mass = seg_def['mass']
                com_offset = seg_def['com_offset']
                torque_Nm = mass * G * com_offset
                torque_kgcm += torque_Nm * 10.19716213 / G
                
            max_torques[joint_name] = torque_kgcm
        return max_torques

class Simulation:
    def __init__(self, base_segments: Dict[str, Dict[str, float]], joint_hierarchy: Dict[str, Dict[str, str]], variations: Dict[str, Dict[str, Dict[str, float]]]):
        self.base_segments = base_segments
        self.joint_hierarchy = joint_hierarchy
        self.variations = variations
        self.results = {}

    def run(self, scenario: Optional[str] = None):
        scenarios = [scenario] if scenario else list(self.variations.keys())
        for name in scenarios:
            variation = self.variations[name]
            current_segments = copy.deepcopy(self.base_segments)
            for seg, props in variation.items():
                if seg in current_segments:
                    current_segments[seg].update(props)
            robot = Robot(current_segments, self.joint_hierarchy)
            self.results[name] = robot.calculate_max_static_torques()

    def save_results_csv(self, filename: str):
        all_joints = sorted(set().union(*[set(torques.keys()) for torques in self.results.values()]))
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Scenario'] + all_joints)
            for scenario, torques in self.results.items():
                row = [scenario] + [f"{torques.get(j, ''):.2f}" if j in torques else '' for j in all_joints]
                writer.writerow(row)

SIMULATION_VARIATIONS = {
    "BaseCase": {},
    "HeavierLimbs": {
        "UpperArm_L": {"mass": 3.5}, "Forearm_L": {"mass": 2.5},
        "UpperArm_R": {"mass": 3.5}, "Forearm_R": {"mass": 2.5},
        "Thigh_L": {"mass": 10.0}, "Shin_L": {"mass": 5.5}, "Foot_L": {"mass": 2.0},
        "Thigh_R": {"mass": 10.0}, "Shin_R": {"mass": 5.5}, "Foot_R": {"mass": 2.0},
    },
    "LongerLimbs": {
        "UpperArm_L": {"length": 0.35, "com_offset": 0.175},
        "Forearm_L": {"length": 0.32, "com_offset": 0.16},
        "UpperArm_R": {"length": 0.35, "com_offset": 0.175},
        "Forearm_R": {"length": 0.32, "com_offset": 0.16},
        "Thigh_L": {"length": 0.50, "com_offset": 0.25},
        "Shin_L": {"length": 0.45, "com_offset": 0.225},
        "Foot_L": {"length": 0.22, "com_offset": 0.11},
        "Thigh_R": {"length": 0.50, "com_offset": 0.25},
        "Shin_R": {"length": 0.45, "com_offset": 0.225},
        "Foot_R": {"length": 0.22, "com_offset": 0.11},
    },
    "HeavierAndLonger": {
        "UpperArm_L": {"mass": 3.5, "length": 0.35, "com_offset": 0.175},
        "Forearm_L": {"mass": 2.5, "length": 0.32, "com_offset": 0.16},
        "UpperArm_R": {"mass": 3.5, "length": 0.35, "com_offset": 0.175},
        "Forearm_R": {"mass": 2.5, "length": 0.32, "com_offset": 0.16},
        "Thigh_L": {"mass": 10.0, "length": 0.50, "com_offset": 0.25},
        "Shin_L": {"mass": 5.5, "length": 0.45, "com_offset": 0.225},
        "Foot_L": {"mass": 2.0, "length": 0.22, "com_offset": 0.11},
        "Thigh_R": {"mass": 10.0, "length": 0.50, "com_offset": 0.25},
        "Shin_R": {"mass": 5.5, "length": 0.45, "com_offset": 0.225},
        "Foot_R": {"mass": 2.0, "length": 0.22, "com_offset": 0.11},
    },
    "CarryingPayload_5kg_Left": {
        "Forearm_L": {"mass": BASE_SEGMENT_DEFINITIONS["Forearm_L"]["mass"] + 5.0},
    }
}

def main():
    parser = argparse.ArgumentParser(description='Robot Joint Torque Simulation')
    parser.add_argument('--scenario', type=str, help='Name of the scenario to run (default: all)')
    parser.add_argument('--csv', type=str, help='Output CSV file for results (optional)')
    args = parser.parse_args()

    sim = Simulation(BASE_SEGMENT_DEFINITIONS, JOINT_HIERARCHY, SIMULATION_VARIATIONS)
    sim.run(args.scenario)

    all_torques_by_joint = {joint: [] for joint in JOINT_HIERARCHY.keys()}
    for variation_name, torques in sim.results.items():
        for joint_name, torque_value in torques.items():
            all_torques_by_joint[joint_name].append(torque_value)

    print("\nMax Static Torque Ranges (Min - Max) kg·cm:")
    for joint_name in sorted(all_torques_by_joint.keys()):
        torques_list = all_torques_by_joint[joint_name]
        if torques_list:
            min_torque = min(torques_list)
            max_torque = max(torques_list)
            print(f"  {joint_name:<18}: {min_torque:.2f} - {max_torque:.2f}")

    if args.csv:
        sim.save_results_csv(args.csv)
        print(f"\nResults saved to {args.csv}")

if __name__ == "__main__":
    main()