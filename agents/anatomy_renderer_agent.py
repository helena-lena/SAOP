"""
Surgical Agent Orchestration Platform (SAOP) - Anatomy Rendering (AR) Agent
"""
import nibabel as nib
import numpy as np
import pyvista as pv
import cv2
import os
import time
from PIL import Image, ImageDraw, ImageFont
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import subprocess
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.llm_io import ollama_generate, check_json_format, extract_all_json_blocks

# ==========================================================
# Action Determination Function
# ==========================================================
def decide_recon_action(state: Dict) -> str:
    """LLM decides action and parameters based on command"""
    start_time = time.time()
    max_retries = 2
    for attempt in range(max_retries):
        try: 
            state['recon_action_results'] = {}
            agent_name = 'Recon Agent'
            agent_description = state['agent_dict'][agent_name]
            current_action = state.get("recon_action", "SHOW_STATIC")
            current_recon_structure = state.get("recon_structure", ['lung_lower_lobe_left', 'lung_upper_lobe_left', \
                                                                    'lung_lower_lobe_right', 'lung_middle_lobe_right', 'lung_upper_lobe_right', \
                                                                    'lung_nodules', 'lung_trachea_bronchia'])
            current_recon_view = state.get("recon_view", "surgical")                
            current_recon_rotation = state.get("recon_rotation", "static")          
            current_target_structure = state.get("recon_target_structure", "")      

            # Use actual current zoom state
            if 'structure_zoom_state' in state and state['structure_zoom_state']:
                actual_current_scale = state['structure_zoom_state'].get('current_scale', 1.0)
                actual_current_center = state['structure_zoom_state'].get('current_center', [0.0, 0.0, 0.0])
                actual_target_structure = state['structure_zoom_state'].get('current_structure', "")
                #print(f"Using actual current scale from structure_zoom_state: {actual_current_scale}")
            else:
                actual_current_scale = state.get("recon_end_scale", 1.0)
                actual_current_center = [0.0, 0.0, 0.0]
                actual_target_structure = ""
                #print(f"Using fallback scale from recon_end_scale: {actual_current_scale}")
            
            previous_start_scale = state.get("recon_start_scale", "1.0")
            previous_end_scale = state.get("recon_end_scale", "1.0")

            # Tumor location
            pt_data = pd.read_excel(
                state["pt_input_path"],
                sheet_name=state["sheet_name"]
            )
            row = pt_data.loc[state["pt_id"]-1]
            tumor_loc = row["tumor_location"]

            command = state['commands'][-1]
            revised_command = state["command_correction_results"][command]["json_output"]["english_command"]

            # Detect reset commands
            reset_keywords = ['reset', 'initialize', 'init', 'clear', 'restart', 'home', 'beginning', 'start over', 'go back to start']
            is_reset_command = any(keyword.lower() in revised_command.lower() for keyword in reset_keywords)
            
            input_lines = [
                f"Decide the action for {agent_name}: {agent_description}",
                "",
                f"Current state:",
                f"- action: {current_action}",
                f"- anatomical_structure: {current_recon_structure}",
                f"- view_mode: {current_recon_view}",
                f"- rotation: {current_recon_rotation}",
                f"- target_structure: {current_target_structure}",
                f"- start_scale: {previous_start_scale}",
                f"- end_scale: {previous_end_scale}",
                f"- actual_current_scale: {actual_current_scale}",
                f"- is_reset_command: {is_reset_command}",
                "",
                "'action' guidelines",
                "- Options: 'SHOW_STATIC', 'SHOW_ROTATE', 'SHOW_ZOOM_IN', 'SHOW_ZOOM_OUT', 'END'",
                "- If command mentions static or remove rotate: select 'SHOW_STATIC'",
                "- If command mentions to rotate: select 'SHOW_ROTATE'",
                "- If command mentions zoom in or moving to other structure: select 'SHOW_ZOOM_IN'",
                "- If command mentions zoom out: select 'SHOW_ZOOM_OUT'",
                "- If command mentions to both zoom and rotate: select 'SHOW_ZOOM_IN' or 'SHOW_ZOOM_OUT'",
                "- If command mentions 'remove all': select 'END'",
                "- If reset command detected: select 'SHOW_STATIC'",
                "- Otherwise: select 'SHOW_STATIC'",
                "",
                "'anatomical_structure' guidelines",
                "- Options: 'lung_lower_lobe_left', 'lung_upper_lobe_left', 'lung_lower_lobe_right', 'lung_middle_lobe_right', 'lung_upper_lobe_right', 'lung_nodules', 'lung_trachea_bronchia'",
                "- Structure name mappings:",
                "  a. Lung nodules (lung_nodules) are independent from lung lobes",
                "  b. Lung includes left lower lobe (LLL), left upper lobe (LUL), right upper lobe (RUL), right middle lobe (RML), right lower lobe (RLL)",
                "  c. Trachea and bronchia (lung_trachea_bronchia) are independent from lung lobes",
                f"- If command mentions to show surgical part/lobe/area or reset command detected: include only {tumor_loc} from lobes and all other non-lobe structures(lung nodues, trachea/bronchia)",
                "- If command mentions zoom to specific structure: keep the current options",
                "- If command mentions zoom without specifying structure: keep the current options",
                "- If no specific structure is mentioned in command: keep the current options",
                "- If command adds specific structures (keywords: add, include, show, display, activate, turn on, enable): add only those mentioned to the list",
                "- If command removes specific structures (keywords: remove, exclude, hide, deactivate, turn off, disable, delete): remove only those mentioned from the list",
                "- Lung lobe changes do not affect lung_nodules and lung_trachea_bronchia",
                "",
                "'view_mode' guidelines (where you look from at the start)",
                "- Options: 'anterior', 'posterior', 'left', 'right', 'superior', 'inferior', 'surgical'",
                "- For rotation commands: the view_mode is starting view before rotation",
                "- For 'rotate to the left/right/superior/inferior' commands: keep the current option because it is rotation direction",
                "- If command mentions starting from anterior, front, forward: select 'anterior'",
                "- If command mentions starting from posterior, back, backward: select 'posterior'",
                "- If command mentions starting from superior, top, upper: select 'superior'",
                "- If command mentions starting from inferior, bottom, lower: select 'inferior'",
                "- If command mentions starting from surgical, oblique, diagnonal, angled: select 'surgical'",
                "- If command mentions starting from left: select 'left'",
                "- If command mentions starting from right: select 'right'",
                "- If no starting view is mentioned in command: keep the current option",
                "- If reset command detected: select 'surgical'",
                "",
                "'rotation' guidelines (how the model rotates and moves)",
                "- Options: 'static', 'left', 'right', 'up', 'down', 'horizontal', 'vertical'",
                "- Use only the predefined options above, not anatomical terms like 'inferior' or 'superior'",
                "- If action is 'SHOW_STATIC' or command mentions static or remove rotate/movement: select 'static'",
                "- If command mentions to rotate/move left: select 'left'",
                "- If command mentions to rotate/move right: select 'right'",
                "- If command mentions to rotate/move to the superior or up: select 'up'",
                "- If command mentions to rotate/move to the inferior or down: select 'down'",
                "- If command mentions to rotate/move left and right: select 'horizontal'",
                "- If command mentions to rotate/move up and down: select 'vertical'",
                "- If no rotation mode is mentioned in command: select 'static'",
                "- If reset command detected: select 'static'",
                "",
                "'target_structure' guidelines:",
                "- Options: 'lung_lower_lobe_left', 'lung_upper_lobe_left', 'lung_lower_lobe_right', 'lung_middle_lobe_right', 'lung_upper_lobe_right', 'lung_nodules', 'lung_trachea_bronchia'",
                "- Extract the specific anatomical structure name from the command for zoom operations",
                "- If no specific target is mentioned for zoom: keep the current option", 
                "- If reset command detected: set to ''",
                "", 
                "'start_scale' and 'end_scale' guidelines:",
                "- Zoom scales follow progression: 1.0 → 2.0 → 4.0 → 8.0 → 16.0 → ...",
                f"- Use actual_current_scale ({actual_current_scale}) as the baseline for calculations",
                "- For SHOW_ROTATE and SHOW_STATIC: start_scale=actual_current_scale, end_scale=actual_current_scale",
                "- For SHOW_ZOOM_IN: start_scale=actual_current_scale, end_scale=actual_current_scale*2",
                "- For SHOW_ZOOM_OUT: start_scale=actual_current_scale, end_scale=actual_current_scale/2 (min 1.0)",
                "- If reset command detected: start_scale=1.0, end_scale=1.0, actual_current_scale=1.0",
                "",
                "Special handling:",
                "- If user command is exactly 'Select Recon Agent': the action is 'SHOW_STATIC', and keep all other current settings unchanged",
                "- Reset command triggers: reset, initialize, origin",
                "",
                "Output the 'action', 'anatomical_structure', 'view_mode', 'rotation', 'target_structure', 'start_scale', 'end_scale', 'actual_current_scale', and 'is_reset_command' in json format.",
                "",
                "Think step-by-step before responding.",
                f"User command: {revised_command}"
            ]
            input_text = "\n".join(input_lines).strip()
            ollama_result = ollama_generate(input_text, model=state['ollama_model'])
            output = json.loads(ollama_result)["response"]

            results = {}
            results['entire_text'] = input_text + "\n\n" + output
            results['output'] = output

            json_blocks = extract_all_json_blocks(results['entire_text'])
            results['json_output'] = check_json_format(json_blocks[-1].strip())
            print("output:", results['json_output'])
            state['recon_action_results'][command] = results
                
            with open(state['save_path'] + "/recon_action_results.json", "w", encoding="utf-8") as f: 
                json.dump(state['recon_action_results'], f, ensure_ascii=False)
            end_time = time.time()
            state['recon_action_time'] = end_time - start_time

            state['recon_action'] = results['json_output']['action']
            state['recon_structure'] = results['json_output']['anatomical_structure']
            state['recon_view'] = results['json_output']['view_mode']
            state['recon_rotation'] = results['json_output']['rotation']
            state['recon_target_structure'] = results['json_output']['target_structure']
            state['recon_start_scale'] = results['json_output']['start_scale']
            state['recon_end_scale'] = results['json_output']['end_scale']
            state['is_reset_command'] = results['json_output'].get('is_reset_command', False)

            #print(f"Scale calculation result: {actual_current_scale} → start:{state['recon_start_scale']}, end:{state['recon_end_scale']}")
            #if state['is_reset_command']:
            #    print(f"Reset command detected! Will initialize grid_zoom_state.")
        
            print(f"Execution time: {state['recon_action_time']:.2f}초")
            return {'recon_action_results': state['recon_action_results'],
                    'recon_action_time': state['recon_action_time'],
                    'recon_action': state['recon_action'],
                    'recon_structure': state['recon_structure'],
                    'recon_view': state['recon_view'],
                    'recon_rotation': state['recon_rotation'],
                    'recon_target_structure': state['recon_target_structure'],
                    'recon_start_scale': state['recon_start_scale'],
                    'recon_end_scale': state['recon_end_scale'],
                    'is_reset_command': state['is_reset_command']}
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                raise e

# ==========================================================
# Renderer Creation
# ==========================================================
def create_renderer(recon_dir: str):
    """
    Loads multiple NIfTI segmentation masks from a directory,
    merges them into a single renderer object,
    and assigns custom colors and opacities for each anatomical structure.
    """

    nifti_files = [
        os.path.join(recon_dir, "lungs.nii.gz"),
        os.path.join(recon_dir, "lung_nodules.nii.gz"),
        os.path.join(recon_dir, "trachea_bronchia.nii.gz"),
    ]
    
    renderer = PyVistaMedical3DRenderer.from_multiple_nifti(nifti_files)
    
    custom_colors = {
        1: (0.8, 0.2, 0.2), 2: (0.2, 0.2, 0.8), 3: (0.8, 0.2, 0.2),
        4: (0.2, 0.8, 0.2), 5: (0.2, 0.2, 0.8), 6: (1.0, 0.3, 0.0), 7: (1.0, 1.0, 0.0)
    }
    custom_opacities = {1: 0.6, 2: 0.6, 3: 0.6, 4: 0.6, 5: 0.6, 6: 0.9, 7: 0.9}
    # 1:left upper lobe, 2:left lower lobe, 3:right upper lobe, 4:right middle lobe, 5:right lower lobe, 6:nodules, 7:trachea

    renderer.setup_scene(label_colors=custom_colors, label_opacities=custom_opacities)
    return renderer

# ==========================================================
# Rotation Matrices
# ==========================================================
def _Rx(deg):
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)

def _Ry(deg):
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

def _Rz(deg):
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

def get_model_rotation_matrix(view_direction: str = 'anterior'):
    """
    Maps anatomical view names (anterior, posterior, left, right, superior, inferior, surgical) 
    to predefined rotation matrices that orient the 3D model accordingly.
    """
    rotations = {
        'anterior': np.eye(3),
        'posterior': _Rz(180),
        'left': _Rz(-90),
        'right': _Rz(90),
        'superior': _Rx(90),
        'inferior': _Rx(-90),
        'surgical': _Rx(-90) @ _Rz(90),
    }
    return rotations.get(view_direction, np.eye(3))

def apply_view_rotation(vertices, view_direction='anterior'):
    R = get_model_rotation_matrix(view_direction)
    return vertices @ R.T

# ==========================================================
# Simplified Center Calculation
# ==========================================================
def calculate_smart_center(mesh, view_rotation_matrix, structure_name):
    """
    Computes the center of a structure after applying rotation.
    For lung nodules, detects the largest connected component and uses its centroid.
    For all other structures, returns the geometric center.
    """
    transformed_points = mesh.points @ view_rotation_matrix.T
    
    if structure_name == 'lung_nodules':
        try:
            connectivity = mesh.connectivity(largest=False)
            region_ids = connectivity['RegionId']
            unique_regions = np.unique(region_ids)
            
            if len(unique_regions) > 1:
                region_sizes = [(rid, np.sum(region_ids == rid)) for rid in unique_regions]
                largest_region_id = max(region_sizes, key=lambda x: x[1])[0]
                # the center of the biggest nodule
                region_mask = region_ids == largest_region_id
                region_mesh = mesh.extract_points(region_mask)
                region_points = region_mesh.points @ view_rotation_matrix.T
                return np.mean(region_points, axis=0)
        except:
            pass
    
    # other structures: geometric center
    return np.mean(transformed_points, axis=0)

# ==========================================================
# PyVista Renderer
# ==========================================================
class PyVistaMedical3DRenderer:
    """
    Class for converting a multi-label 3D segmentation map into PyVista PolyData meshes, 
    assigning colors and opacities, and preparing for rendering.
    """
    def __init__(self, volume_data, labels):
        self.volume_data = volume_data
        self.labels = labels
        self.unique_labels = sorted([int(x) for x in np.unique(labels) if x != 0])
        self.base_meshes = {}
    
    @classmethod
    def from_multiple_nifti(cls, nifti_paths):
        """
        Loads multiple NIfTI masks, re-labeling them into unique integer labels, 
        and creates a unified renderer with merged segmentation classes.
        """
        combined_labels = None
        current_label_max = 0
        
        for i, path in enumerate(nifti_paths, 1):
            nii = nib.load(path)
            data = nii.get_fdata().astype(np.uint8)
            
            if combined_labels is None:
                combined_labels = np.zeros_like(data, dtype=np.uint8)
            
            unique_labels = np.unique(data[data > 0])
            
            if i == 1:
                for lbl in unique_labels:
                    combined_labels[data == lbl] = lbl
                current_label_max = unique_labels.max()
            else:
                new_label = current_label_max + 1
                combined_labels[data > 0] = new_label
                current_label_max = new_label
        
        return cls(combined_labels, combined_labels)
    
    def create_mesh_from_label(self, label_value, decimate=0.95, smooth=True):
        """
        Extracts a surface mesh from a binary mask corresponding to a specific anatomical label, 
        optionally smoothing and decimating it.
        """
        mask = (self.labels == label_value).astype(np.uint8)
        if mask.sum() == 0:
            return None
        
        grid = pv.wrap(mask)
        try:
            mesh = grid.contour([0.5])
        except:
            return None
        
        if smooth:
            mesh = mesh.smooth(n_iter=10, relaxation_factor=0.1)
        
        if decimate > 0:
            mesh = mesh.decimate(decimate)
        
        self.base_meshes[label_value] = mesh
        return mesh
    
    def setup_scene(self, label_colors=None, label_opacities=None):
        """
        Builds the initial mesh library with assigned color and opacity properties for each label.
        """
        default_colors = {
        1: (0.27, 0.51, 0.71),  # Left Upper - dark moderate blue (#4477AA)
        2: (0.40, 0.76, 0.65),  # Left Lower - soft cyan (#66CCEE) 
        3: (0.80, 0.47, 0.65),  # Right Upper - moderate red (#CC6677)
        4: (0.27, 0.51, 0.71),  # Right Middle - dark moderate blue (#4477AA)
        5: (0.80, 0.47, 0.65),  # Right Lower - moderate red (#CC6677)
        6: (0.87, 0.51, 0.32),  # Nodules - raw sienna (#DD8452)
        7: (1.0, 1.0, 0.75),    # Trachea
        }
        
        if label_colors is None:
            label_colors = {}
        if label_opacities is None:
            label_opacities = {}
        
        self.mesh_props = {}
        for label in self.unique_labels:
            color = label_colors.get(label, default_colors.get(label, (0.8, 0.8, 0.8)))
            opacity = label_opacities.get(label, 0.6)
            mesh = self.create_mesh_from_label(label)
            if mesh is not None:
                self.mesh_props[label] = {"color": color, "opacity": opacity}

# ==========================================================
# Overlay Engine
# ==========================================================
class SurgicalOverlayEngine:
    """
    Handles the rendering of animated 3D meshes and their overlay onto surgical videos, including rotation, zooming, and caching.
    """
    def __init__(self, renderer: PyVistaMedical3DRenderer, alpha=0.4):
        self.renderer = renderer
        self.alpha = alpha
        self.rotation_cache = {}
        self.last_mesh = None
        self.status_overlay_cache = {}
    
    def get_dynamic_rotation_angle(self, frame_idx, fps, rotation_type='left', start_scale=1.0, end_scale=1.0):
        """
        Computes the current rotation angle and zoom factor 
        for a given frame index based on animation rules and movement type (left/right/up/down/horizontal/vertical/static).
        """
        t = frame_idx / fps
        zoom_duration = 3.0
        
        if t < zoom_duration:
            progress = t / zoom_duration
            current_scale = start_scale + (end_scale - start_scale) * progress
        else:
            current_scale = end_scale
        
        if rotation_type == 'static':
            return 0.0, 'z', current_scale
        
        rotation_map = {
            'left': ('z', 30.0),
            'right': ('z', -30.0),
            'up': ('x', 30.0),
            'down': ('x', -30.0),
            'horizontal': ('z', 360.0),
            'vertical': ('x', 360.0)
        }
        
        axis, max_angle = rotation_map.get(rotation_type, ('z', 0.0))
        
        if rotation_type in ['left', 'right', 'up', 'down']:
            if t < 3.0:
                angle = (t / 3.0) * max_angle
            elif t < 4.0:
                angle = max_angle
            elif t < 7.0:
                angle = max_angle - ((t - 4.0) / 3.0) * max_angle
            else:
                angle = 0.0
        else:  # horizontal, vertical
            if t < 6.0:
                angle = (t / 6.0) * max_angle
            else:
                angle = max_angle
        
        return round(angle), axis, current_scale
    
    def get_cached_rotated_meshes(self, view_direction, angle, axis, visible_labels=None):
        """
        Caches and returns rotated meshes to avoid recalculating transformations.
        Applies base view rotation, computes centers, rotates each mesh, and saves the result in a cache for reuse.
        """
        labels_key = tuple(sorted(visible_labels)) if visible_labels else 'all'
        cache_key = f"{view_direction}_{axis}_{angle}_{labels_key}"
        
        if cache_key in self.rotation_cache:
            return self.rotation_cache[cache_key]
        
        base_rotation = get_model_rotation_matrix(view_direction)
        transformed_meshes = {}
        all_points = []
        
        for label, base_mesh in self.renderer.base_meshes.items():
            if visible_labels is not None and label not in visible_labels:
                continue
            
            transformed_points = base_mesh.points @ base_rotation.T
            transformed_meshes[label] = (base_mesh.copy(), transformed_points)
            all_points.append(transformed_points)
        
        if all_points:
            center = np.mean(np.vstack(all_points), axis=0)
        else:
            center = np.array([0, 0, 0])
        
        if axis == 'z':
            dynamic_rotation = _Rz(angle)
        elif axis == 'x':
            dynamic_rotation = _Rx(angle)
        else:
            dynamic_rotation = np.eye(3)
        
        rotated_meshes = {}
        for label, (base_mesh, transformed_points) in transformed_meshes.items():
            mesh = base_mesh.copy()
            centered_points = transformed_points - center
            rotated_points = centered_points @ dynamic_rotation.T
            mesh.points = rotated_points + center
            rotated_meshes[label] = mesh
        
        self.rotation_cache[cache_key] = rotated_meshes
        return rotated_meshes

    def create_structure_status_overlay(self, active_structures, overlay_width):
        """
        Generates a 2D side panel showing which anatomical structures are currently active using colored text icons.
        """
        cache_key = tuple(sorted(active_structures))
        
        if cache_key in self.status_overlay_cache:
            cached_panel = self.status_overlay_cache[cache_key]
            if cached_panel.shape[1] == overlay_width:
                return cached_panel
        
        all_structures = {
            1: {'name': 'Left Upper Lobe', 'color': (200, 100, 0)},
            2: {'name': 'Left Lower Lobe', 'color': (0, 100, 200)},
            3: {'name': 'Right Upper Lobe', 'color': (200, 0, 0)},
            4: {'name': 'Right Middle Lobe', 'color': (0, 200, 0)},
            5: {'name': 'Right Lower Lobe', 'color': (0, 0, 200)},
            6: {'name': 'Lung Nodules', 'color': (255, 100, 0)},
            7: {'name': 'Trachea/Bronchia', 'color': (255, 255, 0)},
        }
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        item_height = 24
        panel_height = 16 + len(all_structures) * item_height + 8
        panel_width = overlay_width
        
        pil_image = Image.new('RGB', (panel_width, panel_height), color=(0, 0, 0))
        draw = ImageDraw.Draw(pil_image)
        
        y_offset = 16
        for label, struct_info in all_structures.items():
            is_active = label in active_structures
            status_symbol = "●" if is_active else "○"
            text_color = (255, 255, 255) if is_active else (180, 180, 180)
            
            draw.text((30, y_offset), f"{status_symbol} {struct_info['name']}", 
                     fill=text_color, font=font)
            y_offset += item_height
        
        panel = np.array(pil_image)
        panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        self.status_overlay_cache[cache_key] = panel
        return panel
    
    def render_with_animation(self, width, height, view_direction, frame_idx, fps,
                            rotation_type='left', visible_labels=None,
                            start_scale=1.0, end_scale=1.0,
                            focus_label=None, focus_center=None):
        """Renders a single animation frame of the 3D model"""
        render_size = int(min(width, height) * 0.3)
        
        angle, axis, current_scale = self.get_dynamic_rotation_angle(
            frame_idx, fps, rotation_type=rotation_type,
            start_scale=start_scale, end_scale=end_scale
        )
        
        '''if frame_idx == 0 or frame_idx == int(3 * fps):
            print(f"\n🎬 Frame {frame_idx} Rendering:")
            print(f"   📐 Angle: {angle}°, Axis: {axis}")
            print(f"   📏 Current scale: {current_scale:.2f}")
            print(f"   📍 Focus center: {focus_center}")
            print(f"   🏷️  Focus label: {focus_label}")'''
        
        rotated_meshes = self.get_cached_rotated_meshes(view_direction, angle, axis, visible_labels)
        
        plotter = pv.Plotter(off_screen=True, window_size=(render_size, render_size))
        plotter.set_background((0, 0, 0))

        # Add lighting
        '''light = pv.Light(position=(0, 0, 1000), light_type='scene light')
        light.intensity = 1.2 
        plotter.add_light(light)'''
        
        for label, mesh in rotated_meshes.items():
            props = self.renderer.mesh_props.get(label, {"color": (0.8, 0.8, 0.8), "opacity": 0.8})
            plotter.add_mesh(mesh, color=props["color"], opacity=props["opacity"], smooth_shading=True)
        
        plotter.camera_position = 'xz'
        
        # Bounds caching
        if not hasattr(self, '_fixed_bounds_cache'):
            self._fixed_bounds_cache = {}
        
        labels_key = tuple(sorted(visible_labels)) if visible_labels else 'all'
        bounds_cache_key = f"{view_direction}_static_{axis}_{labels_key}"  # 'static' instead of rotation_type
        
        if bounds_cache_key not in self._fixed_bounds_cache:
            test_angles = [0, 90, 180, 270]
            all_bounds = []
            
            for test_angle in test_angles:
                test_meshes = self.get_cached_rotated_meshes(view_direction, test_angle, axis, visible_labels)
                for mesh in test_meshes.values():
                    all_bounds.append(mesh.bounds)
            
            all_bounds = np.array(all_bounds)
            fixed_bounds = [
                all_bounds[:, 0].min(), all_bounds[:, 1].max(),
                all_bounds[:, 2].min(), all_bounds[:, 3].max(),
                all_bounds[:, 4].min(), all_bounds[:, 5].max(),
            ]
            
            margin = 0.05
            for i in range(0, 6, 2):
                range_val = fixed_bounds[i+1] - fixed_bounds[i]
                fixed_bounds[i] -= range_val * margin
                fixed_bounds[i+1] += range_val * margin
            
            self._fixed_bounds_cache[bounds_cache_key] = fixed_bounds
        
        fixed_bounds = self._fixed_bounds_cache[bounds_cache_key]
        
        plotter.camera.clipping_range = (0.1, 1000.0)
        plotter.reset_camera(bounds=fixed_bounds)
        
        # Focal point setting
        if focus_center is not None and not np.allclose(focus_center, [0, 0, 0]):
            scene_center = np.array([
                (fixed_bounds[0] + fixed_bounds[1]) / 2,
                (fixed_bounds[2] + fixed_bounds[3]) / 2,
                (fixed_bounds[4] + fixed_bounds[5]) / 2
            ])
            
            focus_center_array = np.array(focus_center)
            
            t = frame_idx / fps
            zoom_duration = 3.0
            progress = min(t / zoom_duration, 1.0)
            
            if start_scale != end_scale:
                if end_scale > start_scale:  # zoom in
                    current_focal_point = scene_center + (focus_center_array - scene_center) * progress
                else:  # zoom out
                    current_focal_point = focus_center_array + (scene_center - focus_center_array) * progress
            else:
                current_focal_point = focus_center_array
            
            plotter.camera.focal_point = current_focal_point
            
            '''if frame_idx == 0 or frame_idx == int(3 * fps):
                print(f"   🎯 Scene center: {scene_center}")
                print(f"   🎯 Target focal point: {focus_center_array}")
                print(f"   🎯 Current focal point: {current_focal_point}")
                print(f"   📏 Progress: {progress:.2f}")'''
        
        plotter.camera.zoom(current_scale)
        
        '''if frame_idx == 0 or frame_idx == int(3 * fps):
            print(f"   📷 Camera position: {plotter.camera.position}")
            print(f"   🎯 Camera focal_point: {plotter.camera.focal_point}")
            print(f"   📏 Camera zoom: {current_scale}\n")'''
        
        img = plotter.screenshot(transparent_background=True, return_img=True)
        plotter.close()
        
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def process_video(self, video_path, out_path="overlay_output.mp4", view_direction='surgical',
                     rotation_type='left', start_scale=1.0, end_scale=1.0,
                     focus_label=None, focus_center=None, limit_seconds=None,
                     visible_labels=None, skip_frames=2, show_status=True):
        """
        Iterates through each frame of the original surgical video, periodically renders the 3D model, 
        blends it onto the frame, draws the status overlay, and writes the output to a video file.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        w, h = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        limit_frames = int(limit_seconds * fps) if limit_seconds else total_frames
        
        temp_out_path = out_path.rsplit('.', 1)[0] + '_temp.mp4'
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_out_path, fourcc, fps, (w, h))
        
        if not out.isOpened():
            raise RuntimeError(f"VideoWriter initialization failed: {temp_out_path}")
        
        print(f"Starting video processing: {limit_frames} frames, skip={skip_frames}")
        
        status_panel = None
        if show_status and visible_labels:
            temp_mesh = self.render_with_animation(w, h, view_direction, 0, fps,
                                                   rotation_type, visible_labels,
                                                   start_scale, end_scale, focus_label)
            status_panel = self.create_structure_status_overlay(visible_labels, temp_mesh.shape[1])
        
        frame_idx = 0
        render_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= limit_frames:
                break
            
            if frame_idx % skip_frames == 0:
                mesh_bgr = self.render_with_animation(
                    w, h, view_direction, frame_idx, fps, rotation_type,
                    visible_labels, start_scale, end_scale,
                    focus_label, focus_center
                )
                self.last_mesh = mesh_bgr
                render_count += 1
            else:
                mesh_bgr = self.last_mesh
            
            if show_status and status_panel is not None:
                combined_overlay = np.vstack([mesh_bgr, status_panel])
            else:
                combined_overlay = mesh_bgr
            
            mh, mw = combined_overlay.shape[:2]
            x_offset = w - mw - 20
            y_offset = max(20, min(h - mh - 20, 20))
            
            roi = frame[y_offset:y_offset+mh, x_offset:x_offset+mw]
            gray = cv2.cvtColor(combined_overlay, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            
            roi_bg = cv2.bitwise_and(combined_overlay, combined_overlay, mask=mask_inv)
            roi_fg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            background_layer = cv2.add(roi_bg, roi_fg)
            
            roi_model = cv2.bitwise_and(roi, roi, mask=mask)
            mesh_model = cv2.bitwise_and(combined_overlay, combined_overlay, mask=mask)
            blended_model = cv2.addWeighted(roi_model, 1.0 - self.alpha, mesh_model, self.alpha, 0)
            
            result = cv2.add(background_layer, blended_model)
            frame[y_offset:y_offset+mh, x_offset:x_offset+mw] = result
            
            out.write(frame)
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"  Processing {frame_idx}/{limit_frames}")
        
        cap.release()
        out.release()
        
        # FFmpeg command
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', temp_out_path,
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            out_path, '-loglevel', 'error'
        ]
        
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            if os.path.exists(temp_out_path):
                os.remove(temp_out_path)
        except subprocess.CalledProcessError:
            import shutil
            shutil.move(temp_out_path, out_path)
        
        print(f"Completed: {out_path}")

# ==========================================================
# State Management
# ==========================================================
def ensure_structure_zoom_state(state: Dict) -> Dict:
    """
    Ensures the zoom-state dictionary has all required fields (center, scale, structure, zoom level, history).
    Creates defaults if missing.
    """
    default_state = {
        'history': [],
        'current_center': [0.0, 0.0, 0.0],
        'current_scale': 1.0,
        'current_structure': "",
        'zoom_level': 0
    }
    
    if 'structure_zoom_state' not in state or state['structure_zoom_state'] is None:
        state['structure_zoom_state'] = default_state.copy()
    else:
        zoom_state = state['structure_zoom_state']
        for key, default_value in default_state.items():
            if key not in zoom_state or zoom_state[key] is None:
                zoom_state[key] = default_value
    
    return state['structure_zoom_state']

def save_zoom_state_to_file(state: Dict):
    """
    Serializes the zoom state and stores it as JSON on disk, converting numpy arrays as needed.
    """
    try:
        save_path = state.get('save_path', '.')
        os.makedirs(save_path, exist_ok=True)
        
        zoom_state_path = os.path.join(save_path, 'structure_zoom_state.json')
        zoom_state = state.get('structure_zoom_state', {})
        
        def make_json_safe(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_safe(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            return obj
        
        with open(zoom_state_path, 'w', encoding='utf-8') as f:
            json.dump(make_json_safe(zoom_state), f, indent=2)
        
        return True
    except Exception as e:
        print(f"Save failed: {e}")
        return False

def load_zoom_state_from_file(state: Dict):
    """
    Loads the zoom state from file if it exists; otherwise initializes default zoom state.
    """
    try:
        save_path = state.get('prev_state_path', state.get('save_path', '.'))
        zoom_state_path = os.path.join(save_path, 'structure_zoom_state.json')
        
        if not os.path.exists(zoom_state_path):
            ensure_structure_zoom_state(state)
            return False
        
        with open(zoom_state_path, 'r', encoding='utf-8') as f:
            state['structure_zoom_state'] = json.load(f)
        
        ensure_structure_zoom_state(state)
        return True
    except:
        ensure_structure_zoom_state(state)
        return False

def reset_structure_zoom_state(state: Dict, reason: str = "reset"):
    """
    Resets zoom state to defaults (center at origin, scale=1.0, no structure selected).
    """
    state['structure_zoom_state'] = {
        'history': [], 'current_center': [0.0, 0.0, 0.0],
        'current_scale': 1.0, 'current_structure': "", 'zoom_level': 0
    }

# ==========================================================
# Common Functions
# ==========================================================
STRUCTURE_TO_LABEL = {
    'lung_upper_lobe_left': 1, 'lung_lower_lobe_left': 2,
    'lung_upper_lobe_right': 3, 'lung_middle_lobe_right': 4,
    'lung_lower_lobe_right': 5, 'lung_nodules': 6,
    'lung_trachea_bronchia': 7,
}

def get_common_params(state: Dict):
    """Extract common parameters"""
    video_path = os.path.join(state["data_path"], state['video_path'])
    recon_dir = state['recon_image_path']
    
    keys = list(state['recon_action_results'].keys())
    recon_result = state['recon_action_results'][keys[-1]]["json_output"]
    
    model_names = recon_result.get('anatomical_structure', [])
    if not model_names:
        pt_data = pd.read_excel(state["pt_input_path"], sheet_name=state["sheet_name"])
        tumor_loc = pt_data.loc[state["pt_id"]-1]["tumor_location"]
        model_names = ['lung_nodules', 'lung_trachea_bronchia', tumor_loc]
    
    view = recon_result.get('view_mode')
    rotation = recon_result.get('rotation', 'static')
    target_structure = recon_result.get('target_structure', "")
    
    visible_label_ids = [STRUCTURE_TO_LABEL[s] for s in model_names if s in STRUCTURE_TO_LABEL]
    
    return video_path, recon_dir, model_names, view, rotation, target_structure, visible_label_ids

def generate_output_path(state: Dict, action: str, view: str, rotation: str, 
                        current_level: int, structure: str):
    """Generate output path"""
    ext = "." + state['video_path'].rsplit(".", 1)[-1]
    base = state["video_path"].split("/")[-1].split(ext)[0]
    output_dir = os.path.join(state["save_path"], f"{base}_{view}_{rotation}")
    suffix = f"_{structure}" if structure else "_root"
    return f'{output_dir}_{action}_L{current_level}{suffix}.mp4'

# ==========================================================
# Action Functions
# ==========================================================
def show_rotate_recon_images(state: Dict) -> Dict:
    start_time = time.time()
    state["display_recon"] = True
    print("ROTATE START")
    
    load_zoom_state_from_file(state)
    zoom_state = ensure_structure_zoom_state(state)
    
    if state.get('is_reset_command'):
        reset_structure_zoom_state(state)
        zoom_state = state['structure_zoom_state']
    
    current_center = np.array(zoom_state['current_center'])
    current_scale = float(zoom_state['current_scale'])
    current_level = int(zoom_state['zoom_level'])
    current_structure = str(zoom_state['current_structure'])
    
    #print(f"Saved center: {current_center}")
    #print(f"Saved scale: {current_scale}")
    #print(f"Saved structure: '{current_structure}'")
    #print(f"Zoom level: {current_level}")
    
    #print(f"\nLLM suggested scale: start={state.get('recon_start_scale')}, end={state.get('recon_end_scale')}")
    #print(f"Actual using scale: {current_scale} (loaded from zoom_state)")
    
    video_path, recon_dir, active_structures, view, rotation, _, visible_label_ids = get_common_params(state)
    
    #print(f"Active structures: {active_structures}")
    #print(f"Visible label IDs: {visible_label_ids}")
    #print(f"Rotation type: {rotation}")
    #print(f"View direction: {view}")
    
    '''if not active_structures:
        print("WARNING: active_structures is empty!")
    if not visible_label_ids:
        print("WARNING: visible_label_ids is empty!")'''
    
    renderer = create_renderer(recon_dir)
    
    #print(f"Renderer mesh list: {list(renderer.base_meshes.keys())}")
    #print(f"Renderer props list: {list(renderer.mesh_props.keys())}")
    
    # Determine focal point
    focus_center = None
    focus_label = None
    center_was_recalculated = False
    
    if current_level > 0 and current_structure and current_structure in STRUCTURE_TO_LABEL:
        focus_label = STRUCTURE_TO_LABEL[current_structure]
        
        if not np.allclose(current_center, [0, 0, 0]):
            focus_center = current_center
            #print(f"\nZoom level {current_level}: Using saved '{current_structure}' center")
            #print(f"   Center coordinates: {focus_center}")
            
            # Debugging: 현재 view에서 메시의 실제 중심 재계산 (비교용)
            if focus_label in renderer.base_meshes:
                target_mesh = renderer.base_meshes[focus_label]
                base_rotation = get_model_rotation_matrix(view)
                recalculated_center = calculate_smart_center(target_mesh, base_rotation, current_structure)
                
                center_diff = np.linalg.norm(focus_center - recalculated_center)
                #print(f"   Recalculated center in current view: {recalculated_center}")
                #print(f"   Center difference (distance): {center_diff:.2f}")
                
                # Warn if the difference is too big
                if center_diff > 50.0:
                    print(f"   WARNING: Large center difference! ({center_diff:.2f} > 50.0)")
                    print(f"   Suggestion: Consider using recalculated center.")
        else:
            if focus_label in renderer.base_meshes:
                target_mesh = renderer.base_meshes[focus_label]
                base_rotation = get_model_rotation_matrix(view)
                focus_center = calculate_smart_center(target_mesh, base_rotation, current_structure)
                #print(f"\nZoom level {current_level}: Initial '{current_structure}' center calculation")
                #print(f"   Calculated center: {focus_center}")
                zoom_state['current_center'] = focus_center.tolist()
                center_was_recalculated = True
            else:
                print(f"\nZoom level {current_level}: Cannot find '{current_structure}' mesh")
    else:
        #print(f"Root level (Level {current_level}): Auto-adjusting focal_point (scene_center based)")
        focus_center = None
    
    '''if focus_center is not None:
        print(f"\nFinal center to use: {focus_center}")
        print(f"   (Newly calculated: {center_was_recalculated})")'''
    
    #print(f"Matching check:")
    for vid in visible_label_ids:
        if vid in renderer.base_meshes:
            mesh = renderer.base_meshes[vid]
            #print(f"   Label {vid}: Mesh exists (points: {mesh.n_points})")
            #print(f"      Bounds: {mesh.bounds}")
        else:
            print(f"   Label {vid}: No mesh!")
    
    engine = SurgicalOverlayEngine(renderer, alpha=1.0)
    output_path = generate_output_path(state, "rotate", view, rotation, current_level, current_structure)
    
    # Debugging: check camera status
    '''print(f"Expected camera settings:")
    print(f"   Focal point: {focus_center.tolist() if focus_center is not None else 'auto'}")
    print(f"   Zoom scale: {current_scale}")
    print(f"   Rotation: {rotation}")'''
    
    # Use the saved current_scale
    engine.process_video(
        video_path=video_path, out_path=output_path,
        view_direction=view, rotation_type=rotation,
        start_scale=current_scale,
        end_scale=current_scale,
        focus_label=focus_label,
        focus_center=focus_center.tolist() if focus_center is not None else None,
        visible_labels=visible_label_ids, show_status=True, skip_frames=15
    )
    
    # Save the current_scale to the state
    state['recon_start_scale'] = current_scale
    state['recon_end_scale'] = current_scale
    
    save_zoom_state_to_file(state)
    #print(f"\nCurrent state saved (center: {zoom_state['current_center']}, scale: {zoom_state['current_scale']})")
    
    '''print(f"Current center: {zoom_state['current_center']}")
    print(f"Current scale: {zoom_state['current_scale']}")
    print(f"Current structure: '{zoom_state['current_structure']}'")
    print(f"Saved: {center_was_recalculated}")'''
    print(f"Execution time: {time.time() - start_time:.2f}s")
    
    state["recon_output_path"] = output_path
    state["recon_active_structures"] = active_structures
    state['recon_overlay_time'] = time.time() - start_time
    
    return _save_final_results(state)

def show_static_recon_images(state: Dict) -> Dict:
    start_time = time.time()
    state["display_recon"] = True
    print("STATIC START")
    
    load_zoom_state_from_file(state)
    zoom_state = ensure_structure_zoom_state(state)
    
    if state.get('is_reset_command'):
        reset_structure_zoom_state(state)
        zoom_state = state['structure_zoom_state']
    
    current_center = np.array(zoom_state['current_center'])
    current_scale = float(zoom_state['current_scale'])
    current_level = int(zoom_state['zoom_level'])
    current_structure = str(zoom_state['current_structure'])
    
    '''print(f"Saved center: {current_center}")
    print(f"Saved scale: {current_scale}")
    print(f"Saved structure: '{current_structure}'")
    print(f"Zoom level: {current_level}")
    
    print(f"\nLLM suggested scale: start={state.get('recon_start_scale')}, end={state.get('recon_end_scale')}")
    print(f"Actual using scale: {current_scale} (loaded from zoom_state)")'''
    
    video_path, recon_dir, active_structures, view, _, _, visible_label_ids = get_common_params(state)
    
    '''print(f"\nActive structures: {active_structures}")
    print(f"Visible label IDs: {visible_label_ids}")
    print(f"View direction: {view}")'''
    
    if not active_structures:
        print("WARNING: active_structures is empty!")
    if not visible_label_ids:
        print("WARNING: visible_label_ids is empty!")
    
    renderer = create_renderer(recon_dir)
    
    #print(f"Renderer mesh list: {list(renderer.base_meshes.keys())}")
    #print(f"Renderer props list: {list(renderer.mesh_props.keys())}")
    
    # Determine focal point
    focus_center = None
    focus_label = None
    center_was_recalculated = False
    
    if current_level > 0 and current_structure and current_structure in STRUCTURE_TO_LABEL:
        focus_label = STRUCTURE_TO_LABEL[current_structure]
        
        if not np.allclose(current_center, [0, 0, 0]):
            focus_center = current_center
            #print(f"\nZoom level {current_level}: Using saved '{current_structure}' center")
            #print(f"   Center coordinates: {focus_center}")
            
            # Debugging
            if focus_label in renderer.base_meshes:
                target_mesh = renderer.base_meshes[focus_label]
                base_rotation = get_model_rotation_matrix(view)
                recalculated_center = calculate_smart_center(target_mesh, base_rotation, current_structure)
                
                center_diff = np.linalg.norm(focus_center - recalculated_center)
                #print(f"   Recalculated center in current view: {recalculated_center}")
                #print(f"   Center difference (distance): {center_diff:.2f}")
                
                if center_diff > 50.0:
                    print(f"   WARNING: Large center difference! ({center_diff:.2f} > 50.0)")
                    print(f"   Suggestion: Consider using recalculated center.")
        else:
            if focus_label in renderer.base_meshes:
                target_mesh = renderer.base_meshes[focus_label]
                base_rotation = get_model_rotation_matrix(view)
                focus_center = calculate_smart_center(target_mesh, base_rotation, current_structure)
                #print(f"\nZoom level {current_level}: Initial '{current_structure}' center calculation")
                #print(f"   Calculated center: {focus_center}")
                zoom_state['current_center'] = focus_center.tolist()
                center_was_recalculated = True
            else:
                print(f"\nZoom level {current_level}: Cannot find '{current_structure}' mesh")
    else:
        print(f"\nRoot level (Level {current_level}): Auto-adjusting focal_point (scene_center based)")
        focus_center = None
    
    '''if focus_center is not None:
        print(f"\nFinal center to use: {focus_center}")
        print(f"   (Newly calculated: {center_was_recalculated})")'''
    
    #print(f"\nMatching check:")
    for vid in visible_label_ids:
        if vid in renderer.base_meshes:
            mesh = renderer.base_meshes[vid]
            #print(f"   Label {vid}: Mesh exists (points: {mesh.n_points})")
            #print(f"      Bounds: {mesh.bounds}")
        else:
            print(f"   Label {vid}: No mesh!")
    
    engine = SurgicalOverlayEngine(renderer, alpha=1.0)
    output_path = generate_output_path(state, "static", view, "static", current_level, current_structure)
    
    '''print(f"Expected camera settings:")
    print(f"   Focal point: {focus_center.tolist() if focus_center is not None else 'auto'}")
    print(f"   Zoom scale: {current_scale}")
    print(f"   Rotation: static")'''

    engine.process_video(
        video_path=video_path, out_path=output_path,
        view_direction=view, rotation_type='static',
        start_scale=current_scale,
        end_scale=current_scale,
        focus_label=focus_label,
        focus_center=focus_center.tolist() if focus_center is not None else None,
        visible_labels=visible_label_ids, show_status=True, skip_frames=15
    )
    
    state['recon_start_scale'] = current_scale
    state['recon_end_scale'] = current_scale
    
    save_zoom_state_to_file(state)
    #print(f"\nCurrent state saved (center: {zoom_state['current_center']}, scale: {zoom_state['current_scale']})")
    
    '''print(f"Current center: {zoom_state['current_center']}")
    print(f"Current scale: {zoom_state['current_scale']}")
    print(f"Current structure: '{zoom_state['current_structure']}'")
    print(f"Saved: {center_was_recalculated}")'''
    print(f"Execution time: {time.time() - start_time:.2f}s")
    
    state["recon_output_path"] = output_path
    state["recon_active_structures"] = active_structures
    state['recon_overlay_time'] = time.time() - start_time
    
    return _save_final_results(state)

def show_zoom_in_recon_images(state: Dict) -> Dict:
    start_time = time.time()
    state["display_recon"] = True
    print("ZOOM IN START")
    
    load_zoom_state_from_file(state)
    zoom_state = ensure_structure_zoom_state(state)
    
    if state.get('is_reset_command'):
        reset_structure_zoom_state(state)
        zoom_state = state['structure_zoom_state']
    
    current_center = np.array(zoom_state['current_center'])
    current_scale = float(zoom_state['current_scale'])
    current_structure = str(zoom_state['current_structure'])
    current_level = int(zoom_state['zoom_level'])
    
    # Back up current status
    current_state_backup = {
        'center': current_center.tolist(),
        'scale': current_scale,
        'structure_name': current_structure,
        'zoom_level': current_level,
        'timestamp': time.time()
    }
    zoom_state['history'].append(current_state_backup)
    
    video_path, recon_dir, active_structures, view, _, target_structure, visible_label_ids = get_common_params(state)
    
    new_scale = float(state['recon_end_scale'])
    new_level = current_level + 1
    
    renderer = create_renderer(recon_dir)
    
    # Calculate target center
    if target_structure and target_structure in STRUCTURE_TO_LABEL:
        target_label = STRUCTURE_TO_LABEL[target_structure]
        if target_label in renderer.base_meshes:
            target_mesh = renderer.base_meshes[target_label]
            base_rotation = get_model_rotation_matrix(view)
            target_center = calculate_smart_center(target_mesh, base_rotation, target_structure)
        else:
            target_center = current_center
    else:
        target_center = current_center
    
    engine = SurgicalOverlayEngine(renderer, alpha=1.0)
    output_path = generate_output_path(state, "zoom_in", view, "static", new_level, target_structure)
    
    focus_label = STRUCTURE_TO_LABEL.get(target_structure)
    
    engine.process_video(
        video_path=video_path, out_path=output_path,
        view_direction=view, rotation_type='static',
        start_scale=current_scale, end_scale=new_scale,
        focus_label=focus_label, focus_center=target_center.tolist(),
        visible_labels=visible_label_ids, show_status=True, skip_frames=15
    )
    
    # Update status
    zoom_state['current_center'] = target_center.tolist()
    zoom_state['current_scale'] = new_scale
    zoom_state['current_structure'] = target_structure
    zoom_state['zoom_level'] = new_level
    save_zoom_state_to_file(state)
    
    state["recon_output_path"] = output_path
    state["recon_active_structures"] = active_structures
    state['recon_overlay_time'] = time.time() - start_time
    
    print(f"Completed: {state['recon_overlay_time']:.2f}s")
    return _save_final_results(state)

def show_zoom_out_recon_images(state: Dict) -> Dict:
    start_time = time.time()
    state["display_recon"] = True
    print("ZOOM OUT START")
    
    load_zoom_state_from_file(state)
    zoom_state = ensure_structure_zoom_state(state)
    
    if state.get('is_reset_command'):
        reset_structure_zoom_state(state)
        zoom_state = state['structure_zoom_state']
    
    current_center = np.array(zoom_state['current_center'])
    current_scale = float(zoom_state['current_scale'])
    current_structure = str(zoom_state['current_structure'])
    current_level = int(zoom_state['zoom_level'])
    
    #print(f"Current state: Scale {current_scale:.1f}x, Level {current_level}, Structure '{current_structure}'")
    #print(f"Current center: {current_center}")
    
    # Restore history
    if not zoom_state['history']:
        target_structure = ""
        target_scale = 1.0
        target_level = 0
        print("No history → Return to root level")
    else:
        previous_state = zoom_state['history'].pop()
        target_structure = str(previous_state['structure_name'])
        target_scale = float(previous_state['scale'])
        target_level = int(previous_state['zoom_level'])
        print(f"History restored: '{target_structure}', Scale {target_scale:.1f}x, Level {target_level}")
    
    video_path, recon_dir, active_structures, view, _, _, visible_label_ids = get_common_params(state)
    
    #print(f"Active structures: {active_structures}")
    #print(f"Visible label IDs: {visible_label_ids}")
    
    renderer = create_renderer(recon_dir)
    
    # Calculate the center of the current structure (start point)
    current_focus_center = None
    if current_structure and current_structure in STRUCTURE_TO_LABEL:
        current_focus_label = STRUCTURE_TO_LABEL[current_structure]
        if current_focus_label in renderer.base_meshes:
            current_mesh = renderer.base_meshes[current_focus_label]
            base_rotation = get_model_rotation_matrix(view)
            current_focus_center = calculate_smart_center(current_mesh, base_rotation, current_structure)
            #print(f"Current '{current_structure}' center recalculated: {current_focus_center}")
        else:
            print(f"Cannot find current '{current_structure}' mesh")
    
    # Calculate the center of the current structure (end point)
    target_focus_center = None
    target_focus_label = None
    
    if target_structure and target_structure in STRUCTURE_TO_LABEL:
        target_focus_label = STRUCTURE_TO_LABEL[target_structure]
        #print(f"Target focus label: {target_focus_label} ('{target_structure}')")
        
        if target_focus_label in renderer.base_meshes:
            target_mesh = renderer.base_meshes[target_focus_label]
            base_rotation = get_model_rotation_matrix(view)
            target_focus_center = calculate_smart_center(target_mesh, base_rotation, target_structure)
            #print(f"Target '{target_structure}' center recalculated: {target_focus_center}")
        else:
            print(f"Cannot find '{target_structure}' mesh")
    #else:
    #    print(f"Return to root level: no focal_point (auto-adjust)")
    
    # Zoom out animation from start to end point
    if current_focus_center is not None and target_focus_center is not None:
        focus_center_for_animation = current_focus_center
        #print(f"Animation: {current_structure} center → {target_structure} center")
    elif current_focus_center is not None:
        focus_center_for_animation = current_focus_center
        #print(f"Animation: {current_structure} center → full view")
    elif target_focus_center is not None:
        focus_center_for_animation = target_focus_center
        #print(f"Animation: full view → {target_structure} center")
    else:
        focus_center_for_animation = None
        #print(f"Animation: auto-adjust")
    
    engine = SurgicalOverlayEngine(renderer, alpha=1.0)
    
    from_suffix = f"_{current_structure}" if current_structure else "_root"
    to_suffix = f"_{target_structure}" if target_structure else "_root"
    ext = "." + state['video_path'].rsplit(".", 1)[-1]
    base = state["video_path"].split("/")[-1].split(ext)[0]
    output_path = os.path.join(state["save_path"], 
                              f"{base}_{view}_static_zoom_out_L{target_level}_from{from_suffix}_to{to_suffix}.mp4")
    
    engine.process_video(
        video_path=video_path, out_path=output_path,
        view_direction=view, rotation_type='static',
        start_scale=current_scale, end_scale=target_scale,
        focus_label=target_focus_label,
        focus_center=focus_center_for_animation.tolist() if focus_center_for_animation is not None else None,
        visible_labels=visible_label_ids, show_status=True, skip_frames=15
    )
    
    # Update status
    if target_focus_center is not None:
        zoom_state['current_center'] = target_focus_center.tolist()
    else:
        zoom_state['current_center'] = [0.0, 0.0, 0.0]  # Root level
    zoom_state['current_scale'] = target_scale
    zoom_state['current_structure'] = target_structure
    zoom_state['zoom_level'] = target_level
    save_zoom_state_to_file(state)
    
    state["recon_output_path"] = output_path
    state["recon_active_structures"] = active_structures
    state['recon_overlay_time'] = time.time() - start_time
    
    print(f"Completed: {state['recon_overlay_time']:.2f}s")
    return _save_final_results(state)

def remove_all_recon_images(state: Dict) -> Dict:
    print("REMOVE ALL")
    start_time = time.time()
    
    state["display_recon"] = False
    state["recon_status_display"] = False
    
    video_path = os.path.join(state["data_path"], state['video_path'])
    ext = "." + state['video_path'].rsplit(".", 1)[-1]
    output_path = os.path.join(state["save_path"], 
                              state['video_path'].split("/")[-1].split(ext)[0] + "_without_recon.mp4")
    
    cmd = ["ffmpeg", "-y", "-i", video_path, "-c", "copy",
           "-movflags", "faststart", output_path, "-loglevel", "quiet"]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Completed: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    
    state["recon_output_path"] = output_path
    state["recon_active_structures"] = []
    
    elapsed = time.time() - start_time
    print(f"Execution time: {elapsed:.2f}s")

    return _save_final_results(state)

def _save_final_results(state: Dict) -> Dict:
    """Save brief results"""
    final_results = {
        "recon_action": state.get("recon_action"),
        "recon_structure": state.get("recon_structure"),
        "recon_view": state.get("recon_view"),
        "recon_rotation": state.get("recon_rotation"),
        "recon_target_structure": state.get("recon_target_structure"),
        "display_recon": state.get("display_recon"),
        "recon_start_scale": state.get("recon_start_scale"),
        "recon_end_scale": state.get("recon_end_scale"),
        "structure_zoom_state": state.get("structure_zoom_state"),
        "recon_active_structures": state.get("recon_active_structures", []),
        "recon_status_display": state.get("recon_status_display", True),
    }
    
    if 'brief_results' not in state:
        state['brief_results'] = {}
    
    for command in state.get('brief_results', {}):
        state['brief_results'][command]["final_results"] = final_results
    
    brief_path = os.path.join(state['save_path'], "brief_results.json")
    with open(brief_path, "w", encoding="utf-8") as f:
        json.dump(state['brief_results'], f, ensure_ascii=False, indent=2)
    
    return {
        "brief_results": state['brief_results'],
        "display_recon": state['display_recon'],
        "recon_active_structures": state.get("recon_active_structures", []),
        "recon_status_display": state.get("recon_status_display", True)
    }
