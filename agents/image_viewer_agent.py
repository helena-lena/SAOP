"""
Surgical Agent Orchestration Platform (SAOP) - Image Viewer (IV) Agent
"""
import os
import json
import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from typing import Dict
import sys
import numpy as np
import pydicom
import subprocess
import time
import re
import threading
import queue
from functools import lru_cache
from math import ceil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.llm_io import ollama_generate, check_json_format, extract_all_json_blocks

# ==========================================================
# Common Settings
# ==========================================================
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.6
LABEL_THICKNESS = 1
LABEL_TEXT_COLOR = (255, 255, 255)
LABEL_OUT_PADDING = 8
LABEL_COL_PADDING = 14
LABEL_INNER_PADDING = 6

# ==========================================================
# Utility Functions
# ==========================================================
def natural_sort_key(filename):
    """Return a natural sort key by extracting the first number from a filename."""
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0

def draw_text_white(frame, text, x, y):
    """Draw white text on a video frame at (x, y)."""
    cv2.putText(frame, text, (x, y), LABEL_FONT, LABEL_FONT_SCALE, 
                LABEL_TEXT_COLOR, LABEL_THICKNESS, cv2.LINE_AA)

def _resize_with_letterbox(img, target_w, target_h, pad_color=(0,0,0)):
    """
    Resize image while preserving aspect ratio using letterbox padding.
    Returns a canvas of size (target_h, target_w).
    """
    ih, iw = img.shape[:2]
    if ih == 0 or iw == 0:
        return np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)

    scale = min(target_w / iw, target_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)

    x_offset = (target_w - nw) // 2
    y_offset = (target_h - nh) // 2
    canvas[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized
    return canvas

def _ease_in_out_cubic(t: float) -> float:
    """Cubic ease-in-out function for smooth animation transitions."""
    return 4*t*t*t if t < 0.5 else 1 - pow(-2*t + 2, 3) / 2

def _make_pos_sequence(start: int, end: int, n_frames: int, easing=True):
    """
    Generate a list of interpolated integer positions from start to end
    over n_frames using optional easing.
    """
    if n_frames <= 1 or start == end:
        return [end] * max(1, n_frames)
    seq = []
    for i in range(n_frames):
        u = i / (n_frames - 1)
        if easing:
            u = _ease_in_out_cubic(u)
        val = round(start + (end - start) * u)
        seq.append(val)
    return seq

# ==========================================================
# DICOM Image Loading (with cache)
# ==========================================================
@lru_cache(maxsize=2048)
def _cached_dicom_img(folder: str, view: str, index: int):
    """Load a DICOM image from cache; falls back to disk if not cached."""
    return get_dicom_image_from_folder(folder, view, index, scale=1.0)

def read_dicom_as_image(dicom_path: str, scale: float = 1.0):
    """
    Read a DICOM file and convert it into an 8-bit BGR image.
    Includes normalization and optional scaling.
    """
    try:
        ds = pydicom.dcmread(dicom_path, force=True)
        pixel_array = ds.pixel_array
        
        # Normalize (0-255)
        pixel_array = pixel_array.astype(np.float64)
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
        pixel_array = pixel_array.astype(np.uint8)
        
        # Grayscale to BGR
        if len(pixel_array.shape) == 2:
            pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2BGR)
        
        # Resize
        if scale != 1.0:
            pixel_array = cv2.resize(pixel_array, (0, 0), fx=scale, fy=scale)
        
        return pixel_array
    except Exception as e:
        print(f"⚠️ DICOM 읽기 오류: {e}")
        return None

def get_dicom_image_from_folder(dicom_folder_path: str, view: str, position: int, scale: float = 1.0):
    """
    Retrieve a specific DICOM slice image from a view folder (axial/coronal/sagittal)
    based on slice position.
    """
    view_folder = os.path.join(dicom_folder_path, view)
    
    if not os.path.exists(view_folder):
        print(f"{view} folder does not exist: {view_folder}")
        return None
    
    dicom_files = [f for f in os.listdir(view_folder)]
    dicom_files.sort(key=natural_sort_key)

    if not dicom_files:
        print(f"No DICOM files in {view} folder")
        return None
    
    file_index = max(0, min(position - 1, len(dicom_files) - 1))
    selected_file = dicom_files[file_index]
    file_path = os.path.join(view_folder, selected_file)
    
    return read_dicom_as_image(file_path, scale)

# ==========================================================
# Streaming Overlay Processor
# ==========================================================
class StreamingOverlayProcessor:
    """Process CT overlay in streaming mode"""

    def __init__(self, state: Dict):
        self.state = state
        self.frame_queue = queue.Queue(maxsize=10)
        self.ffmpeg_process = None

        # Basic settings
        self.dicom_folder_path = (state["ct_image_path"][0] 
                                 if isinstance(state["ct_image_path"], list) 
                                 else state["ct_image_path"])
        self.display_mode = state.get("ct_display_mode", "none")
        self.main_view = state.get("ct_main_view", "axial")

        # Animation plan
        self.anim_sequences = None
        self.anim_total_frames = 0
        self.after_anim_freeze = {}
        self._frame_index = 0
    
    def _get_current_positions(self):
        """Compute slice indices to use for current frame"""
        if self.anim_sequences and self._frame_index < self.anim_total_frames:
            return {
                "sagittal": self.anim_sequences["sagittal"][self._frame_index],
                "coronal":  self.anim_sequences["coronal"][self._frame_index],
                "axial":    self.anim_sequences["axial"][self._frame_index],
            }
        if self.after_anim_freeze:
            return self.after_anim_freeze
        return {
            "sagittal": self.state.get("sagittal_pos", 15),
            "coronal":  self.state.get("coronal_pos", 15),
            "axial":    self.state.get("axial_pos", 15),
        }

    def _get_images_for_positions(self, pos: Dict):
        """Load CT images for the current positions depending on display mode."""
        imgs = {}
        
        # Small views
        if self.display_mode in ["small_views", "zoom_view"]:
            imgs["small_views"] = [
                (_cached_dicom_img(self.dicom_folder_path, "axial",    pos["axial"]),    "axial"),
                (_cached_dicom_img(self.dicom_folder_path, "coronal",  pos["coronal"]),  "coronal"),
                (_cached_dicom_img(self.dicom_folder_path, "sagittal", pos["sagittal"]), "sagittal"),
            ]
        
        # Main view
        if self.display_mode == "zoom_view" and self.main_view != "none":
            mv = self.main_view
            mv_pos = pos["sagittal"] if mv == "sagittal" else pos["coronal"] if mv == "coronal" else pos["axial"]
            imgs["main_view"] = (_cached_dicom_img(self.dicom_folder_path, mv, mv_pos), mv)
        
        return imgs
    
    def _apply_overlay_to_frame(self, frame):
        """
        Overlay CT images (main + small views) on a given video frame,
        applying correct scaling, padding, and labeling.
        """
        height, width = frame.shape[:2]

        cur_pos = self._get_current_positions()
        overlay_images = self._get_images_for_positions(cur_pos)

        # Main image
        if 'main_view' in overlay_images and overlay_images['main_view'][0] is not None:
            main_overlay_image, main_view = overlay_images['main_view']
            target_main_h = int(height * 0.65)
            target_main_w = int(target_main_h * (main_overlay_image.shape[1] / max(1, main_overlay_image.shape[0])))
            
            if target_main_w > int(width * 0.8):
                target_main_w = int(width * 0.8)
                target_main_h = int(target_main_w * (main_overlay_image.shape[0] / max(1, main_overlay_image.shape[1])))

            main_img = _resize_with_letterbox(main_overlay_image, target_main_w, target_main_h)
            main_h, main_w = main_img.shape[:2]
            main_x = (width - main_w) // 2
            main_y = (height - main_h) // 2
            frame[main_y:main_y + main_h, main_x:main_x + main_w] = main_img

            current_pos = (cur_pos["sagittal"] if main_view == "sagittal" 
                          else cur_pos["coronal"] if main_view == "coronal" 
                          else cur_pos["axial"])

            label_text = f"VIEW: {main_view.upper()} {current_pos}"
            ((tw, th), _) = cv2.getTextSize(label_text, LABEL_FONT, LABEL_FONT_SCALE, LABEL_THICKNESS)
            label_x = max(0, min(width - tw - LABEL_OUT_PADDING, main_x))
            label_y = main_y - LABEL_OUT_PADDING
            
            if label_y - th < 0:
                label_y = main_y + main_h + th + LABEL_OUT_PADDING
                if label_y > height:
                    label_x = main_x + LABEL_INNER_PADDING
                    label_y = main_y + th + LABEL_INNER_PADDING
            
            label_x = max(0, min(label_x, width - tw))
            label_y = max(th, min(label_y, height - 1))
            draw_text_white(frame, label_text, label_x, label_y)

        # Small images
        if 'small_views' in overlay_images and overlay_images['small_views']:
            n = len(overlay_images['small_views'])
            padding = LABEL_COL_PADDING
            available_h = height - padding * (n + 1)
            box_h = max(1, available_h // n)
            max_col_w = int(width * 0.28)
            box_w = min(box_h, max_col_w)
            
            if box_w + padding * 2 > width:
                box_w = max(1, width - padding * 2)
                box_h = min(box_h, box_w)

            x_offset = width - box_w - padding
            y_offset = padding

            for img, view_type in overlay_images['small_views']:
                if img is None:
                    continue
                
                tile = _resize_with_letterbox(img, box_w, box_h)
                box_h_cur, box_w_cur = tile.shape[:2]
                
                if y_offset + box_h_cur > height:
                    rem_h = height - y_offset - padding
                    if rem_h <= 10:
                        break
                    tile = _resize_with_letterbox(img, box_w, rem_h)
                    box_h_cur = rem_h

                frame[y_offset:y_offset + box_h_cur, x_offset:x_offset + box_w_cur] = tile

                cur = (cur_pos["sagittal"] if view_type == "sagittal" 
                      else cur_pos["coronal"] if view_type == "coronal" 
                      else cur_pos["axial"])
                
                label_text = f"{view_type.upper()} {cur}"
                ((tw, th), _) = cv2.getTextSize(label_text, LABEL_FONT, LABEL_FONT_SCALE, LABEL_THICKNESS)
                label_x = x_offset + LABEL_INNER_PADDING
                label_y = y_offset + th + LABEL_INNER_PADDING
                draw_text_white(frame, label_text, label_x, label_y)

                y_offset += box_h_cur + padding

        return frame
    
    def _start_ffmpeg_process(self, output_path, fps, width, height):
        """Start FFmpeg process"""
        cmd = [
            "ffmpeg", "-y", 
            "-f", "rawvideo", 
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", 
            "-pix_fmt", "bgr24", 
            "-r", str(int(fps)),
            "-i", "-", 
            "-c:v", "libx264", 
            "-preset", "ultrafast",  # fast
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "faststart", 
            output_path, 
            "-loglevel", "error"  # display only errors
        ]
        self.ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def _build_animation_plan(self, fps: float):
        """
        Precompute slice-movement animation sequences (sagittal/coronal/axial)
        based on previous and target positions.
        """
        anim_sec = float(self.state.get("ct_move_anim_sec", 0.8))
        n_frames = max(1, ceil(fps * anim_sec))

        prev = self.state.get("ct_prev_positions")
        targ = self.state.get("ct_target_positions")
        
        if not prev or not targ:
            self.anim_sequences = None
            self.anim_total_frames = 0
            self.after_anim_freeze = {
                "sagittal": self.state.get("sagittal_pos", 15),
                "coronal":  self.state.get("coronal_pos", 15),
                "axial":    self.state.get("axial_pos", 15),
            }
            return

        sag_seq = _make_pos_sequence(prev["sagittal"], targ["sagittal"], n_frames, easing=True)
        cor_seq = _make_pos_sequence(prev["coronal"],  targ["coronal"],  n_frames, easing=True)
        axi_seq = _make_pos_sequence(prev["axial"],    targ["axial"],    n_frames, easing=True)

        self.anim_sequences = {"sagittal": sag_seq, "coronal": cor_seq, "axial": axi_seq}
        self.anim_total_frames = n_frames
        self.after_anim_freeze = {
            "sagittal": targ["sagittal"], 
            "coronal": targ["coronal"], 
            "axial": targ["axial"]
        }
        print(f"Animation: {n_frames} frames ({anim_sec:.2f}s)")
    
    def _frame_writer_thread(self):
        """
        Background thread that reads frames from the queue, applies CT overlays,
        and streams them into FFmpeg.
        """
        try:
            while True:
                frame = self.frame_queue.get()
                if frame is None:
                    break
                
                overlayed_frame = self._apply_overlay_to_frame(frame.copy())

                if self.ffmpeg_process and self.ffmpeg_process.stdin:
                    try:
                        self.ffmpeg_process.stdin.write(overlayed_frame.tobytes())
                    except BrokenPipeError:
                        print("FFmpeg pipe broken")
                        break

                self._frame_index += 1
                self.frame_queue.task_done()
        except Exception as e:
            print(f"Frame writing error: {e}")
        finally:
            if self.ffmpeg_process and self.ffmpeg_process.stdin:
                self.ffmpeg_process.stdin.close()

    def process_video_streaming(self, output_path):
        """
        Process the entire input video frame-by-frame, applying CT overlays
        and generating a new MP4 file in streaming mode.
        """
        input_video_path = os.path.join(self.state["data_path"], self.state["video_path"])
        cap = cv2.VideoCapture(input_video_path)
        
        if not cap.isOpened():
            print("Failed to open video")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height}, {fps:.1f}fps, {total_frames} frames")

        self._build_animation_plan(fps)
        self._start_ffmpeg_process(output_path, fps, width, height)
        
        writer_thread = threading.Thread(target=self._frame_writer_thread)
        writer_thread.start()

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_queue.put(frame)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    print(f"Processing: {frame_count}/{total_frames}")

            self.frame_queue.put(None)
            writer_thread.join()
            
            if self.ffmpeg_process:
                self.ffmpeg_process.wait()
                if self.ffmpeg_process.returncode == 0:
                    print(f"Completed: {output_path}")
                else:
                    print(f"FFmpeg error: {self.ffmpeg_process.returncode}")
        except Exception as e:
            print(f"Processing error: {e}")
            return False
        finally:
            cap.release()
            if self.ffmpeg_process:
                self.ffmpeg_process.terminate()
        
        return True

# ==========================================================
# Action Determination Function
# ==========================================================
def decide_ct_action(state: Dict) -> str:
    """LLM decides action and parameters based on command"""
    start_time = time.time()
    max_retries = 2
    
    for attempt in range(max_retries):
        try: 
            state['ct_action_results'] = {}
            agent_name = 'CT Agent'
            agent_description = state['agent_dict'][agent_name]
            current_action = state.get("ct_action", "show_move")
            current_sagittal = state.get("sagittal_pos", 15)
            current_coronal = state.get("coronal_pos", 15)
            current_axial = state.get("axial_pos", 15)
            current_display_mode = state.get("ct_display_mode", "none")
            current_main_view = state.get("ct_main_view", "none")
            
            command = state['commands'][-1]
            revised_command = state["command_correction_results"][command]["json_output"]["english_command"]
            
            input_lines = [
                f"Decide the action for {agent_name}: {agent_description}",
                "",
                f"Current state:",
                f"- action: {current_action}",
                f"- display_mode: {current_display_mode} (none/small_views/zoom_view)",
                f"- main_view: {current_main_view}",
                f"- current positions: axial={current_axial}, coronal={current_coronal}, sagittal={current_sagittal}",
                "",
                "'action' guidelines:",
                "- 'SHOW_MOVE': Show small CT views (axial, coronal, sagittal) on the right side + move position",
                "- 'ZOOM_IN_MOVE': Show one main large CT view in center + move position", 
                "- 'ZOOM_OUT': Remove main large view, keep small views if they exist",
                "- 'REMOVE': Remove all CT views (both small and main)",
                "",
                "'display_mode' guidelines:",
                "- If action is 'SHOW_MOVE' or 'ZOOM_OUT': small_views",
                "- If action is 'ZOOM_IN_MOVE': zoom_view", 
                "- If action is 'REMOVE': none",
                "",
                "For SHOW_MOVE and ZOOM_IN_MOVE, determine new positions based on movement commands:",
                "Position adjustment guidelines:",
                "- 'move right/rightward', 'x-axis up/higher/plus': increase sagittal position",
                "- 'move left/leftward', 'x-axis down/lower/minus': decrease sagittal position",
                "- 'move front/forward/anterior', 'y-axis up/higher/plus': increase coronal position",
                "- 'move back/backward/posterior', 'y-axis down/lower/minus': decrease coronal position",
                "- 'move up/upward/superior/head', 'z-axis up/higher/plus': increase axial position",
                "- 'move down/downward/inferior/feet', 'z-axis down/lower/minus': decrease axial position",
                "- Movement amounts:",
                "  * If command explicitly mentions amount: use specified amount",
                "  * If command mentions movement without amount: use default 10 units",
                "  * If movement direction is unclear or ambiguous: keep current positions unchanged",
                "  * If command specifies exact coordinates: use those exact values",
                "- Position constraints: axial/coronal/sagittal minimum 0, axial maximum 438, coronal/sagittal maximum 512",
                "",
                "For ZOOM_IN_MOVE (only when explicitly requested), determine which view to show:",
                "- If command specifies 'axial', 'coronal', or 'sagittal': use that view",
                "- If movement is along a specific axis, set main view to that corresponding axis",
                "- Otherwise, use current main view or default to 'axial'",
                "",
                "Special handling:",
                "- If user command is 'Select CT Agent': keep all current values unchanged",
                "- If user command contains 'reset/initialize/origin': set all positions to (0,0,0), display_mode to 'small_view', main_view to 'none'",
                "",
                "Output the 'action', 'axial_pos' (int), 'coronal_pos' (int), 'sagittal_pos' (int), 'main_view', 'display_mode' in json format.",
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
            results['json_output'] = check_json_format(json_blocks[-1]) if json_blocks else None
            print("output:", results['json_output'])
            
            state['ct_action_results'][command] = results
                
            with open(os.path.join(state['save_path'], "ct_action_results.json"), "w", encoding="utf-8") as f: 
                json.dump(state['ct_action_results'], f, ensure_ascii=False)
            
            end_time = time.time()
            state['ct_action_time'] = end_time - start_time
        
            print(f"Execution time: {state['ct_action_time']:.2f}초")
            return {
                'ct_action_results': state['ct_action_results'],
                'ct_action_time': state['ct_action_time'],
            }
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                raise e

# ==========================================================
# Common Functions
# ==========================================================
def _generate_ct_video_and_save(state: Dict, suffix: str) -> Dict:
    """Generate CT video and save state"""
    start_time = time.time()
    
    ext = "." + state['video_path'].rsplit(".", 1)[-1]
    state["ct_output_path"] = os.path.join(
        state["save_path"], 
        state["video_path"].split("/")[-1].split(ext)[0] + suffix + ".mp4"
    )

    if "ct_image_path" not in state or not state["ct_image_path"]:
        print("CT image path not set")
        return state

    processor = StreamingOverlayProcessor(state)
    success = processor.process_video_streaming(state["ct_output_path"])

    end_time = time.time()
    state['ct_overlay_time'] = end_time - start_time
    print(f"Processing time: {state['ct_overlay_time']:.2f}s")

    return _save_final_results(state)

def _save_final_results(state: Dict) -> Dict:
    """Save brief results"""
    final_results = {
        "ct_action": state.get("ct_action"),
        "sagittal_pos": state.get("sagittal_pos"),
        "coronal_pos": state.get("coronal_pos"), 
        "axial_pos": state.get("axial_pos"),
        "ct_display_mode": state.get("ct_display_mode"),
        "ct_main_view": state.get("ct_main_view"),
        "ct_output_path": state.get("ct_output_path")
    }

    for command in state.get('brief_results', {}):
        state['brief_results'][command]["final_results"] = final_results

    brief_path = os.path.join(state['save_path'], "brief_results.json")
    with open(brief_path, "w", encoding="utf-8") as f:
        json.dump(state['brief_results'], f, ensure_ascii=False)

    return {
        "brief_results": state['brief_results'],
        "ct_display_mode": state["ct_display_mode"],
        "ct_main_view": state["ct_main_view"],
    }

# ==========================================================
# Action Functions
# ==========================================================
def show_and_move_ct_images(state: Dict) -> Dict:
    print("SHOW & MOVE CT (Small Views)")
    
    keys = list(state['ct_action_results'].keys())
    ct_result = state['ct_action_results'][keys[-1]]["json_output"]

    # Save previous positions
    prev_pos = {
        "sagittal": state.get("sagittal_pos", 15),
        "coronal":  state.get("coronal_pos", 15),
        "axial":    state.get("axial_pos", 15),
    }

    # Apply new values
    state["ct_action"] = ct_result.get("action")
    state["sagittal_pos"] = ct_result.get("sagittal_pos", state.get("sagittal_pos", 15))
    state["coronal_pos"]  = ct_result.get("coronal_pos",  state.get("coronal_pos", 15))
    state["axial_pos"]    = ct_result.get("axial_pos",    state.get("axial_pos", 15))
    state["ct_main_view"] = ct_result.get("main_view", "none")
    state["ct_display_mode"] = "small_views"

    # Animation settings
    state["ct_prev_positions"] = prev_pos
    state["ct_target_positions"] = {
        "sagittal": state["sagittal_pos"],
        "coronal":  state["coronal_pos"],
        "axial":    state["axial_pos"],
    }
    state.setdefault("ct_move_anim_sec", 4.0)

    print(f"Position: Sagittal={state['sagittal_pos']}, Coronal={state['coronal_pos']}, Axial={state['axial_pos']}")
    
    return _generate_ct_video_and_save(state, "_after_ct_agent")

def zoom_in_and_move_ct_images(state: Dict) -> Dict:
    print("ZOOM IN & MOVE CT (Main View)")
    
    keys = list(state['ct_action_results'].keys())
    ct_result = state['ct_action_results'][keys[-1]]["json_output"]

    prev_pos = {
        "sagittal": state.get("sagittal_pos", 15),
        "coronal":  state.get("coronal_pos", 15),
        "axial":    state.get("axial_pos", 15),
    }

    state["ct_action"] = ct_result.get("action")
    state["sagittal_pos"] = ct_result.get("sagittal_pos", state.get("sagittal_pos", 15))
    state["coronal_pos"]  = ct_result.get("coronal_pos",  state.get("coronal_pos", 15))
    state["axial_pos"]    = ct_result.get("axial_pos",    state.get("axial_pos", 15))
    state["ct_main_view"] = ct_result.get("main_view", "axial")
    state["ct_display_mode"] = "zoom_view"

    state["ct_prev_positions"] = prev_pos
    state["ct_target_positions"] = {
        "sagittal": state["sagittal_pos"],
        "coronal":  state["coronal_pos"],
        "axial":    state["axial_pos"],
    }
    state.setdefault("ct_move_anim_sec", 4.0)

    print(f"Position: Sagittal={state['sagittal_pos']}, Coronal={state['coronal_pos']}, Axial={state['axial_pos']}")
    print(f"Main view: {state['ct_main_view']}")
    
    return _generate_ct_video_and_save(state, "_after_ct_agent")

def zoom_out_ct_images(state: Dict) -> Dict:
    print("ZOOM OUT CT (Small Views Only)")
    
    keys = list(state['ct_action_results'].keys())
    ct_result = state['ct_action_results'][keys[-1]]["json_output"]
    
    state["ct_action"] = ct_result.get("action")
    state["ct_main_view"] = "none"
    state["ct_display_mode"] = "small_views"
    
    print(f"Position: Sagittal={state['sagittal_pos']}, Coronal={state['coronal_pos']}, Axial={state['axial_pos']}")
    
    return _generate_ct_video_and_save(state, "_after_ct_agent")

def remove_all_ct_images(state: Dict) -> Dict:
    print("REMOVE ALL CT")
    start_time = time.time()
    
    keys = list(state['ct_action_results'].keys())
    ct_result = state['ct_action_results'][keys[-1]]["json_output"]
    
    state["ct_action"] = ct_result.get("action")
    state["ct_display_mode"] = "none"
    state["ct_main_view"] = "none"
  
    ext = "." + state['video_path'].rsplit(".", 1)[-1]
    input_path = os.path.join(state["data_path"], state["video_path"])
    output_path = os.path.join(
        state["save_path"], 
        state["video_path"].split("/")[-1].split(ext)[0] + "_without_ct.mp4"
    )
    state["ct_output_path"] = output_path

    # Copy original with FFmpeg
    cmd = [
        "ffmpeg", "-y", "-i", input_path, 
        "-c", "copy", "-movflags", "faststart", 
        output_path, "-loglevel", "error"
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Completed: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")

    elapsed = time.time() - start_time
    print(f"Execution time: {elapsed:.2f}s")

    return _save_final_results(state)