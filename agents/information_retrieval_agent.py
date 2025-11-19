"""
Surgical Agent Orchestration Platform (SAOP) - Information Retrieval (IR) Agent
"""
import os
import json
import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from typing import Dict
import sys
import numpy as np
import subprocess
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.llm_io import ollama_generate, check_json_format, extract_all_json_blocks

# ==========================================================
# Action Determination Function
# ==========================================================
def decide_patient_info_action(state: Dict) -> str:
    """LLM decides action and fields to display"""
    start_time = time.time()
    max_retries = 2
    
    for attempt in range(max_retries):
        try: 
            state['patient_info_action_results'] = {}
            agent_name = 'Patient Info Agent'
            agent_description = state['agent_dict'][agent_name]
            command = state['commands'][-1]
            revised_command = state["command_correction_results"][command]["json_output"]["english_command"]
            current_action = state.get("pt_action")
            current_fields = state.get("pt_fields")
            
            input_lines = [
                f"Decide the action for {agent_name}: {agent_description}",
                "",
                f"Current state:",
                f"- action: {current_action}",
                f"- fields: {current_fields}",
                "",
                "Classify the 'action' as 'SHOW' or 'HIDE'.",
                "For 'SHOW' action: select only the requested fields to display among the data columns.",
                "Data Columns: ['sex/age', 'diagnosis', 'operation_name', 'comorbidities', 'tumor_location', 'height (cm)', 'bw (kg)', 'BMI', 'PFT_PRE_FVC (L)', 'PFT_PRE_FVC (%)', 'PFT_PRE_FEV1 (L)', 'PFT_PRE_FEV1 (%)', 'PFT_PRE_FEV1/FVC (%)', 'PFT_PRE_DLCO (%)']",
                "Treat 'physical information' as a specific request for: ['height (cm)', 'bw (kg)', 'BMI'].",
                "Treat 'diagnosis' and 'disease' as: ['diagnosis', 'comorbidities'].",
                "Treat 'underlying disease' and 'comorbidities' as: ['comorbidities'].",
                "If no specific information is requested, include the entire list of columns.",
                "For 'HIDE' action: return an empty list [] for fields.",
                "",
                "Special handling:",
                "- If user command is 'Select Patient Info Agent': keep all current values unchanged",
                "- If user command contains 'reset/initialize/origin': action='SHOW', fields=all columns",
                "",
                "Output the 'action' (SHOW or HIDE) and 'fields' (list) in json format.",
                "Think step-by-step before responding.",
                f"User command: {revised_command}"
            ]
            
            input_text = "\n".join(input_lines).strip()
            ollama_result = ollama_generate(input_text, model=state['ollama_model'])
            output = json.loads(ollama_result)["response"]

            results = {
                'entire_text': input_text + "\n\n" + output,
                'output': output
            }

            json_blocks = extract_all_json_blocks(results['entire_text'])
            results['json_output'] = check_json_format(json_blocks[-1]) if json_blocks else None
            print("output:", results['json_output'])

            state['patient_info_action_results'][command] = results
            
            with open(os.path.join(state['save_path'], "patient_info_action_results.json"), 
                     "w", encoding="utf-8") as f: 
                json.dump(state['patient_info_action_results'], f, ensure_ascii=False)
            
            end_time = time.time()
            state['patient_info_action_time'] = end_time - start_time
            
            print(f"Execution time: {state['patient_info_action_time']:.2f}초")
            return {
                'patient_info_action_results': state['patient_info_action_results'],
                'patient_info_action_time': state['patient_info_action_time']
            }
        except Exception as e:
            print(f"⚠️ 시도 {attempt+1} 실패: {e}")
            if attempt == max_retries - 1:
                raise e

# ==========================================================
# Build Patient Info Text
# ==========================================================
def _build_patient_info_text(row: pd.Series, columns_to_show: list) -> str:
    """Construct a formatted multiline text string containing only the requested patient data fields."""
    pt_info_lines = []
    
    # operation (display operation_name as 'operation')
    if 'operation_name' in columns_to_show:
        pt_info_lines.append(f"operation = {row['operation_name']}")
    
    # diagnosis
    if 'diagnosis' in columns_to_show:
        pt_info_lines.append(f"diagnosis = {row['diagnosis']}")
    
    # comorbidities
    if 'comorbidities' in columns_to_show:
        pt_info_lines.append(f"comorbidities = {row['comorbidities']}")
    
    # tumor_location
    if 'tumor_location' in columns_to_show:
        pt_info_lines.append(f"tumor_location = {row['tumor_location']}")
    
    # PFT_FEV1 (L & % combined)
    if 'PFT_PRE_FEV1 (L)' in columns_to_show and 'PFT_PRE_FEV1 (%)' in columns_to_show:
        pt_info_lines.append(f"PFT FEV1 = {row['PFT_PRE_FEV1 (L)']}L ({row['PFT_PRE_FEV1 (%)']}%)")
    
    # PFT_FVC (L & % combined)
    if 'PFT_PRE_FVC (L)' in columns_to_show and 'PFT_PRE_FVC (%)' in columns_to_show:
        pt_info_lines.append(f"         FVC = {row['PFT_PRE_FVC (L)']}L ({row['PFT_PRE_FVC (%)']}%)")
    
    # PFT_FEV1/FVC
    if 'PFT_PRE_FEV1/FVC (%)' in columns_to_show:
        pt_info_lines.append(f"         FEV1/FVC = {row['PFT_PRE_FEV1/FVC (%)']}%")
    
    # PFT_DLCO
    if 'PFT_PRE_DLCO (%)' in columns_to_show:
        pt_info_lines.append(f"         DLCO = {row['PFT_PRE_DLCO (%)']}%")
    
    # sex/age
    if 'sex/age' in columns_to_show:
        pt_info_lines.append(f"sex/age = {row['sex/age']}")
    
    # height/bw (combined display)
    if 'height (cm)' in columns_to_show and 'bw (kg)' in columns_to_show:
        pt_info_lines.append(f"height/bw = {row['height (cm)']}/{row['bw (kg)']}")
    else:
        if 'height (cm)' in columns_to_show:
            pt_info_lines.append(f"height = {row['height (cm)']}")
        if 'bw (kg)' in columns_to_show:
            pt_info_lines.append(f"bw = {row['bw (kg)']}")
    
    # BMI
    if 'BMI' in columns_to_show:
        pt_info_lines.append(f"BMI = {row['BMI']}")

    return "\n".join(pt_info_lines)

# ==========================================================
# Streaming Overlay Video Generation
# ==========================================================
def _generate_overlay_video_streaming(state: Dict) -> Dict:
    """
    Generate a video with patient information overlaid on each frame using
    memory-efficient streaming. 
    """
    input_path = os.path.join(state["data_path"], state["video_path"])
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print("Failed to open video")
        return state

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height}, {fps:.1f}fps, {total_frames} frames")

    output_path = state["pt_output_path"]
    
    # FFmpeg command
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(int(fps)),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast",  # 🔥 3 times faster!
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "faststart",
        output_path,
        "-loglevel", "error"  # display only errors
    ]
    
    # Text overlay settings
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    margin_right, margin_top = 10, 10
    pos_x = width - margin_right
    pos_y = margin_top
    text_color = (255, 255, 255)
    
    # Calculate text alignment
    lines = state["pt_info"].split('\n')
    line_height = font.getbbox("가")[3] - font.getbbox("가")[1] if hasattr(font, 'getbbox') else 25
    
    # Calculate max width for lines excluding operation and diagnosis
    left_align_x = pos_x
    if lines:
        remaining_lines = [line for line in lines 
                          if not ("operation" in line.lower() or "diagnosis" in line.lower())]
        if remaining_lines:
            temp_img = Image.new("RGBA", (width, height))
            temp_draw = ImageDraw.Draw(temp_img)
            max_width = max(temp_draw.textlength(line, font=font) for line in remaining_lines)
            left_align_x = pos_x - max_width
    
    try:
        with subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process each frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb).convert("RGBA")
                
                # Draw text
                draw = ImageDraw.Draw(pil_img)
                y_text = pos_y
                
                for line in lines:
                    if "operation" in line.lower() or "diagnosis" in line.lower():
                        # Right align
                        line_width = draw.textlength(line, font=font)
                        x_pos = pos_x - line_width
                    else:
                        # Left align (based on longest line in group)
                        x_pos = left_align_x
                    
                    draw.text((x_pos, y_text), line, font=font, fill=text_color)
                    y_text += line_height + 5
                
                # Send RGB frame to FFmpeg
                rgb_frame = np.array(pil_img.convert("RGB"))
                proc.stdin.write(rgb_frame.tobytes())
                
                idx += 1
                if idx % 100 == 0:
                    print(f"Processing: {idx}/{total_frames}")
        
        print(f"Completed: {output_path}")
    
    except Exception as e:
        print(f"Streaming processing error: {e}")
    finally:
        cap.release()
    
    return state

# ==========================================================
# Save Final Results
# ==========================================================
def _save_final_results(state: Dict) -> Dict:
    """Save brief results"""
    final_results = {
        "pt_action": state.get("pt_action"),
        "pt_fields": state.get("pt_fields"),
        "pt_info": state.get("pt_info", ""),
        "pt_output_path": state.get("pt_output_path"),
        "display_pt_info": state.get("display_pt_info"),
    }

    for command in state.get('brief_results', {}):
        state['brief_results'][command]["final_results"] = final_results

    brief_path = os.path.join(state['save_path'], "brief_results.json")
    with open(brief_path, "w", encoding="utf-8") as f:
        json.dump(state['brief_results'], f, ensure_ascii=False)

    return {
        "brief_results": state['brief_results'],
        "pt_action": state.get("pt_action"),
        "pt_fields": state.get("pt_fields"),
        "display_pt_info": state.get("display_pt_info"),
        "patient_info_overlay_time": state.get("patient_info_overlay_time"),
    }

# ==========================================================
# Action Functions
# ==========================================================
def show_patient_info(state: Dict) -> Dict:
    """Display patient information"""
    print("Showing patient information")
    start_time = time.time()
    
    # Update action and fields from LLM results
    keys = list(state['patient_info_action_results'].keys())
    pt_result = state['patient_info_action_results'][keys[-1]]["json_output"]
    
    state["pt_action"] = pt_result.get("action", "SHOW")
    state["pt_fields"] = pt_result.get("fields", [])
    state["display_pt_info"] = True

    ext = "." + state['video_path'].rsplit(".", 1)[-1]
    state["pt_output_path"] = os.path.join(
        state["save_path"], 
        state["video_path"].split("/")[-1].split(ext)[0] + "_with_overlay_pt_info.mp4"
    )

    # Load patient data (with cache)
    try:
        if 'patient_data_cache' not in state:
            #print("Loading patient data...")
            state['patient_data_cache'] = pd.read_excel(
                state["pt_input_path"],
                sheet_name=state["sheet_name"]
            )
        data = state['patient_data_cache']
    except Exception as e:
        print(f"Failed to load patient data: {e}")
        return state
    
    # Generate patient info text
    row = data.loc[state["pt_id"] - 1]
    columns_to_show = state["pt_fields"]
    state["pt_info"] = _build_patient_info_text(row, columns_to_show)
    
    #print(f"Displaying {len(columns_to_show)} fields")

    # Generate streaming overlay video
    _generate_overlay_video_streaming(state)

    end_time = time.time()
    state['patient_info_overlay_time'] = end_time - start_time
    print(f"Execution time: {state['patient_info_overlay_time']:.2f}초")

    return _save_final_results(state)

def remove_patient_info(state: Dict) -> Dict:
    """Remove patient info overlay"""
    print("Removing patient information")
    start_time = time.time()
    
    # Update state
    state["pt_action"] = "HIDE"
    state["pt_fields"] = []
    state["display_pt_info"] = False
  
    ext = "." + state['video_path'].rsplit(".", 1)[-1]
    input_path = os.path.join(state["data_path"], state["video_path"])
    output_path = os.path.join(
        state["save_path"], 
        state["video_path"].split("/")[-1].split(ext)[0] + "_without_pt_info.mp4"
    )
    state["pt_output_path"] = output_path

    # Copy original with FFmpeg
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c", "copy",
        "-movflags", "faststart",
        output_path,
        "-loglevel", "error"
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Completed: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")

    end_time = time.time()
    state['patient_info_overlay_time'] = end_time - start_time
    print(f"Execution time: {state['patient_info_overlay_time']:.2f}초")

    return _save_final_results(state)