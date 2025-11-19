"""
Surgical Agent Orchestration Platform (SAOP) - Integrated Version

Supports three modes:
- Real-time audio recording from edge laptop
- Synthesized audio files
- Text input

USAGE:co
    # Real-time mode
    xvfb-run -s "-screen 0 800x600x24" python saop_integrated.py config.yaml
    xvfb-run -s "-screen 0 800x600x24" python saop_integrated.py config.yaml --mode realtime
    
    # Synthesized audio mode
    - single file
    xvfb-run -s "-screen 0 800x600x24" python saop_integrated.py config.yaml --mode synthesized --audio_path ./datasets/tts_outputs/en-US-AriaNeural/1.mp3
    
    - folder
    xvfb-run -s "-screen 0 800x600x24" python saop_integrated.py config.yaml --mode synthesized --audio_path ./datasets/tts_outputs/en-US-AriaNeural
    
    - folder with range
    xvfb-run -s "-screen 0 800x600x24" python saop_integrated.py config.yaml --mode synthesized --audio_path ./datasets/tts_outputs/en-US-AriaNeural -s 2 -e 4
    
    # Text mode
    - single command
    xvfb-run -s "-screen 0 800x600x24" python saop_integrated.py config.yaml --mode text --text_command "Show patient information"
    
    - consecutive commands file
    xvfb-run -s "-screen 0 800x600x24" python saop_integrated.py config.yaml --mode text --text_commands_file ./datasets/text_commands.txt
"""

import os
import torch
from transformers import set_seed
import json
from typing import List, Dict, Any
from typing_extensions import TypedDict
import re
import time
from datetime import datetime
import subprocess
import pandas as pd
import yaml
from faster_whisper import WhisperModel
import sys
import glob
from pydub import AudioSegment
import argparse

from agents.information_retrieval_agent import decide_patient_info_action, show_patient_info, remove_patient_info 
from agents.image_viewer_agent import decide_ct_action, show_and_move_ct_images, zoom_in_and_move_ct_images, zoom_out_ct_images, remove_all_ct_images
from agents.anatomy_renderer_agent import decide_recon_action, show_rotate_recon_images, show_zoom_in_recon_images, show_zoom_out_recon_images, show_static_recon_images, remove_all_recon_images

from utils.llm_io import ollama_generate, check_json_format, extract_all_json_blocks

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

class SAOP_Workflow_State(TypedDict):
    """Surgical Agent Orchestration Platform (SAOP) workflow state."""
    # Core settings
    device: str
    torch_dtype: Any
    hf_token: str
    stt_model: Any
    ollama_model: str
    load_time: float
    save_load_keys: List
    
    # Paths and data
    data_path: str
    video_path: str
    audio_path: str
    audio_mode: str  # "realtime", "synthesized", or "text"
    text_command: str  # for text mode
    save_path: str
    
    # Speech processing
    stt_chunks: List
    commands: List
    corrected_commands: List
    agent_history: List
    stt_time: float
    
    # Remote audio config (for realtime mode)
    edge_user: str
    edge_ip: str
    edge_remote_path: str
    edge_py_filename: str
    edge_conda_env: str
    server_remote_path: str
    server_remote_user: str
    server_remote_ip: str
    server_remote_port: str
    
    # Agent info
    agent_info: str
    agent_dict: Dict
    command_correction_results: Dict
    command_correction_time: float
    command_reasoning_results: Dict
    command_reasoning_time: float
    brief_results: Dict
    
    # Patient Info Agent
    pt_input_path: str
    sheet_name: str
    pt_id: int
    pt_info: str
    pt_action: str
    pt_fields: List
    display_pt_info: bool
    patient_info_action_results: Dict
    patient_info_action_time: float
    
    # CT Agent
    ct_image_path: list
    ct_action: str
    sagittal_pos: int
    coronal_pos: int
    axial_pos: int
    ct_display_mode: str
    ct_main_view: str
    ct_action_results: Dict
    ct_action_time: float
    
    # 3D Recon Agent
    recon_image_path: str
    display_recon: bool
    recon_action: str
    recon_structure: List
    recon_view: str
    recon_rotation: str
    recon_target_structure: str
    recon_start_scale: float
    recon_end_scale: float
    grid_zoom_state: Dict
    recon_action_results: Dict
    recon_action_time: float
    
    # Workflow control
    is_reset_command: bool
    prev_state_path: str

def realtime_audio(state: SAOP_Workflow_State) -> SAOP_Workflow_State:
    """Obtain real-time audio from edge laptop."""
    print("---Real-time Audio Stage---")
    start_time = time.time()
    
    # Load settings
    edge_user = state.get('edge_user')
    edge_ip = state.get('edge_ip')
    edge_remote_path = state.get('edge_remote_path')
    edge_py_filename = state.get('edge_py_filename')
    edge_conda_env = state.get('edge_conda_env')
    server_remote_path = state.get('server_remote_path')
    server_remote_user = state.get('server_remote_user')
    server_remote_ip = state.get('server_remote_ip')
    server_remote_port = state.get('server_remote_port')
    
    now = datetime.now()
    date_now = now.strftime("%Y%m%d")
    time_now = now.strftime("%H%M%S")
    
    # Create audio save path
    ext = "."+state['video_path'].rsplit(".", 1)[-1]
    state['audio_path'] = os.path.join(state["data_path"], state["video_path"].split(ext)[0]) \
                          + "/audios/" + date_now + "/" + time_now + "/record.wav"
    state['audio_path'] = state['audio_path'].replace(" ", "_")
    state['save_path'] = state['audio_path'].rsplit("/", 1)[0]
    if not os.path.exists(state['save_path']): os.makedirs(state['save_path'])
    
    # Execute on edge laptop via SSH
    cmd = (
    f'ssh {edge_user}@{edge_ip} '
    f'"cmd /C \\"cd /d {os.path.dirname(edge_remote_path)} && conda activate {edge_conda_env} && '
    f'python {os.path.join(edge_remote_path, edge_py_filename)} --date {date_now} --time {time_now} --remote_user {server_remote_user} --remote_host {server_remote_ip} --remote_port {server_remote_port} --remote_path {server_remote_path} --save_path {state['save_path']}\\""'
    )

    # Real-time print
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, bufsize=1, encoding="utf-8") as proc: 
        for line in proc.stdout:
            print(line, end="")

    elapsed = time.time() - start_time
    print(f"Execution time: {elapsed:.2f}s")
    
    return {
        'audio_path': state['audio_path'],
        'save_path': state['save_path'],
        'load_time': elapsed
    }

def load_synthesized_audio(state: SAOP_Workflow_State) -> SAOP_Workflow_State:
    """Load and preprocess synthesized audio file."""
    print("---Loading & Preprocessing Synthesized Audio---")
    start_time = time.time()
    
    if not state.get('audio_path') or state['audio_path'] == "None":
        raise ValueError("No audio path provided for synthesized mode")
    
    if not os.path.exists(state['audio_path']):
        raise FileNotFoundError(f"Audio file not found: {state['audio_path']}")
    
    audio = AudioSegment.from_file(state['audio_path'])
    ext = state['audio_path'].rsplit(".", 1)[-1]
    state['save_path'] = state['audio_path'].rsplit(".", 1)[0]
    
    if not os.path.exists(state['save_path']): 
        os.makedirs(state['save_path'])
    
    # Convert to wav with 16kHz if needed
    if ext != "wav" or audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)
        wav_path = state['save_path'] + ".wav"
        audio.export(wav_path, format="wav")
        state['audio_path'] = wav_path
    
    elapsed = time.time() - start_time
    print(f"Execution time: {elapsed:.2f}s")
    
    return {
        'audio_path': state['audio_path'],
        'save_path': state['save_path'],
        'load_time': elapsed
    }

def text_to_command(state: SAOP_Workflow_State) -> SAOP_Workflow_State:
    """Convert text command directly to command without audio/STT."""
    print("---Text to Command Stage---")
    start_time = time.time()
    
    try:
        # Get text command
        command_text = state.get('text_command', '')
        if not command_text or command_text == "None":
            raise ValueError("No text command provided for text mode")
        
        # Create save path
        now = datetime.now()
        date_now = now.strftime("%Y%m%d")
        time_now = now.strftime("%H%M%S")
        ext = "."+state['video_path'].rsplit(".", 1)[-1]
        save_path = os.path.join(state["data_path"], state["video_path"].split(ext)[0]) \
                              + "/text_commands/" + date_now + "/" + time_now
        save_path = save_path.replace(" ", "_")
        
        if not os.path.exists(save_path): 
            os.makedirs(save_path)
        
        # Process command (skip STT)
        print("Input text command:", command_text)
        commands = command_text
        
        # Filter invalid commands
        commands_low = commands.lower().strip()
        if commands_low == "":
            commands = "None"
        else:
            words = commands_low.split()
            is_invalid = False
            
            if commands_low == "you": # exact match
                is_invalid = True
            elif any(word in commands_low for word in ["thank you", "thanks", "okay", "감사합니다", "고맙습니다", "뉴스", "구독"]):
                is_invalid = True
            
            if is_invalid:
                commands = "None"

        # Append to commands list
        updated_commands = state['commands'].copy()
        updated_commands.append(commands)
        
        # Save text command result
        text_result = {
            'text': command_text,
            'timestamp': time_now
        }
        with open(save_path+"/text_command.json", "w", encoding="utf-8") as f: 
            json.dump(text_result, f, ensure_ascii=False)
        
        elapsed = time.time() - start_time
        print(f"Execution time: {elapsed:.2f}s")
        
        # ✅ save_path도 return에 포함
        return {
            'save_path': save_path,
            'commands': updated_commands,
            'stt_time': elapsed,
        }
    
    except Exception as e:
        print(f"Text to command failed: {e}")
        raise

def speech_to_text(state: SAOP_Workflow_State) -> SAOP_Workflow_State:
    """Convert speech to text using Whisper model."""
    print("---Speech-to-Text Stage---")
    try: 
        start_time = time.time()

        segments, info = state['stt_model'].transcribe(state['audio_path'], 
                                                       beam_size=8, 
                                                       language="en",
                                                       temperature=[0.0],
                                                       no_speech_threshold=0.3,
                                                       log_prob_threshold=-2.0,
                                                       compression_ratio_threshold=None,
                                                       )
        results = []
        for segment in segments: 
            results.append({'start':segment.start, 'end':segment.end, 'text':segment.text.strip()})
        state['stt_chunks'] = results

        # Save STT results
        with open(state['save_path']+"/stt_result.json", "w", encoding="utf-8") as f: 
            json.dump(state['stt_chunks'], f, ensure_ascii=False)
        
        # Process commands
        commands_list = [result['text'].strip() for result in state['stt_chunks']]
        commands = " ".join(commands_list)
        print("Input commands:", commands)

        # Filter invalid commands
        commands_low = commands.lower().strip()
        if commands_low == "":
            commands = "None"
        else:
            words = commands_low.split()
            is_invalid = False
            
            if commands_low == "you": # exact match
                is_invalid = True
            elif any(word in commands_low for word in ["thank you", "thanks", "감사합니다", "고맙습니다", "뉴스", "구독"]):
                is_invalid = True
            
            if is_invalid:
                commands = "None"

        state['commands'].append(commands)

        elapsed = time.time() - start_time
        print(f"Execution time: {elapsed:.2f}s")
        
        return {
            'stt_chunks': state['stt_chunks'],
            'commands': state['commands'],
            'stt_time': elapsed,
        }
    
    except Exception as e:
        print(f"STT failed: {e}")
        raise

def command_correction(state: SAOP_Workflow_State) -> SAOP_Workflow_State:
    """Correct and validate user commands."""
    print("---Command Correction Stage---")
    start_time = time.time()
    max_retries = 2

    for attempt in range(max_retries):
        try: 
            state['command_correction_results'] = {}
            command = state['commands'][-1]
            print("command:", command)
            input_lines = [
                "Here are the available agents for Da Vinci Surgical System.",
                f"{state['agent_info']}",
                "Translate the current user command into English and then check if it is valid.",
                "Preserve the full detail of the current user command.", 
                "Pay special attention to accurate translation of direction.",
                "Do not generate any extra information or explanation.",
                "For rotation commands, distinguish between the view (where you look from) and rotation direction (how the model rotates and moves) (e.g. Rotate the inferior view to the superior).",
                "If no view is menioned in rotation commands, do not mention it (e.g. Rotate to the left, Rotate left and right).",
                "If current user command is 'None', write 'Select' followed by the last agent name in the previous commands (e.g. 'Select Patient Info Agent').",
                "Never infer a default agent based on empty or missing history.",
                "",
                "Text correction rules:",
                "Replace 'city', '스티', '세트', '시트' with 'CT' when detected, as it is likely a misrecognition of 'CT' based on context.",
                "Replace '-', '마이너스' with 'minus' if it is followed by a number.",
                "Replace '+', '플러스' with 'plus' if it is followed by a number.",
                "Replace 'exhale', 'Exhale', 'XCR', '액시얼', '엑시얼' similar terms with 'axial'.",
                "Replace COVID-related terms, '코로나', '코로나19' similar terms with 'coronal'.",
                "Replace 'surge time', '사지탈', '사지털', '서지털' similar terms with 'sagittal'.",
                "Replace '포스테리어', '포스테리오' similar terms with 'posterior'.",
                "Replace '슈퍼리어', '슈퍼리얼' similar terms with 'superior'.",
                "Replace 'Ontario' similar terms with 'anterior'.",
                "Replace 'red', 'at' similar terms with 'add'.",
                "Replace '백워드' similar terms with 'backward' only when it contains '워' or '워드'.",
                "Do NOT replace '백' when the term is a pure number ('백(100)', '이백(200)', '삼백(300)', '사백(400)', '오백(500)') or a number with unit like '백칸', '이백칸', '삼백칸'.",
                "Replace '리컨', '닉컨' similar terms with 'recon'.",
                "Replace '호리젠털', '오리지널' similar terms with 'horizontal'.",
                "Replace '버티컬', '벌티컬' similar terms with 'vertical'."
                "Replace '레프트뷰', '랩트뷰' similar terms with 'left view'",
                "Replace '라이트뷰' similar terms with 'right view'",
                "Replace 'wright' similar terms with 'right'",
                "Replace '중인', 'June in' with 'zoom in'.",
                "Replace 'long', '렁', '롱', '폐' similar terms with 'lung'."
                "Replace '노듈', '노즐', '노듀' similar terms with 'nodule'.",
                "Replace '우상엽' similar terms with 'right upper lobe'.",
                "Replace '우중엽' similar terms with 'right middle lobe'.",
                "Replace '우하엽' similar terms with 'right lower lobe.",
                "Replace '써지컬', '썰지컬' similar terms with 'surgical'.",
                "Replace '제거' similar terms with 'remove'.",
                "Replace 'track here', '트라키아', '트락키아' similar terms with 'trachea'.",
                "Replace 'blonkia' similar terms with 'bronchia'.",
                "Replace 'info', '인포' similar terms with 'information'.",
                "Replace 'road', '롯' similar terms with 'load'.",
                "Replace '리스테', '리셋' similar terms with 'reset.",
                "",
                "Validation rules:",
                "- The command is VALID if related to the available agents (data display/request/addition/elimination/manipulation/initialization).",
                "- The command is VALID even if it only contains single data entity keywords (e.g. diagnosis, disease, etc.) within the agent's scope.",
                "- The command is INVALID if no user command is provided or completely unrelated to available agents.",
                "",
                "Output the 'english_command' and 'validity' (VALID or INVALID) in json format and step-by-step reasoning.",
                f"Previous commands: {state["corrected_commands"][-3:]}",
                f"Agent history: {state["agent_history"][-3:]}",
                f"Current user command: {command}"
                ]
            
            input_text = "\n".join(input_lines).strip()
            ollama_result = ollama_generate(input_text, model=state['ollama_model'])
            output = json.loads(ollama_result)["response"]
            print("output:", output)

            results = {
                'entire_text': input_text + "\n\n" + output,
                'output': output
            }

            json_blocks = extract_all_json_blocks(results['entire_text'])
            results['json_output'] = check_json_format(json_blocks[-1].strip())

            state['command_correction_results'][command] = results

            # Add corrected commands or remove invalid ones
            if results['json_output']['validity'] == "VALID":
                state['corrected_commands'].append(results["json_output"]["english_command"])
            elif results['json_output']['validity'] == "INVALID":
                state['commands'] = state['commands'][:-1]

            # Save results
            with open(state['save_path']+"/command_correction_result.json", "w", encoding="utf-8") as f: 
                json.dump(state['command_correction_results'], f, ensure_ascii=False)
            
            elapsed = time.time() - start_time
            print(f"Execution time: {elapsed:.2f}s")
            
            return {
                'command_correction_results': state['command_correction_results'],
                'command_correction_time': elapsed
            }
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                raise

def command_reasoning(state: SAOP_Workflow_State) -> SAOP_Workflow_State:
    """Analyze commands and select appropriate agent."""
    print("---Command Reasoning Stage---")
    start_time = time.time()
    max_retries = 2

    for attempt in range(max_retries):
        try:
            state['command_reasoning_results'] = {}
            for command, results in state['command_correction_results'].items():
                input_lines = [
                    "Here are the available agents for Da Vinci Surgical System.",
                    f"{state['agent_info']}",
                    "Analyze the following user command and select the most appropriate agent.",
                    "For ambiguous commands that could apply to multiple agents, use the most recent agent.",
                    "Output the 'agent_name' in json format.",
                    "If no suitable agent is available, fill in as 'None'.",
                    "Think and write your step-by-step reasoning before responding.",
                    f"Previous commands: {state["corrected_commands"][-4:-1]}",
                    f"Agent history: {state["agent_history"][-3:]}",
                    f"User command: {results['json_output']['english_command']}",
                    ]
                
                input_text = "\n".join(input_lines).strip()
                ollama_result = ollama_generate(input_text, model=state['ollama_model'])
                output = json.loads(ollama_result)["response"]
                print("output:", output)

                results = {
                    'entire_text': input_text + "\n\n" + output,
                    'output': output
                }

                json_blocks = extract_all_json_blocks(results['entire_text'])
                results['json_output'] = check_json_format(json_blocks[-1].strip())

                state['command_reasoning_results'][command] = results
                state["agent_history"].append(results['json_output']['agent_name'])
            
            # Save results
            with open(state['save_path']+"/command_reasoning_results.json", "w", encoding="utf-8") as f: 
                json.dump(state['command_reasoning_results'], f, ensure_ascii=False)
            
            state['brief_results'] = {}
            for command, evaluation_result in state['command_reasoning_results'].items():
                state['brief_results'][command] = {
                    "command_correction": state['command_correction_results'][command]['json_output'],
                    "command_reasoning": state['command_reasoning_results'][command]['json_output']
                }
                
            elapsed = time.time() - start_time
            print(f"Execution time: {elapsed:.2f}s")
            
            return {
                'command_reasoning_results': state['command_reasoning_results'],
                'command_reasoning_time': elapsed
            }
        
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                raise

class LLMOrchestrator:
    """Workflow Orchestrator Agent for SAOP."""

    def __init__(self, initial_state: SAOP_Workflow_State):
        self.state = initial_state
        self.max_iterations = 15
        self.current_iteration = 0
        
        # Available functions mapping
        self.functions = {
            'realtime_audio': realtime_audio,
            'load_synthesized_audio': load_synthesized_audio,
            'text_to_command': text_to_command,
            'speech_to_text': speech_to_text,
            'command_correction': command_correction,
            'command_reasoning': command_reasoning,
            'decide_patient_info_action': decide_patient_info_action,
            'show_patient_info': show_patient_info,
            'remove_patient_info': remove_patient_info,
            'decide_ct_action': decide_ct_action,
            'show_and_move_ct_images': show_and_move_ct_images,
            'zoom_in_and_move_ct_images': zoom_in_and_move_ct_images,
            'zoom_out_ct_images': zoom_out_ct_images,
            'remove_all_ct_images': remove_all_ct_images,
            'decide_recon_action': decide_recon_action,
            'show_rotate_recon_images': show_rotate_recon_images,
            'show_zoom_in_recon_images': show_zoom_in_recon_images,
            'show_zoom_out_recon_images': show_zoom_out_recon_images,
            'show_static_recon_images': show_static_recon_images,
            'remove_all_recon_images': remove_all_recon_images
        }

        # Agent configuration - centralized mapping
        self.agent_config = {
            'Patient Info Agent': {
                'results_key': 'patient_info_action_results',
                'decide_function': 'decide_patient_info_action',
                'actions': {
                    'SHOW': 'show_patient_info',
                    'REMOVE': 'remove_patient_info'
                }
            },
            'CT Agent': {
                'results_key': 'ct_action_results',
                'decide_function': 'decide_ct_action',
                'actions': {
                    'SHOW_MOVE': 'show_and_move_ct_images',
                    'ZOOM_IN_MOVE': 'zoom_in_and_move_ct_images',
                    'ZOOM_OUT': 'zoom_out_ct_images',
                    'REMOVE': 'remove_all_ct_images'
                }
            },
            'Recon Agent': {
                'results_key': 'recon_action_results',
                'decide_function': 'decide_recon_action',
                'actions': {
                    'SHOW_ROTATE': 'show_rotate_recon_images',
                    'SHOW_ZOOM_IN': 'show_zoom_in_recon_images',
                    'SHOW_ZOOM_OUT': 'show_zoom_out_recon_images',
                    'SHOW_STATIC': 'show_static_recon_images',
                    'END': 'remove_all_recon_images'
                }
            }
        }
        
        # Extract all agent action functions for easy reference
        self.agent_action_functions = set()
        for agent_info in self.agent_config.values():
            self.agent_action_functions.update(agent_info['actions'].values())
        
        self.executed_actions = set()

    def initialize(self):
        """Initialize system by loading global state."""
        print("===== LLM Orchestrator Initialization =====")

        # Global state path
        video_dir = os.path.join(
            self.state["data_path"], 
            "/".join(self.state["video_path"].split("/")[:-1])
        )
        global_state_path = os.path.join(video_dir, "global_state.json")
        if not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)
            print(f"Created directory: {video_dir}")
        
        # Load global state file if exists, otherwise start with initial state
        if os.path.exists(global_state_path):
            print(f"Loading existing global state from: {global_state_path}.")
            global_state = self.load_global_state(global_state_path)
            self.state.update(global_state)
            print("Global state loaded successfully.")
        else:
            print(f"No existing global state found at: {global_state_path}. Setting a new one.")

        self.state['global_state_path'] = global_state_path

    def get_orchestration_decision(self):
        """LLM decides the next function to execute."""
        start_time = time.time()
        current_status = self._analyze_current_status()
        
        # Determine which input function to include based on mode
        audio_mode = self.state.get('audio_mode', 'realtime')
        
        if audio_mode == "text":
            input_function = "text_to_command"
            input_description = "Convert text command to internal command"
            next_function = "command_correction"
        elif audio_mode == "synthesized":
            input_function = "load_synthesized_audio"
            input_description = "Load synthesized audio file"
            next_function = "speech_to_text"
        else:  # realtime
            input_function = "realtime_audio"
            input_description = "Record real-time audio input"
            next_function = "speech_to_text"
        
        input_lines = [
            "You are an orchestrator for the Surgical Agent Orchestration Platform workflow.",
            "Based on the current status and functions, decide the next function to execute.",
            "",
            "Available workflow functions",
            f"- {input_function}: {input_description}",
        ]
        
        # Add speech_to_text only if not in text mode
        if audio_mode != "text":
            input_lines.append("- speech_to_text: Convert audio to text commands")
        
        input_lines.extend([
            "- command_correction: Correct the commands and validate it",
            "- command_reasoning: Select appropriate agent for the corrected command",
            "",
            "Patient Info Agent functions",
            "- decide_patient_info_action: Decide patient info action",
            "- show_patient_info: Display patient information",
            "- remove_patient_info: Hide patient information",
            "",
            "CT Agent functions",
            "- decide_ct_action: Decide CT action",
            "- show_and_move_ct_images: Show and move CT images",
            "- zoom_in_and_move_ct_images: Zoom in and move CT images",
            "- zoom_out_ct_images: Zoom out CT images", 
            "- remove_all_ct_images: Remove all CT images",
            "",
            "Recon Agent functions",
            "- decide_recon_action: Decide 3D reconstruction action",
            "- show_rotate_recon_images: Show and rotate 3D recon",
            "- show_zoom_in_recon_images: Zoom in 3D recon",
            "- show_zoom_out_recon_images: Zoom out 3D recon",
            "- show_static_recon_images: Show static 3D recon",
            "- remove_all_recon_images: Remove 3D recon images",
            "",
            "Workflow decision rules",
        ])
        
        if audio_mode == "text":
            input_lines.extend([
                f"- Start with '{input_function}' if text command is available",
                f"- Follow: {input_function} -> command_correction",
            ])
        else:
            input_lines.extend([
                f"- Start with '{input_function}' if no audio is recorded",
                f"- Follow: {input_function} -> speech_to_text -> command_correction",
            ])
        
        input_lines.extend([
            f"- If command is INVALID, return to '{input_function}'",
            "- If command is VALID, continue: command_reasoning",
            "- Based on agent selection, route to appropriate agent functions",
            "- If status shows 'action decided, execution needed', execute the corresponding action function",
            "- If status shows 'workflow finished' or 'agent completed', use 'END' to finish the workflow",
            "",
            f"Current status: {current_status}",
            "",
            "Output with all function names and 'END' as keys and their probabilities as values in json format.",
            "Fill in the template below with appropriate probability values (float). All probabilities must sum to 1.0.",
            "Template:",
            "{",
            f"  '{input_function}': float,",
        ])
        
        # Add speech_to_text only if not in text mode
        if audio_mode != "text":
            input_lines.append("  'speech_to_text': float,")
        
        input_lines.extend([
            "  'command_correction': float,",
            "  'command_reasoning': float,",
            "  'decide_patient_info_action': float,",
            "  'show_patient_info': float,",
            "  'remove_patient_info': float,",
            "  'decide_ct_action': float,",
            "  'show_and_move_ct_images': float,",
            "  'zoom_in_and_move_ct_images': float,",
            "  'zoom_out_ct_images': float,",
            "  'remove_all_ct_images': float,",
            "  'decide_recon_action': float,",
            "  'show_rotate_recon_images': float,",
            "  'show_zoom_in_recon_images': float,",
            "  'show_zoom_out_recon_images': float,",
            "  'show_static_recon_images': float,",
            "  'remove_all_recon_images': float,",
            "  'END': float",
            "}",
        ])
        
        input_text = "\n".join(input_lines).strip()
        print("=== Orchestration Decision ===")

        ollama_result = ollama_generate(input_text, model=self.state['ollama_model'])
        output = json.loads(ollama_result)["response"]

        results = {
            'entire_text': input_text + "\n\n" + output,
            'output': output
        }

        json_blocks = extract_all_json_blocks(results['entire_text'])
        decision_probabilities = check_json_format(json_blocks[-1].strip())
        
        # Select the function with the highest probability
        if isinstance(decision_probabilities, dict):
            # Build expected functions list dynamically
            expected_functions = [input_function, 'command_correction', 'command_reasoning',
                                'decide_patient_info_action', 'show_patient_info', 'remove_patient_info',
                                'decide_ct_action', 'show_and_move_ct_images', 'zoom_in_and_move_ct_images',
                                'zoom_out_ct_images', 'remove_all_ct_images', 'decide_recon_action',
                                'show_rotate_recon_images', 'show_zoom_in_recon_images', 'show_zoom_out_recon_images',
                                'show_static_recon_images', 'remove_all_recon_images', 'END']
            
            # Add speech_to_text only if not in text mode
            if audio_mode != "text":
                expected_functions.insert(1, 'speech_to_text')
            
            missing_functions = [f for f in expected_functions if f not in decision_probabilities]
            
            # Add missing functions with a probability of 0.0            
            if missing_functions:
                print(f"Warning: Missing functions in response: {missing_functions}")
                for func in missing_functions:
                    decision_probabilities[func] = 0.0

            best_function = max(decision_probabilities.items(), key=lambda x: x[1])
            selected_function = best_function[0]
            selected_probability = best_function[1]
            
            print(f"Selected function: {selected_function} (probability: {selected_probability:.3f})")
            print(f"Execution time: {time.time() - start_time:.2f}s")
            return selected_function
        
        else:
            print("Warning: Response is not a dictionary format")

    def _analyze_current_status(self):
        """Analyze current workflow status."""
        status = []
        audio_mode = self.state.get('audio_mode', 'realtime')
        
        # For text mode, check text_command and save_path
        if audio_mode == "text":
            if 'text_command' not in self.state or not self.state.get('text_command') or self.state.get('text_command') == "None":
                return "No text command available"
            
            if 'save_path' not in self.state or not self.state.get('save_path'):
                return "Text command exists, processing pending"
        else:
            # For audio modes (realtime and synthesized)
            # Check if audio is loaded/recorded
            if 'audio_path' not in self.state or not self.state.get('audio_path') or self.state.get('audio_path') == "None":
                return "No audio available"
            
            # Check if audio is preprocessed (save_path exists)
            if 'save_path' not in self.state or not self.state.get('save_path'):
                audio_action = "realtime recording" if audio_mode == "realtime" else "audio loading"
                return f"Audio path exists, {audio_action} pending"
            
            # Check STT status (not needed for text mode)
            if 'stt_chunks' not in self.state or not self.state.get('stt_chunks'):
                return "Audio recorded, STT pending"
        
        # Check command correction status (common for all modes)
        if 'command_correction_results' not in self.state or not self.state.get('command_correction_results'):
            if audio_mode == "text":
                return "Text command processed, correction pending"
            else:
                return "STT complete, correction pending"
        
        # Check command validity
        if self.state.get('command_correction_results'):
            last_command = list(self.state['command_correction_results'].keys())[-1]
            correction_result = self.state['command_correction_results'][last_command]
            
            # Reset workflow state if command is invalid
            if correction_result['json_output']['validity'] == "INVALID":
                self.initialize()
                if audio_mode == "text":
                    self.state['text_command'] = None
                else:
                    self.state['audio_path'] = None
                    self.state['stt_chunks'] = None
                self.state['command_correction_results'] = {}
                return "Last command invalid, workflow reset - need new input"

        # Check command reasoning status
        if 'command_reasoning_results' not in self.state or last_command not in self.state.get('command_reasoning_results', {}):
            return "Command valid, reasoning pending"

        reasoning_result = self.state['command_reasoning_results'][last_command]
        agent_name = reasoning_result['json_output'].get('agent_name')
        
        if agent_name == 'None':
            return "No agent selected"
        
        # Check agent execution status
        if agent_name in self.agent_config:
            agent_info = self.agent_config[agent_name]
            result_key = agent_info['results_key']
            
            if result_key not in self.state or not self.state.get(result_key):
                return f"{agent_name} selected, action decision pending"
            
            action_result = self.state[result_key][last_command]
            action = action_result['json_output'].get('action')
            action_key = f"{last_command}_{agent_name}_{action}"
            
            if action_key in self.executed_actions:
                return f"{agent_name} completed, workflow finished"
            
            return f"{agent_name} action decided ({action}), execution needed"
        
        return "System ready"

    def execute_function(self, function_name: str):
        """Execute selected function."""
        if function_name == "END":
            print("===== Workflow Completed =====")
            return False
        
        if function_name not in self.functions:
            print(f"Warning: Function '{function_name}' not found")
            return True
            
        print(f"=== Executing: {function_name} ===")
        try:
            updated_state = self.functions[function_name](self.state)
            if updated_state:
                self.state.update(updated_state)
            
            self._mark_action_executed(function_name)

            # Save global state after the execution of agent function
            if function_name in self.agent_action_functions:
                if self.state.get('save_path'):
                    ext = "." + self.state['video_path'].rsplit(".", 1)[-1]
                    save_path = os.path.join(
                        self.state['save_path'], 
                        f"state_{self.state['video_path'].split('/')[-1].split(ext)[0]}.json"
                    )
                    self.state["prev_state_path"] = self.state['save_path']
                    self.save_state_to_json(save_path)
            
            print(f"=== Completed: {function_name} ===")
            return True
        except Exception as e:
            print(f"Error executing {function_name}: {e}")
            return True

    def _mark_action_executed(self, function_name: str):
        """Mark agent action if executed."""
        if not self.state.get('commands'):
            return
        
        # Check if this is an agent action function
        if function_name not in self.agent_action_functions:
            return
            
        last_command = self.state['commands'][-1]
        
        if 'command_reasoning_results' not in self.state or last_command not in self.state['command_reasoning_results']:
            return
            
        reasoning_result = self.state['command_reasoning_results'][last_command]
        agent_name = reasoning_result['json_output'].get('agent_name')
        
        if agent_name not in self.agent_config:
            return
        
        agent_info = self.agent_config[agent_name]
        results_key = agent_info['results_key']
        
        if results_key in self.state and last_command in self.state[results_key]:
            action = self.state[results_key][last_command]['json_output'].get('action')
            if action:
                action_key = f"{last_command}_{agent_name}_{action}"
                self.executed_actions.add(action_key)
                print(f"Marked action as executed: {action_key}")

    def route_agent_action(self):
        """If there is action determination result (action), route to appropriate agent action."""
        if not self.state.get('brief_results'):
            return None
            
        for command, results in self.state['brief_results'].items():
            agent_name = results.get("command_reasoning", {}).get("agent_name")
            
            if agent_name not in self.agent_config:
                continue
            
            agent_info = self.agent_config[agent_name]
            results_key = agent_info['results_key']
            
            if results_key in self.state:
                for cmd, action_results in self.state[results_key].items():
                    action = action_results.get("json_output", {}).get("action")
                    action_key = f"{cmd}_{agent_name}_{action}"
                    
                    if action_key not in self.executed_actions:
                        return agent_info['actions'].get(action)
        
        return None
    
    def save_state_to_json(self, path):
        """Save state to JSON file"""
        def is_json_serializable(value):
            try:
                json.dumps(value)
                return True
            except (TypeError, OverflowError):
                return False
            
        # Save complete local state (only JSON-serializable values)
        json_safe_state = {k: v for k, v in self.state.items() if is_json_serializable(v)}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(json_safe_state, f, ensure_ascii=False, indent=2)
        
        # Save global state (filtered by save_load_keys)
        global_state = {k: v for k, v in self.state.items() if k in self.state["save_load_keys"]}
        
        with open(self.state["global_state_path"], "w", encoding="utf-8") as f: 
            json.dump(global_state, f, ensure_ascii=False)

        # Save timing results (keys containing "time")
        time_results = {k: v for k, v in self.state.items() if "time" in k}
        with open(self.state['save_path']+"/time_results.json", "w", encoding="utf-8") as f: 
            json.dump(time_results, f, ensure_ascii=False)
        
        print(f"Saved global state: {path}")

    def load_global_state(self, global_state_path):
        """Load global state from JSON file"""
        if not os.path.exists(global_state_path):
            print(f"Global state file not found: {global_state_path}")
            return {}

        # Load json file
        with open(global_state_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        # Filter by save_load_keys
        filtered = {k: json_data[k] for k in self.state["save_load_keys"] if k in json_data}
        return filtered

    def run(self):
        """Main execution loop."""
        print("===== Starting SAOP Workflow =====")
        self.initialize()
        
        while True:
            while self.current_iteration < self.max_iterations:
                self.current_iteration += 1
                
                # Check agent action routing
                agent_action = self.route_agent_action()
                if agent_action:
                    next_function = agent_action
                else:
                    # Workflow orchestrator agent
                    next_function = self.get_orchestration_decision()
                
                # Execute function
                should_continue = self.execute_function(next_function)
                if not should_continue:
                    break
                
                time.sleep(0.5)
        
            if self.current_iteration >= self.max_iterations:
                print("===== Maximum iterations reached =====")
            
            return self.state

def load_config(config_path='config.yaml'):
    """Load configuration file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"Warning: Config file '{config_path}' not found.")
    except Exception as e:
        print(f"Error loading config file: {e}")
        raise

def natural_sort_key(file_path):
    """Natural number sorting key function"""
    filename = os.path.basename(file_path)
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[0])
    return 0

def get_audio_files(folder_path, start=None, end=None):
    """Get audio file paths from folder (optionally with range)"""
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a', '*.ogg', '*.aac']
    
    # If range is specified
    if start is not None and end is not None:
        audio_files = []
        for i in range(start, end + 1):
            for ext in ['mp3', 'wav', 'flac', 'm4a', 'ogg', 'aac']:
                audio_path = os.path.join(folder_path, f"{i}.{ext}")
                if os.path.exists(audio_path):
                    audio_files.append(audio_path)
                    break
        return audio_files
    
    # If no range specified - all files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    # Natural sort by numbers
    audio_files.sort(key=natural_sort_key)
    return audio_files

def increment_video_path(video_path):
    """
    Increment the video file number in the path.
    Example: v1_26.mp4 -> v1_27.mp4
    """
    # Extract directory, filename, and extension
    directory = os.path.dirname(video_path)
    filename = os.path.basename(video_path)
    name, ext = os.path.splitext(filename)
    
    # Find all numbers in the filename
    parts = re.split(r'(\d+)', name)
    
    # Increment the last number found
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].isdigit():
            parts[i] = str(int(parts[i]) + 1)
            break
    
    # Reconstruct the filename
    new_name = ''.join(parts)
    new_filename = new_name + ext
    
    # Reconstruct the full path
    if directory:
        new_path = os.path.join(directory, new_filename)
    else:
        new_path = new_filename
    
    return new_path

def setup_initial_state(config, audio_path=None, audio_mode=None, text_command=None):
    """Set up initial system state."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

    random_seed = 42
    set_seed(random_seed)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Get audio configuration from config
    audio_config = config.get('audio', {})
    
    # Determine audio_mode (priority: parameter > config > default)
    if audio_mode is None:
        audio_mode = audio_config.get('audio_mode', 'realtime')
    
    print(f"===== Audio Mode: {audio_mode} =====")

    # Load Models (skip STT for text mode)
    if audio_mode != "text":
        stt_config = config.get('stt')
        stt_model_id = stt_config.get('model_id')
        stt_device = stt_config.get('device')
        stt_compute_type = stt_config.get('compute_type')
        try:
            stt_model = WhisperModel(stt_model_id, device=stt_device, compute_type=stt_compute_type, local_files_only=True)
            print("Loaded model from local files.")
        except Exception as e:
            print("Local model not found. Downloading...")
            stt_model = WhisperModel(stt_model_id, device=stt_device, compute_type=stt_compute_type, local_files_only=False)
    else:
        stt_model = None
        print("Text mode: Skipping STT model loading.")

    # Data configuration
    patient_config = config.get('patient')
    pt_id = patient_config.get('pt_id')
    data_path = patient_config.get('data_path')
    pt_input_path = patient_config.get('pt_input_path')
    
    # Set audio path and text command based on mode
    if audio_mode == "text":
        # Text mode: use text_command
        if text_command is None:
            text_command = audio_config.get('text_command')
        if text_command is None:
            raise ValueError("Text command must be provided for text mode")
        audio_path = "None"
        print(f"Text command: {text_command}")
    elif audio_mode == "synthesized":
        # Synthesized mode: use audio_path
        if audio_path is None:
            audio_path = audio_config.get('audio_path')
        if audio_path is None or audio_path == "None":
            raise ValueError("Audio path must be provided for synthesized mode")
        text_command = "None"
        print(f"Audio path: {audio_path}")
    else:  # realtime
        # Realtime mode: audio_path will be set by realtime_audio function
        audio_path = "None"
        text_command = "None"
        print("Real-time audio recording mode")
    
    # Agent information
    agent_dict = {
        'Patient Info Agent': 'Shows or removes basic patient information such as age, sex, diagnosis of current disease, underlying diseases(comorbidities), operation name, tumor location, pulmonary function test(PFT), etc.',
        'Recon Agent': 'Shows 3D reconstruction (recon) anatomical structures from the target view (anterior/posterior/left/right/inferior/superior/surgical), add/remove anatomical structure (left lower/upper lobe, right lower/middle/upper lobe, nodules, trachea, bronchia), rotate with direction (left/right/up/down/horizontal|left and right/vertical|up and down), and zoom in/out.',
        'CT Agent': 'Shows 2-Dimension CT views (sagittal/coronal/axial) and move the target view with (plus/minus number) or (right/left/forward|anterior/backward|posterior/up|superior/down|inferior).',
    }
    agent_info = ""
    for k, v in agent_dict.items(): agent_info = agent_info + "\t" + k + ": " + v + "\n" 

    # Tumor configuration
    all_structure = ['lung_lower_lobe_left', 'lung_upper_lobe_left', 'lung_lower_lobe_right', 'lung_middle_lobe_right', 'lung_upper_lobe_right', 'lung_nodules', 'lung_trachea_bronchia']
    sheet_name = patient_config.get('sheet_name', {})
    df = pd.read_excel(pt_input_path, sheet_name=patient_config.get('sheet_name', {}))
    tumor_loc = df[df['patient_id'] == pt_id]['tumor_location'].values[0]
    tumor_map = {'LLL': 'lung_lower_lobe_left', 'LUL': 'lung_upper_lobe_left', 'RLL': 'lung_lower_lobe_right', 'RML': 'lung_middle_lobe_right', 'RUL': 'lung_upper_lobe_right'}
    tumor_lobe = re.split(r'[,/]', tumor_loc)
    tumor_lobe = [tumor_map[item.strip()] for item in tumor_lobe if item.strip()]
    all_lobe = ['lung_lower_lobe_left', 'lung_upper_lobe_left', 'lung_lower_lobe_right', 'lung_middle_lobe_right', 'lung_upper_lobe_right']
    non_tumor_lobe = [item for item in all_lobe if item not in set(tumor_lobe)]
    recon_structure = [item for item in all_structure if item not in set(non_tumor_lobe)]

    save_load_keys = ['commands', 'corrected_commands', 'agent_history', 
                     'pt_action', 'pt_fields', 'pt_info', 'display_pt_info', 
                     'ct_display_mode', 'ct_action', 'ct_main_view', 'sagittal_pos', 'coronal_pos', 'axial_pos',
                     'display_recon', 'recon_action', 'recon_structure', 'recon_view', 'recon_rotation', 'recon_target_structure', 'recon_start_scale', 'recon_end_scale', 'grid_zoom_state', 'is_reset_command', 
                     'prev_state_path']

    # Edge laptop configuration (for realtime mode)
    remote_audio_config = config.get('remote_audio', {})
    
    initial_state = {
        "device": device,
        "torch_dtype": torch_dtype,
        "hf_token": config.get('hf_token'),
        "stt_model": stt_model,
        "ollama_model": config.get('ollama_model'),
        "data_path": data_path,
        "video_path": patient_config.get('video_path'),
        "audio_path": audio_path,
        "audio_mode": audio_mode,
        "text_command": text_command,
        "agent_info": agent_info,
        "agent_dict": agent_dict,
        "pt_id": pt_id,
        "sheet_name": sheet_name,
        "pt_input_path": pt_input_path,
        "pt_action": "",
        "pt_fields": {},
        "pt_info": "pt_info",
        "display_pt_info": False,
        "display_recon": False,
        "recon_image_path": os.path.normpath(os.path.join(data_path, "3D_recon")),
        "ct_image_path": os.path.normpath(os.path.join(data_path, "CT")),
        "sagittal_pos": 0,
        "coronal_pos": 0,
        "axial_pos": 0,
        "ct_display_mode": "none",
        "ct_main_view": "none",
        "recon_action": "SHOW_STATIC",
        "recon_structure": recon_structure,
        "recon_view": "surgical",
        "recon_rotation": "static",
        "recon_target_structure": "",
        "recon_start_scale": 1.0,
        "recon_end_scale": 1.0,
        "grid_zoom_state": {
            'history': [],
            'zoom_path': [],
            'current_center': [0.0, 0.0, 0.0],
            'current_scale': 1.0,
            'zoom_level': 0
        },
        "is_reset_command": False,
        "prev_state_path": "",
        "commands": [],
        "corrected_commands": [],
        "agent_history": [],
        "save_load_keys": save_load_keys,
        # Remote audio recording configuration (for realtime mode)
        "edge_user": remote_audio_config.get('edge_user'),
        "edge_ip": remote_audio_config.get('edge_ip'),
        "edge_remote_path": remote_audio_config.get('edge_remote_path'),
        "edge_py_filename": remote_audio_config.get('edge_py_filename'),
        "edge_conda_env": remote_audio_config.get('edge_conda_env'),
        "server_remote_path": remote_audio_config.get('server_remote_path'),
        "server_remote_user": remote_audio_config.get('server_remote_user'),
        "server_remote_ip": remote_audio_config.get('server_remote_ip'),
        "server_remote_port": remote_audio_config.get('server_remote_port'),
    }

    return initial_state

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SAOP Integrated - Real-time, Synthesized Audio, or Text Commands')
    parser.add_argument('config_path', nargs='?', default='config.yaml', help='Path to config file')
    parser.add_argument('--audio_path', help='Path to audio file or folder (overrides config, for synthesized mode)')
    parser.add_argument('--text_command', help='Text command input (overrides config, for text mode)')
    parser.add_argument('--text_commands_file', help='Path to text commands file (overrides config, for text mode)')
    parser.add_argument('-s', '--start', type=int, help='Start index for audio files')
    parser.add_argument('-e', '--end', type=int, help='End index for audio files')
    parser.add_argument('--mode', choices=['realtime', 'synthesized', 'text'], 
                       help='Audio mode (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config_path)
    
    # Get audio configuration from config
    audio_config = config.get('audio', {})
    
    # Determine audio mode (priority: CLI arg > config > default)
    if args.mode:
        audio_mode = args.mode
    else:
        audio_mode = audio_config.get('audio_mode', 'realtime')
    
    print(f"===== Audio Mode: {audio_mode} =====")
    
    # Process based on mode
    if audio_mode == "text":
        # Text mode
        text_commands = []
        
        # Check for text command (priority: CLI > config)
        if args.text_command:
            text_commands = [args.text_command]
        elif args.text_commands_file:  # CLI argument 우선
            text_file = args.text_commands_file
            if os.path.exists(text_file):
                with open(text_file, 'r', encoding='utf-8') as f:
                    text_commands = [line.strip() for line in f if line.strip()]
                print(f"===== Loaded {len(text_commands)} text commands from CLI file =====")
            else:
                print(f"===== Error: Text commands file not found: {text_file} =====")
                exit(1)
        elif audio_config.get('text_command'):
            text_commands = [audio_config.get('text_command')]
        elif audio_config.get('text_commands_file'):
            # Load from config file
            text_file = audio_config.get('text_commands_file')
            if os.path.exists(text_file):
                with open(text_file, 'r', encoding='utf-8') as f:
                    text_commands = [line.strip() for line in f if line.strip()]
                print(f"===== Loaded {len(text_commands)} text commands from config file =====")
            else:
                print(f"===== Error: Text commands file not found: {text_file} =====")
                exit(1)
        else:
            print("===== Error: No text command provided for text mode =====")
            print("Please provide text command via --text_command or --text_commands_file argument or in config.yaml")
            exit(1)
        
        # Get initial video_path from config
        initial_video_path = config.get('patient', {}).get('video_path')
        current_video_path = initial_video_path
        
        # Process each text command
        for idx, text_cmd in enumerate(text_commands, 1):
            if len(text_commands) > 1:
                print(f"\n{'='*60}")
                print(f"Processing text command {idx}/{len(text_commands)}: {text_cmd}")
                print(f"Video path: {current_video_path}")
                print(f"{'='*60}\n")
            
            # Update config with current video_path for this iteration
            config['patient']['video_path'] = current_video_path
            
            # Set up fresh initial state for each command
            initial_state = setup_initial_state(config, text_command=text_cmd, audio_mode="text")
            
            # Execute Workflow Orchestrator Agent
            orchestrator = LLMOrchestrator(initial_state)
            try:
                final_state = orchestrator.run()
            except Exception as e:
                print(f"Error processing command '{text_cmd}': {e}")
                import traceback
                traceback.print_exc()
                continue  # Skip to next command instead of crashing
            
            if len(text_commands) > 1:
                print(f"\n{'='*60}")
                print(f"Completed processing: {text_cmd}")
                print(f"{'='*60}\n")
                
                # Increment video_path for next iteration
                current_video_path = increment_video_path(current_video_path)
    
    elif audio_mode == "synthesized":
        # Synthesized audio mode
        audio_files = []
        
        # Determine audio path (priority: CLI arg > config)
        audio_path_input = args.audio_path if args.audio_path else audio_config.get('audio_path')
        
        if not audio_path_input or audio_path_input == "None":
            print("===== Error: No audio path provided for synthesized mode =====")
            print("Please provide audio path via --audio_path argument or in config.yaml")
            exit(1)
        
        if os.path.isdir(audio_path_input):
            # Folder case
            start_idx = args.start if args.start is not None else audio_config.get('start')
            end_idx = args.end if args.end is not None else audio_config.get('end')
            
            if start_idx is not None and end_idx is not None:
                print(f"===== Loading audio files from {start_idx} to {end_idx} =====")
                audio_files = get_audio_files(audio_path_input, start_idx, end_idx)
                if not audio_files:
                    print(f"===== No audio files found in range {start_idx}-{end_idx} =====")
                    exit(1)
            else:
                audio_files = get_audio_files(audio_path_input)
                if not audio_files:
                    print(f"===== No audio files found in folder: {audio_path_input} =====")
                    exit(1)
            print(f"===== Found {len(audio_files)} audio files =====")
        elif os.path.isfile(audio_path_input):
            # Single file case
            audio_files = [audio_path_input]
            print(f"===== Audio File: {audio_path_input} =====")
        else:
            print(f"===== Error: Path does not exist: {audio_path_input} =====")
            exit(1)
        
        # Get initial video_path from config
        initial_video_path = config.get('patient', {}).get('video_path')
        current_video_path = initial_video_path
        
        # Process each audio file
        for idx, audio_file_path in enumerate(audio_files, 1):
            if len(audio_files) > 1:
                print(f"\n{'='*60}")
                print(f"Processing audio file {idx}/{len(audio_files)}: {os.path.basename(audio_file_path)}")
                print(f"Video path: {current_video_path}")
                print(f"{'='*60}\n")
            
            # Update config with current video_path for this iteration
            config['patient']['video_path'] = current_video_path
            
            # Set up initial state
            initial_state = setup_initial_state(config, audio_path=audio_file_path, audio_mode="synthesized")
            
            # Execute Workflow Orchestrator Agent
            orchestrator = LLMOrchestrator(initial_state)
            final_state = orchestrator.run()
            
            if len(audio_files) > 1:
                print(f"\n{'='*60}")
                print(f"Completed processing: {os.path.basename(audio_file_path)}")
                print(f"{'='*60}\n")
                
                # Increment video_path for next iteration
                current_video_path = increment_video_path(current_video_path)
        
        # Restore original video_path in config
        config['patient']['video_path'] = initial_video_path
        
        if len(audio_files) > 1:
            print(f"\n{'='*60}")
            print(f"All audio files processed successfully! Total: {len(audio_files)}")
            print(f"Video paths used: {initial_video_path} to {increment_video_path(current_video_path)}")
            print(f"{'='*60}")
    
    else:
        # Real-time mode
        print("===== Real-time Audio Mode =====")
        
        # Set up initial state
        initial_state = setup_initial_state(config, audio_mode="realtime")
        
        # Execute Workflow Orchestrator Agent
        orchestrator = LLMOrchestrator(initial_state)
        final_state = orchestrator.run()
        
        print("===== Real-time session completed =====")