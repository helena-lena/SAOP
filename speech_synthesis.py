"""
TTS Audio File Generator from Excel Commands

Usage:
    # Use default model (en-US-AriaNeural)
    python speech_synthesis.py
    
    # Specify a custom TTS model (other models: https://gist.github.com/BettyJJ/17cbaa1de96235a7f5773b8690a20462)
    python speech_synthesis.py --model en-US-AriaNeural
    python speech_synthesis.py --model en-US-GuyNeural
    python speech_synthesis.py --model en-US-JennyNeural
    python speech_synthesis.py --model en-US-ChristopherNeural
    
    # View help
    python speech_synthesis.py --help

Description:
    This script reads commands from an Excel file and generates TTS audio files for each command. 
    If a command is not given, it creates a 4-second silent audio file.
"""

import pandas as pd
import edge_tts
import os
import asyncio
import argparse
from pydub import AudioSegment

# Parse arguments
parser = argparse.ArgumentParser(description='Generate TTS audio files from excel commands')
parser.add_argument('--model', type=str, default='en-US-AriaNeural',
                    help='TTS model to use (default: en-US-AriaNeural). Options: en-US-AriaNeural, en-US-GuyNeural, en-US-JennyNeural, en-US-ChristopherNeural. Other options: https://gist.github.com/BettyJJ/17cbaa1de96235a7f5773b8690a20462.')
args = parser.parse_args()

# TTS model configuration
TTS_MODEL = args.model

# Load data & sort by num column
df = pd.read_excel("./datasets/SAOP_command_dataset_with_results.xlsx", sheet_name="all_commands")
df_sorted = df.sort_values('num').reset_index(drop=True)

# Convert to dictionary format
command_dict = dict(zip(df_sorted['num'], df_sorted['command']))

# Create output directory
output_dir = f"./datasets/tts_outputs/{TTS_MODEL}"
os.makedirs(output_dir, exist_ok=True)

async def generate_tts_file(num, text, model):
    """Generate single TTS file"""
    output_file = os.path.join(output_dir, f"{num}.mp3")
    
    # Check for NaN (given no command)
    if pd.isna(text):
        # Create 4-second silent file
        silent = AudioSegment.silent(duration=4000)
        silent.export(output_file, format="mp3")
        print(f"Silent file created: {output_file}")
    else:
        # Generate regular TTS
        try:
            tts = edge_tts.Communicate(text, model)
            await tts.save(output_file)
            print(f"Generated: {output_file}")
        except Exception as e:
            print(f"Error occurred (num={num}): {e}")
            return None
    
    return output_file

async def generate_all_tts_sequential():
    """Generate TTS files sequentially for all commands"""
    results = []
    total = len(command_dict)
    
    for idx, (num, command) in enumerate(command_dict.items(), 1):
        print(f"[{idx}/{total}] Processing...")
        result = await generate_tts_file(num, command, TTS_MODEL)
        results.append(result)
        
        # Small delay to prevent API rate limiting
        await asyncio.sleep(0.5)
    
    print(f"\nTotal of {len(command_dict)} TTS files have been generated!")
    return results

# Main execution
if __name__ == "__main__":
    asyncio.run(generate_all_tts_sequential())