# Surgical Agent Orchestration Platform (SAOP)
Official repository for [Surgical Agent Orchestration Platform for Voice-directed Patient Data Interaction]

## Datasets
- SAOP_command_dataset_with_results.xlsx: excel file of command metadata and associated results
  - num: execution order of the command
  - agent: name of agent responsible for the command
  - command: user command
  - structure: Single or Composite
  - type: Explicit or Implicit or Natural Language Question (NLQ)
  - expression: Baseline or Abbreviation or Paraphrase
    
- tts_outputs: directory containing synthesized .mp3 audio files generated from the command dataset using speech_synthesis.py

## Codes
### 1. Synthesizing .mp3 audio files from the command dataset
- Using the default model (en-US-AriaNeural)
```bash
python speech_synthesis.py
```

- Using a custom TTS model 
  (List of supported models: https://gist.github.com/BettyJJ/17cbaa1de96235a7f5773b8690a20462)
```bash
python speech_synthesis.py --model en-US-AriaNeural
python speech_synthesis.py --model en-US-GuyNeural
python speech_synthesis.py --model en-US-JennyNeural
python speech_synthesis.py --model en-US-ChristopherNeural
```

### 2. Run SAOP
- Setting virtual environments
```bash
pip install -r requirements.txt
```

- Setting Ollama
  - Install ollama: https://ollama.com/
  - Download model: ollama pull gemma3:27b-it-qat
  - Generate a new model with predefined parameters: ollama create vinci:gemma3-27b-it-qat -f ./modelfiles/Modelfile
  - Preload the model on GPU: curl http://localhost:11434/api/generate -d '{"model": "vinci:gemma3-27b-it-qat", "keep_alive": -1}'

- Run SAOP
Coming soon!

### 3. Evaluation
Coming soon!

## Video Results
👉 [Click here to access the videos](https://helena-lena.github.io/SAOP/)
