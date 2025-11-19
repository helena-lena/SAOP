# Surgical Agent Orchestration Platform (SAOP)
Official repository for [Surgical Agent Orchestration Platform for Voice-directed Patient Data Interaction](https://arxiv.org/abs/2511.07392)

<p align="left">
  <img src="https://raw.githubusercontent.com/helena-lena/SAOP/main/docs/SAOP_workflow.png" width="90%">
</p>

## Video Results
👉 [Click here to access the videos](https://helena-lena.github.io/SAOP/)

## Datasets
- **SAOP_command_dataset_with_results.xlsx**: excel file containing command metadata and real-time voice results
  - num: execution order of the command
  - agent: name of the agent responsible for the command
  - command: user command
  - structure: Single or Composite
  - type: Explicit or Implicit or Natural Language Question (NLQ)
  - expression: Baseline or Abbreviation or Paraphrase

- **tts_outputs**: directory containing mp3 audio command files generated using speech_synthesis.py

- **SAOP_synthesized_audio_dataset_with_results.xlsx**: excel file containing results for the synthesized audio files

- **patient_data**: A set of fake patient data for reference
  - **clinical_info.xlsx**: patient clinical information used by Information Retrieval (IR) Agent
  - **P1**: patient ID
    - **full_video**
      - **video_v1.mp4**: sample video
      - **v1_segments**: 10-second video clips
    - **CT**: axial, coronal, and sagittal folders with DICOM files used by the Image Viewer (IV) agent
    - **3D_recon**: 3D anatomical models used by the Anatomy Rendering (AR) agent
      - Generated from the above CT images using 3D Slicer software (https://www.slicer.org/)
      - **lungs.nii.gz**: Segmentation - Total Segmentator - Segmentation task (total) - Apply - Export visible segments (only superior lobe of left lung, inferior lobe of left lung, superior lobe of right lung, middle lobe of right lung, inferior lobe of right lung) to binary label map as NifTI (.nii.gz.) file
      - **lung_nodules.nii.gz**: Segmentation - Total Segmentator - Segmentation task (lung: nodules) - Apply - Export visible segments (only lung_nodules) to binary label map as NifTI (.nii.gz.) file
      - **trachea_bronchia.nii.gz**: Segmentation - Total Segmentator - Segmentation task (lung: vessels) - Apply - Export visible segments (only trachea and bronchus) to binary label map as NifTI (.nii.gz.) file

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
conda create -n saop python=3.12
conda activate saop
pip install -r requirements.txt
```

- Setting Ollama
  - Install Ollama (https://ollama.com/)
  - Run the following commands
  ```bash
  # Download the model
  ollama pull gemma3:27b-it-qat
  # Generate a custom model with predefined parameters
  ollama create vinci:gemma3-27b-it-qat -f ./modelfiles/Modelfile
  # Preload the model on GPU
  curl http://localhost:11434/api/generate -d '{"model": "vinci:gemma3-27b-it-qat", "keep_alive": -1}'
  ```

- Update **config.yaml** file: fill in configuration file with your server details

- Running SAOP code **saop_integrated.py**: supports three modes
  - **Real-time mode**: audio interactino from edge laptop
  ```bash
  # default
  xvfb-run -s "-screen 0 800x600x24" python saop_integrated.py config.yaml
  xvfb-run -s "-screen 0 800x600x24" python saop_integrated.py config.yaml --mode realtime
  ```
  - **Synthesized audio mode**: using synthesized mp3 audio files
  ```bash
  # single file
  xvfb-run -s "-screen 0 800x600x24" python saop_integrated.py config.yaml --mode synthesized --audio_path ./datasets/tts_outputs/en-US-AriaNeural/1.mp3
  # folder
  xvfb-run -s "-screen 0 800x600x24" python saop_integrated.py config.yaml --mode synthesized --audio_path ./datasets/tts_outputs/en-US-AriaNeural
  # folder with range
  xvfb-run -s "-screen 0 800x600x24" python saop_integrated.py config.yaml --mode synthesized --audio_path ./datasets/tts_outputs/en-US-AriaNeural -s 10 -e 20
  ```
  - **Text mode**: using text input
  ```bash
  # single command
  xvfb-run -s "-screen 0 800x600x24" python saop_integrated.py config.yaml --mode text --text_command "Show patient information"
  # text file with multiple commands
  xvfb-run -s "-screen 0 800x600x24" python saop_integrated.py config.yaml --mode text --text_commands_file ./datasets/text_commands.txt
  ```

### 3. Evaluation
- Execute **evaluation.ipynb** to check Multi-level Orchestration Evaluation Metric (MOEM)

## Citation
If you find this repository useful, please consider citing this paper:

```bibtex
@article{park2025saop,
  title={Surgical Agent Orchestration Platform for Voice-directed Patient Data Interaction},
  author={Park, Hyeryun and Gu, Byung Mo and Lee, Jun Hee and Choi, Byeong Hyeon and Kim, Sekeun and Kim, Hyun Koo and Kim, Kyungsang},
  journal={arXiv preprint arXiv:2511.07392},
  year={2025}
}
```

