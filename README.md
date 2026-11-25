# Kuppai-to-Kaasu: Offline Voice Assistant for Scrap Segregation & Resale Value

**Arm Bharat AI-SoC Challenge 2026 · Phase 1 · PS2: Smart Recycling & Waste-Segregation Assistant (Multimodal Interfaces)**

> Point your phone at waste, ask a question by voice or text, and hear how to segregate it, how to prepare it, and what it is worth as scrap. Fully offline, running on an Arm-based Android phone.

---

## 1. Problem

Many households and small workers do not know which waste is recyclable or what it is worth. Most recycling apps need internet, so they fail in low-connectivity areas and raise privacy concerns. As a result, recyclable scrap is thrown away or sold under-valued, and people have little reason to segregate.

## 2. Solution

An on-device multimodal assistant:

1. **See** – a camera photo is classified into a waste category.
2. **Value** – an offline scrap-rate table gives an approximate resale value (Rs/kg).
3. **Advise** – a Llama-family model running on-device via **ExecuTorch** explains how to prepare and dispose of the item and answers follow-up questions.
4. **Speak** – the answer is shown on screen and spoken aloud; users can ask by voice or text.

No internet is needed at run time. The app does not request the `INTERNET` permission.

## 3. Key Features

- Multimodal input: camera, voice, and text
- Waste classes: plastic, paper/cardboard, metal, glass, e-waste, organic/other
- Scrap-value estimate from an editable table (`assets/scrap_rates.json`, approximate static values)
- On-device LLM advisory: Llama 3.2 1B Instruct, 4-bit quantised, ExecuTorch with the XNNPACK backend (Arm CPU)
- Offline English speech recognition and text-to-speech
- Private by design: no data leaves the device

## 4. Architecture

```
Camera ──► Vision classifier (MobileNet, TFLite) ──► waste class + confidence
                                                          │
Voice ─► Android offline speech recognition ─► query ─────┤
Text  ────────────────────────────────────────────────────┤
                                                          ▼
                       Prompt builder (class + query + scrap-rate facts)
                                                          │
                                                          ▼
              Llama 3.2 1B (.pte) on ExecuTorch (XNNPACK, Arm CPU)
                                                          │
                                                          ▼
                    Answer text ─► Android offline TTS ─► screen + speaker
```

The vision pipeline and the LLM pipeline are separate modules.

## 5. Tech Stack

| Layer | Choice |
|---|---|
| Target hardware | Arm-based Android phone (arm64-v8a) |
| LLM runtime | ExecuTorch (XNNPACK backend) |
| LLM | Llama 3.2 1B Instruct, 8da4w quantised, 4-bit embeddings |
| Vision | MobileNetV2 fine-tuned on a public waste dataset (TrashNet) |
| Speech | Android offline SpeechRecognizer and TextToSpeech (English, India) |
| App | Kotlin, Android Studio |
| Export environment | WSL2 Ubuntu on Windows (build time only) |

## 6. Offline and Arm Compliance

- Runs on an Arm-based phone: **Device:** _model_, **SoC:** _chipset_, **ABI:** `arm64-v8a`.
- Internet is used only on the development laptop (downloads, export, training). At run time the phone is in airplane mode in the demo video.
- `AndroidManifest.xml` has no `INTERNET` permission.
- The model, tokenizer, classifier, and data files are stored on the device.

## 7. Repository Structure

```
.
├── README.md
├── android-app/            # Kotlin app (camera, classifier, LLM, voice)
│   └── app/src/main/assets/
│       ├── waste_classifier.tflite
│       └── scrap_rates.json
├── model-export/           # Llama export commands and notes
├── vision-training/        # classifier training notebook
├── docs/                   # report, diagrams, screenshots
└── demo/                   # demo video link
```

The `.pte` model and tokenizer are not stored in git (large files). See setup below.

## 8. Setup

**Requirements:** Windows laptop with WSL2 (Ubuntu), Android Studio, Git, Python 3.10+, an arm64 Android phone with 8 GB RAM, Hugging Face access to Llama 3.2 (licence accepted).

1. **Export the LLM (WSL2 Ubuntu)**
   ```bash
   git clone https://github.com/pytorch/executorch.git ~/executorch
   cd ~/executorch && git submodule update --init --recursive && ./install_executorch.sh
   # download Llama 3.2 1B Instruct (Meta link or Hugging Face), then:
   python3 -m examples.models.llama.export_llama \
     --checkpoint $CK/consolidated.00.pth --params $CK/params.json \
     -kv --use_sdpa_with_kv_cache -X --xnnpack-extended-ops -qmode 8da4w \
     --group_size 64 -d fp32 \
     --metadata '{"get_bos_id":128000, "get_eos_ids":[128009, 128001, 128006, 128007]}' \
     --embedding-quantize 4,32 \
     --output_name="llama3_1B_kv_sdpa_xnn_qe_4_64_1024_embedding_4bit.pte" \
     --max_seq_length 1024 --max_context_length 1024
   ```
2. **Train/export the classifier** with `vision-training/train.ipynb` and place `waste_classifier.tflite` in the app assets.
3. **Install the English (India) offline speech pack** on the phone (Settings → Languages → Speech services).
4. **Push the model** to the phone:
   ```
   adb shell mkdir -p /data/local/tmp/llama
   adb push <model>.pte /data/local/tmp/llama/
   adb push tokenizer.model /data/local/tmp/llama/
   ```
5. **Build and install** the app from `android-app/` in Android Studio.
6. **Switch on airplane mode** and run.

## 9. Usage

1. Open the app and tap the camera to photograph an item, or tap the mic / type a question.
2. The app shows the waste class, the approximate scrap value, and the advice.
3. The answer is read aloud. Ask follow-up questions by voice or text.

## 10. Results (replace with your measured values)

| Metric | Value |
|---|---|
| Classifier test accuracy | _TBD_ |
| Image-to-answer latency | _TBD_ s |
| LLM generation speed | _TBD_ tokens/s |
| Peak app RAM | _TBD_ MB |
| LLM model size / classifier size | _TBD_ MB / _TBD_ MB |
| Works in airplane mode | _Yes/No_ |
| Test device (model, SoC, RAM) | _TBD_ |

## 11. Limitations

- Scrap rates are indicative static values, not live market prices.
- Single-item recognition; accuracy depends on lighting and image quality.
- English only in Phase 1.
- The 1B model can produce imperfect advice; local municipal rules are not included.

## 12. Phase 2 Roadmap

- Indian languages (Tamil, Hindi) with offline speech and a multilingual model
- Region-wise, updatable scrap-rate packs and a nearby buyer directory
- Multi-item detection and a stronger vision model
- Further Arm optimisation and benchmarking (KleidiAI/NEON/SME2) across devices
- Municipal segregation-rule packs

## 13. Team

_Team name · Team leader · Members · Institution/Organisation_

## 14. Licence & Acknowledgements

Code: _choose a licence (e.g. MIT)_. Llama models are subject to the Meta Llama Community Licence. Built with ExecuTorch, Android, and public waste datasets (TrashNet). Submitted for the Arm Bharat AI-SoC Challenge 2026.
