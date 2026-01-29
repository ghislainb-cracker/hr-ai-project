# Maya - AI-Powered HR & Recruitment Platform

An intelligent conversational AI assistant built on fine-tuned Llama 3.1 8B for automated HR onboarding and talent matching, deployed on AWS SageMaker.

## Overview

Maya is a WhatsApp-integrated AI assistant that streamlines the hiring process by automating user onboarding across multiple personas:

- **HR Professionals** - Collect job requirements, manage hiring workflows
- **Job Candidates** - Gather CVs, salary expectations, preferences
- **Clients** - Capture freelance project requirements
- **Freelancers** - Collect portfolios, skills, availability

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│           WhatsApp / Chat Interface                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│     Maya AI (Llama 3.1 8B Fine-tuned with QLoRA)       │
│     - Context-aware conversation management             │
│     - Multi-flow user onboarding                        │
│     - Natural language information extraction           │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Backend Matching Engine                    │
│     - Candidate ↔ Job matching                          │
│     - Freelancer ↔ Project matching                     │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
hr-ai-project/
├── docker/
│   ├── train-image/          # Training container (SageMaker)
│   │   ├── Dockerfile
│   │   └── buildspec.yml
│   └── infer-image/          # Inference container (vLLM)
│       ├── Dockerfile
│       └── buildspec.yml
├── docs/                     # API documentation
│   ├── Maya_NodeJS_v2.0.docx
│   └── Maya_Python_v2.0.docx
├── model/
│   └── adapters/             # LoRA adapter configuration
├── notebooks/
│   ├── training_reference.ipynb    # Training guide
│   ├── deployment.ipynb            # SageMaker deployment
│   └── client_integration.ipynb    # API integration examples
├── prompts/                  # System prompts for each user flow
│   ├── hr.txt
│   ├── candidate.txt
│   ├── client.txt
│   └── freelancer.txt
├── scripts/
│   └── train.py              # QLoRA fine-tuning script
├── requirements.txt
└── README.md
```

## Tech Stack

### Machine Learning
- **Base Model**: Meta Llama 3.1 8B Instruct
- **Fine-tuning**: QLoRA (4-bit quantization) with PEFT
- **Training Framework**: Hugging Face Transformers, TRL (SFTTrainer)
- **Inference**: vLLM for optimized serving

### Infrastructure
- **Training & Hosting**: AWS SageMaker
- **Model Storage**: Amazon S3
- **Containerization**: Docker
- **CI/CD**: AWS CodeBuild

## Model Details

| Parameter | Value |
|-----------|-------|
| Architecture | Llama 3.1 8B |
| Hidden Size | 4096 |
| Layers | 32 |
| Attention Heads | 32 |
| Context Length | 131,072 tokens |
| Quantization | 4-bit (bitsandbytes) |
| LoRA Rank | 16 |
| LoRA Alpha | 16 |

## Training

### Prerequisites
```bash
pip install -r requirements.txt
```

### Fine-tuning with QLoRA
```bash
python scripts/train.py \
    --model_id meta-llama/Llama-3.1-8B-Instruct \
    --train_file data/train.jsonl \
    --eval_file data/eval.jsonl \
    --output_dir ./output \
    --s3_bucket your-bucket-name
```

### Key Training Parameters
- **Learning Rate**: 1e-5
- **Epochs**: 1 (to prevent overfitting)
- **Batch Size**: 4 (with gradient accumulation of 2)
- **Dropout**: 0.1
- **Evaluation**: Every 10 steps with best model selection

## Deployment

### Build Docker Images
```bash
# Training image
cd docker/train-image && docker build -t maya-train .

# Inference image
cd docker/infer-image && docker build -t maya-infer .
```

### Deploy to SageMaker
Refer to `notebooks/deployment.ipynb` for detailed deployment instructions.

## API Integration

See `notebooks/client_integration.ipynb` for examples of:
- Endpoint invocation
- Request/response handling
- Multi-turn conversation management

## Prompt Engineering

Each user flow has a dedicated system prompt in `/prompts/`:

- **hr.txt** - Onboards hiring managers, collects job requirements
- **candidate.txt** - Collects CV, salary expectations, preferences
- **client.txt** - Captures freelance project requirements
- **freelancer.txt** - Gathers portfolio, skills, availability

### Key Prompt Features
- Context-aware (checks conversation history)
- Anti-hallucination safeguards
- Natural conversational flow
- Mid-flow transition support
- WhatsApp formatting (bold, italics)

## License

Proprietary - All rights reserved.

## Contact

For questions or support, please reach out to the development team.
