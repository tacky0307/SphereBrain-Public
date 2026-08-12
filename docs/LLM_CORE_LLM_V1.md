# LLM → Core → LLM v1

## Purpose

This is an isolated experiment that uses an LLM as SphereBrain's input and output interface while keeping learning inside SphereBrain Core.

```text
text
  -> OpenAI embedding
  -> fixed random projection (128 dimensions)
  -> stable signed stimulus nodes
  -> isolated SphereBrain Core
  -> Core activity overlap observer
  -> OpenAI Decoder
  -> text
```

## Research safety boundary

The experiment does not modify the existing research data.

Existing paths left untouched:

- `data/brain.json`
- `data/brain_semantic_v2.json`
- `data/semantic_v2.db`
- `semantic_encoder_v2.py`
- `semantic_encoder_v2_lab.py`

Experiment-only data:

- `data/llm_core_v1/brain.json`
- `data/llm_core_v1/experiences.db`
- `data/llm_core_v1/projection.npy`

Returning to the previous research means launching the previous Semantic Encoder v2 program. No conversion or rollback operation is required.

## Setup

Set `OPENAI_API_KEY` as an environment variable. Do not write the key into source files or commit it to GitHub.

Optional environment variables:

- `SPHERE_EMBEDDING_MODEL`
- `SPHERE_DECODER_MODEL`
- `SPHERE_STIMULUS_DIM`
- `SPHERE_SOURCE_COUNT`
- `SPHERE_PROJECTION_SEED`

Run:

```text
run_llm_core_lab.bat
```

The lab opens at `http://127.0.0.1:5078`.

## What v1 proves and does not prove

v1 checks whether similar LLM embeddings produce overlapping Core routes and whether experience changes later Core activity.

The Decoder receives only experience candidates selected by Core overlap. However, the candidates still include their original text so that the LLM can verbalize the observation. Therefore this version is an experimental bridge, not yet a fully language-free Decoder.

## Recommended comparison

Run the same training and probe set under:

1. Semantic Encoder v2
2. LLM → Core probe without Decoder
3. LLM → Core → LLM

Record separately:

- input stimulus
- activated nodes
- traversed edges
- overlap ranking
- Decoder input
- Decoder output

This separation is required to distinguish Encoder effects, Core effects, and Decoder completion.
