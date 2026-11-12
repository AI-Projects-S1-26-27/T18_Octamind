# Data ingestion notes (Role 1)

Status of the raw datasets as of **2026-11-12**.

## Quick status

| Dataset | Format | Status | Expected path |
|---|---|---|---|
| MuSiQue-Ans | JSONL | Downloaded; sampled (600 = 200 per hop 2-4) | `data/raw/musique/musique_ans_v1.0_dev.jsonl` |
| 2WikiMultiHopQA | JSON | Downloaded; sampled (200) | `data/raw/twiki/dev.json` |
| HotpotQA (distractor) | JSON (rebuilt from parquet) | Downloaded; sampled (200) | `data/raw/hotpotqa/hotpot_dev_distractor_v1.json` |

## Download sources that work

```bash
mkdir -p data/raw/musique data/raw/twiki data/raw/hotpotqa
curl -L -o data/raw/musique/musique_ans_v1.0_dev.jsonl \
  https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/main/musique_ans_v1.0_dev.jsonl
curl -L -o data/raw/twiki/dev.json \
  https://huggingface.co/datasets/voidful/2WikiMultihopQA/resolve/main/dev.json
curl -L -o data/raw/hotpotqa/validation-00000-of-00001.parquet \
  https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet
python -m data.convert_hotpotqa          # parquet -> native HotpotQA JSON (needs pyarrow)
```

- MuSiQue: the file's SHA-256 (`15fa6379...e48abf3b`) matches the HuggingFace LFS hash. The
  official release is also on Google Drive (`gdown` id `1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h`).
- 2Wiki: the official release is the Dropbox `data.zip`.
- HotpotQA: the official CMU URLs (502) and the hotpotqa.github.io links are dead, but the
  official HuggingFace repo `hotpotqa/hotpot_qa` serves the distractor dev set (7,405 questions)
  as parquet. `data/convert_hotpotqa.py` rewrites it into the original JSON layout, so the loader
  is unchanged. (The earlier blocker used the `thinkall/HotpotQA` mirror, whose parquet links
  are broken.)

## Paragraph ids are unique across the dataset (fixed 2026-11-12)

Ids were `<dataset>::<title>`, made unique only *within* one question. MuSiQue (and 2Wiki, in
the full dev set) reuse a title for *different* paragraphs in different questions, so one id
named several texts: in the MuSiQue sample 287 ids carried 2+ texts and 284 of 600 questions had
an ambiguous gold paragraph id, which would corrupt edge provenance and evidence
recall/precision. Now `make_ids_globally_unique` (run by every loader) gives every variant of
such a title the id `<dataset>::<title>#<sha1(text)[:8]>`; titles with a single text keep the
clean id. `validate_records` rejects any id that names two texts. Question sampling is
unchanged (it depends only on question ids).

## Running the sampler

```bash
python -m data.sampling              # all three datasets
python -m data.sampling --skip-missing
python -m pytest tests/test_data.py -q
```
