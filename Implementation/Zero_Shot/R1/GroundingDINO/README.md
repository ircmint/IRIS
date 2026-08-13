# Zero-Shot — R1 / GroundingDINO

Open-vocabulary object detection using Grounding DINO on Rider1_NJ frames.

## Files

| File | Purpose |
|------|---------|
| `run_grounding_dino.py` | Run GroundingDINO with traffic-relevant text queries on event frames |
| `run_grounding_dino.sh` | SLURM job script for Ada HPC |

## Usage

```bash
sbatch run_grounding_dino.sh

# Or directly
python run_grounding_dino.py
```

## Text Queries Used

GroundingDINO detects objects specified by text prompts. Queries used:
```
"vehicle . car . truck . motorcycle . pedestrian . pothole . obstacle"
```

## Output

Results saved to `results_grounding_dino.csv` with columns:
`event_id`, `frame_path`, `detected_objects`, `bboxes`, `confidence_scores`
