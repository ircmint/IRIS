# Zero-Shot — R1 / Florence2

Object detection and scene understanding using Microsoft Florence-2 on Rider1_NJ frames.

## Files

| File | Purpose |
|------|---------|
| `run_florence2.py` | Extract frames from candidate event clips and run Florence-2 detection |
| `run_florence2.sh` | SLURM job script for Ada HPC |

## Usage

```bash
# Submit as SLURM job
sbatch run_florence2.sh

# Or run directly on GPU node
source /ssd_scratch/abhishek.vedula/envs/florence/bin/activate
python run_florence2.py
```

## Task Modes

Florence-2 supports multiple tasks — this script uses:
- `<OD>` — Object Detection (detect vehicles, pedestrians, obstacles)
- `<CAPTION>` — Image captioning for scene description
- `<GROUNDING_CAPTION>` — Caption with bounding boxes

## Output

Results saved to `results_florence2.csv` with columns:
`event_id`, `frame_path`, `detections`, `caption`, `score`
