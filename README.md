Run the model

```bash
uv run finetuning.py --data_path '/Users/fredericlegrand/Documents/GitHub/ng-video-lecture/data/messages' --your_name "Frederic Legrand" --model Qwen/Qwen-1_8B --num_epochs 3 --batch_size 2

```

Inference

```bash
uv run test_model.py --model_path ./fine_tuned_model/checkpoints/checkpoint-500
```
