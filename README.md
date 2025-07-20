Run the model

```bash
uv run finetuning.py --data_path '/Users/fredericlegrand/Documents/GitHub/ng-video-lecture/data/messages' --your_name "Frederic Legrand" --model gpt2-medium --num_epochs 1 --batch_size 8
```

Inference

```bash
uv run test_model.py --model_path ./fine_tuned_model/checkpoints/checkpoint-500
```
