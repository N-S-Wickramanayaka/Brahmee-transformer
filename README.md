# Brahmee-transformer
Identifying sinhala brahmee carvings using visformer

### Launch arguements for converter.py

python converter.py `
    --input_dir "dataset_original" `
    --output_dir "dataset_stone" `
    --texture "textures/stone1.jpg" "textures/stone2.jpg" `
    --workers 8

  Brahmee model/
│
├── converter.py
│
├── textures/
│   ├── stone1.jpg
│   └── stone2.jpg
│
├── dataset_original/
│   ├── train/
│   ├── val/
│   └── test/
│
└── dataset_stone/
