# LLM Fine-tuning con GPT-2 sobre clúster HPC

Fine-tuning de GPT-2 sobre el dataset TinyStories, ejecutado con Slurm sobre un clúster HPC de tres nodos con Rocky Linux 9. Incluye scripts para entrenamiento en un solo nodo y entrenamiento distribuido con PyTorch DDP.

Este repositorio acompaña el Capítulo 5 del manual *Automatización de clúster HPC con Ansible* (Universidad Católica de Colombia / Pontificia Universidad Javeriana, 2026).

---

## Hardware del clúster

| Nodo | GPU | VRAM | CPU | RAM |
|------|-----|------|-----|-----|
| master | NVIDIA Quadro P1000 (Pascal) | 4 GB | Intel Xeon W-1290 (20 hilos) | 64 GB |
| worker1 | NVIDIA Quadro P1000 (Pascal) | 4 GB | Intel Xeon W-1290 (20 hilos) | 64 GB |
| worker2 | NVIDIA Quadro P1000 (Pascal) | 4 GB | Intel Xeon W-1290 (20 hilos) | 64 GB |

**SO:** Rocky Linux 9 · **Driver NVIDIA:** 580.126.09 · **CUDA:** 13.0

---

## Entorno de software

El entorno `llm` se gestiona con micromamba. Versiones usadas:

| Paquete | Versión |
|---------|---------|
| Python | 3.11 |
| PyTorch | 2.4.0 (CUDA 12.4) |
| Transformers | 5.0.0 |
| Datasets | — |
| Accelerate | — |

Para verificar que el entorno está operativo y que PyTorch detecta la GPU:

```bash
micromamba run -n llm python -c \
  "import torch; print('torch:', torch.__version__); \
   print('CUDA available:', torch.cuda.is_available()); \
   print('GPU:', torch.cuda.get_device_name(0))"
```

Salida esperada:

```
torch: 2.4.0
CUDA available: True
GPU: Quadro P1000
```

---

## Modelo y dataset

- **Modelo base:** [`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2) (GPT-2 small, 117 M parámetros, licencia MIT)
- **Dataset:** [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories) (licencia CC BY 4.0)

GPT-2 small se eligió porque cabe en los 4 GB de VRAM de la Quadro P1000 con margen para los gradientes y el optimizador (~3 GB en uso durante el entrenamiento). Los modelos más grandes (GPT-2 medium en adelante) superan el presupuesto de memoria disponible.

---

## Estructura del repositorio

```
├── llm_express_3h_gpt2.sbatch   # Job Slurm: fine-tuning en un nodo (3 h)
├── llm_compare_gpt2.sbatch      # Job Slurm: comparación base vs. checkpoint
├── llm_ddp.sbatch               # Job Slurm: fine-tuning distribuido (3 nodos)
├── train_distributed.py         # Script Python para entrenamiento con DDP
└── compare_base_vs_ckpt.py      # Script Python para comparación de generación
```

---

## Uso

### 1. Preparar el directorio de trabajo en NFS

Todos los scripts asumen que el directorio base es `/data/nfs-hpc/llm-express`, accesible desde todos los nodos a través del montaje NFS del clúster. Ajustar la variable `BASE` en cada sbatch si se usa una ruta distinta.

### 2. Fine-tuning en un nodo

Lanza el entrenamiento en la partición `debug` con una sola GPU:

```bash
sbatch llm_express_3h_gpt2.sbatch
```

El job tiene un límite de 3 horas. Con los hiperparámetros por defecto
(`--max_steps 3500`, `--logging_steps 20`) produce hasta tres checkpoints
en `/data/nfs-hpc/llm-express/checkpoints/gpt2-tinystories-express/`.

**Hiperparámetros principales:**

| Parámetro | Valor | Notas |
|-----------|-------|-------|
| `--block_size` | 256 | Longitud de bloque en tokens |
| `--per_device_train_batch_size` | 1 | Limitado por VRAM |
| `--gradient_accumulation_steps` | 16 | Batch efectivo = 16 |
| `--learning_rate` | 5e-5 | |
| `--warmup_steps` | 200 | |
| `--max_steps` | 3 500 | ~3 h en la P1000 |
| `--fp16` | activado | Reduce VRAM a ~3 GB |

### 3. Comparación base vs. checkpoint ajustado

Una vez disponible al menos `checkpoint-2000`, lanza el job de comparación:

```bash
sbatch llm_compare_gpt2.sbatch
```

El job carga el modelo base (`openai-community/gpt2`) y el checkpoint ajustado,
genera texto con tres prompts fijos y escribe la salida en el archivo `.out` del
job. La comparación confirma que el fine-tuning adoptó el estilo de TinyStories.

### 4. Fine-tuning distribuido en tres nodos

Requiere que los tres nodos estén disponibles en la partición `gpu` y que el
directorio NFS sea accesible desde todos ellos.

```bash
sbatch llm_ddp.sbatch
```

`torchrun` coordina un proceso por nodo usando el backend `c10d` sobre la red
interna punto a punto del clúster. El script `train_distributed.py` detecta
automáticamente el rango de cada proceso a través de las variables
`LOCAL_RANK`, `RANK` y `WORLD_SIZE` que inyecta `torchrun`.

> **Nota:** el job distribuido está implementado pero no produjo resultados
> verificados durante el desarrollo de este repositorio.

---

## Archivos de salida

| Ruta | Contenido |
|------|-----------|
| `/data/nfs-hpc/llm-express/checkpoints/gpt2-tinystories-express/` | Checkpoints del entrenamiento de un nodo |
| `/data/nfs-hpc/llm-express/checkpoints/gpt2-tinystories-ddp/` | Checkpoints del entrenamiento distribuido |
| `/data/nfs-hpc/llm-express/hf_cache/` | Caché de modelos y datasets de Hugging Face (~12 GB) |
| `/data/nfs-hpc/llm-express/logs/` | Logs `.out` y `.err` de cada job; logs de GPU por nodo |

---

## Referencia

Eldan, R., & Li, Y. (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* arXiv:2305.07759.
