# BookSync — Rail & Road Auction Desktop App

BookSync is a desktop client for the Rail & Road Auction catalog system. It processes scanned **books** and **train slides** locally with AI, then syncs the results (titles, authors, ISBNs, descriptions, railroad names, traits) to the shared MongoDB database used by the web app.

All heavy inference (OCR, object detection, vision/text LLMs) runs on the local machine. Only the final catalog records go to the cloud.

---

## Features

**Books**
- Groups scanned images into books by filename (`<BookID>_<page>.jpg`).
- Finds the ISBN with EasyOCR, then fetches metadata from OpenLibrary and Google Books.
- When no ISBN is found, uses a full OCR pipeline: DocLayout-YOLO column cropping, then MinerU/EasyOCR text extraction, then a local LLM (Ollama) that extracts the title, author and edition and writes an auction-ready description.
- Avoids duplicates by cover-image SHA-256 hash and by fuzzy title/ISBN matching against the database.
- Lets you review and edit the generated title and subtitle before saving.

**Train slides**
A 5-step pipeline runs for each slide:
1. YOLO detects and crops the locomotive.
2. Grounding DINO finds the emblem/logo region.
3. A fine-tuned DINOv2 dual-stream model matches the railroad against stored class embeddings.
4. YOLO detects the train's parts.
5. The locomotive type is classified.

From the UI you can also add a **new railroad class**: upload example images, then click *Generate Embeddings*. No retraining is needed.

**App**
- Secure connection token: the web app issues an AES-256-GCM encrypted token that links this desktop client to a user account.
- Background sync with optional automatic folder watching on a configurable interval.
- Model health check and a one-time downloader for missing weights.
- Live pipeline logs and a recent-activity dashboard.
- GPU memory management: models are unloaded between stages so the vision models and the LLM can share one GPU.

---

## Project structure

```
DesktopApp/
├── main.py                    # Entry point — launches the UI
├── run.sh                     # Linux launcher (sets cuDNN path, activates venv)
├── requirements.txt
├── .env.example               # Template for environment variables
├── sync_config.json           # Saved UI/sync settings (folders, interval, watch mode)
├── olive_high.json            # Bundled data file used by the UI
├── final_build.yml            # Alternate CI build workflow
├── DEPLOYMENT_GUIDE.md        # How to install and run the Windows EXE
│
├── frontend/
│   ├── sync_app_ui.py         # Main window (SyncApp): dashboard, uploads, sync, settings, logs
│   ├── isbn_extractor_ui.py   # ISBN OCR + OpenLibrary / Google Books lookup
│   ├── train_slides_ui.py     # Slide lot detail viewer (zoomable canvas)
│   ├── interactive_cropper.py # Manual crop tool
│   └── icons/
│
├── backend/
│   ├── main_mineru_ocr.py     # Per-book OCR pipeline + Ollama title/author/description
│   ├── column_cropper.py      # DocLayout-YOLO layout detection → column crops
│   ├── train_slides_logic.py  # TrainSlidesAnalyzer (YOLO → Grounding DINO → DINOv2 → VLM)
│   ├── dinov2_model.py        # DINOv2 dual-stream model definition
│   ├── db_connector.py        # MongoDB access, retries, duplicate matching
│   ├── book_grouper.py        # Groups image files into books by filename
│   ├── model_manager.py       # Model path resolution, health check, downloads
│   ├── crypto_utils.py        # Token decryption, file hashing
│   ├── error_handler.py       # Global exception logging
│   └── models/torch/hub/      # Offline DINOv2 hub code
│
└── models/                    # Model weights go here (not in the repo — see below)
    ├── crops.json
    ├── auto_cropper.py
    ├── generate_embeddings.py # Builds val_embeddings.pt
    └── train_phase5.py        # DINOv2 fine-tuning script (Phase 5, frozen backbone)
```

---

## Requirements

| Requirement | Notes |
|---|---|
| Python | 3.10 or 3.11 (CI builds with 3.11; `run.sh` expects a `venv310`) |
| GPU | NVIDIA GPU with CUDA recommended. CPU works but is slow. |
| Ollama | Required for the vision and text LLMs — https://ollama.com |
| Disk | ~5 GB free for models and intermediate files |
| MongoDB | Access to the project's MongoDB Atlas cluster |

---

## Setup (from source)

### 1. Clone the repo

```bash
git clone https://github.com/Ayesha-Majeed/Rail-and-Road-Auction.git
cd Rail-and-Road-Auction
```

### 2. Create a virtual environment

`run.sh` expects the venv one level above `DesktopApp/`:

```bash
python3.10 -m venv venv310
source venv310/bin/activate          # Windows: venv310\Scripts\activate
pip install --upgrade pip
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r DesktopApp/requirements.txt
```

### 3. Install Ollama models

```bash
ollama pull minicpm-v:latest   # vision model (covers, slide verification)
ollama pull llama3.2:1b        # text model (author/description)
```

The app starts `ollama serve` itself if it is not running.

### 4. Configure environment

```bash
cp DesktopApp/.env.example DesktopApp/.env
```

Then edit `DesktopApp/.env`:

| Variable | Description |
|---|---|
| `MONGO_URI` | MongoDB connection string |
| `DB_NAME` | Database name |
| `COLLECTION` | Books collection (default `Book Data`) |
| `GOOGLE_BOOKS_API_KEY` | Google Books API key for ISBN metadata |
| `CONNECTION_SECRET_KEY` | Secret used to decrypt connection tokens from the web app |
| `VISION_MODEL` | Ollama vision model (default `minicpm-v:latest`) |
| `TEXT_MODEL` | Ollama text model (default `llama3.2:1b`) |
| `CLEANUP_ENABLED` | `true` (default) deletes per-book intermediate crops after processing |

> ⚠️ Never commit a real `.env` or real credentials. Keep secrets in `.env` locally and in **GitHub Secrets** for CI.

### 5. Model files

The model weights are **not stored on GitHub**. They are shared separately and sent to the client with every build. Place them like this:

| File | Location |
|---|---|
| `yolo11x.pt` — locomotive detection | `models/` |
| `best.pt` — train parts detection | `models/` |
| `phase5_best.pth` — fine-tuned DINOv2 railroad classifier | `models/` |
| `val_embeddings.pt` — reference embeddings per railroad class | `models/` |
| `dinov2_vitb14_pretrain.pth` — DINOv2 pretrained checkpoint | `backend/models/torch/hub/checkpoints/` |
| `doclayout_yolo_docstructbench_imgsz1024.pt` — layout detection | app root (`DesktopApp/`, or next to the EXE), **not** `models/` |

---

## Running

**Linux**
```bash
./DesktopApp/run.sh
```
`run.sh` puts the venv's bundled cuDNN first on the library path. This avoids `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` errors.

**Windows / any OS**
```bash
cd DesktopApp
python main.py
```

**First launch**
1. Paste the **secure connection token** from the web app and wait for *Verified*.
2. Let the model health check finish. Missing models are downloaded once.
3. Choose a mode:
   - **Upload Books** — select a folder of book images.
   - **Upload Slides** — select a folder of train slide lots.
   - **Process Single Book Files / Single Lot of Slides** — run a one-off job.
4. (Optional) Open **Automation Settings**, enable *automatic folder watching*, and set the refresh interval.

---

## Input file naming

Book images must be named `<BookID>_<page>.<ext>`. Hyphens and spaces also work as separators.

```
642_01.jpg   → Book 642, front cover
642_02.jpg   → Book 642, back cover
642_03.jpg   → Book 642, interior page
1189_01.jpg  → Book 1189, front cover
```

Page roles:
- 1 page: `_01` = front cover.
- 2 pages: `_01` = front cover, `_02` = interior.
- 3+ pages: `_01` = front cover, `_02` = back cover, the rest = interior.

Supported formats: `jpg`, `jpeg`, `png`, `tiff`, `bmp`, `webp`.

---

## Output

- **MongoDB**: books go to the `Book Data` collection and slide lots go to `Train Slides Data`, linked to the user from the connection token.
- **Local**: per-book OCR results go to `mineru_results/<BookID>/`:
  ```
  mineru_results/642/
  ├── ocr_output.txt
  ├── ocr_output.json
  ├── color_palette.png
  └── book_metadata.json
  ```
- **Logs**: uncaught errors are written to `syna_error.log`.

---

## Building the Windows EXE

Every push to `main` runs [`.github/workflows/build.yml`](../.github/workflows/build.yml). It builds a PyInstaller `--onedir` app called **BookSync** and uploads `BookSync-Windows-Portable.zip` as a workflow artifact (kept for 3 days).

Configure these repository secrets: `MONGO_URI`, `DB_NAME`, `COLLECTION`, `GOOGLE_BOOKS_API_KEY`.

The models are **not** bundled in the build. They are sent to the client with each build, and go in these places after extracting the ZIP:
- the `models/` folder next to `BookSync.exe`
- `doclayout_yolo_docstructbench_imgsz1024.pt` next to `BookSync.exe`

For full install steps, the offline Grounding DINO cache, and troubleshooting, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).

---

## Trained railroad classes

The DINOv2 railroad classifier (`phase5_best.pth`) was fine-tuned on **50 railroad classes**. These are the 50 classes with the most original (non-augmented) images in the training split. Together they cover about 15,000 training images.

| # | Class | # | Class |
|---|---|---|---|
| 1 | Alton Railroad | 26 | Erie Lackawanna Railway |
| 2 | Amtrak - National Railroad Passenger Corporation | 27 | Florida East Coast Railway |
| 3 | Atchison, Topeka & Santa Fe Railway | 28 | Grand Trunk Western Railroad |
| 4 | Baltimore & Ohio Railroad | 29 | Great Northern Railway |
| 5 | Baltimore Transit Company | 30 | Illinois Central Railroad |
| 6 | BNSF Railway | 31 | Illinois Terminal Railroad |
| 7 | Boston & Maine Railroad | 32 | Kansas City Southern Railway |
| 8 | Burlington Northern Railroad | 33 | Louisville & Nashville Railroad |
| 9 | Burlington Northern Santa Fe | 34 | Milwaukee Road - Chicago, Milwaukee, St. Paul & Pacific Railroad |
| 10 | Canadian National Railway | 35 | Missouri Pacific Railroad |
| 11 | Canadian Pacific Railway | 36 | Missouri-Kansas-Texas Railroad |
| 12 | Chesapeake & Ohio Railway | 37 | New York Central Railroad |
| 13 | Chicago & North Western Railway | 38 | Norfolk & Western Railway |
| 14 | Chicago North Shore & Milwaukee Railroad | 39 | Norfolk Southern Railway |
| 15 | Chicago South Shore & South Bend Railroad | 40 | Pacific Electric Railway |
| 16 | Chicago Transit Authority | 41 | Philadelphia Transportation Company |
| 17 | Chicago, Burlington & Quincy Railroad | 42 | Rio Grande |
| 18 | Chicago, Indianapolis & Louisville - Monon | 43 | Seaboard Coast Line Railroad |
| 19 | Chicago, Milwaukee, St. Paul & Pacific Railroad | 44 | Seaboard System Railroad |
| 20 | Chicago, Rock Island & Pacific Railroad | 45 | Soo Line Railroad |
| 21 | Conrail - Consolidated Rail Corporation | 46 | Southern Pacific Transportation Company |
| 22 | CSX Transportation | 47 | Southern Railway |
| 23 | Dallas Area Rapid Transit | 48 | St. Louis-San Francisco Railway |
| 24 | Denver & Rio Grande Railway | 49 | Toronto Transit Commission |
| 25 | Southern Pacific Transportation Company | 50 | Union Pacific Railroad |

At runtime, matching uses the reference embeddings in `val_embeddings.pt`, which cover **138 railroad classes**: the 50 trained classes plus others added through embeddings. A new class can be added from the app without retraining (see below).

---

## Training & adding railroad classes

- **Add a class without retraining:** in the app, use *Add a new slide class in model*, upload images, then click *Generate Embeddings*. This updates `models/val_embeddings.pt`.
- **Regenerate all embeddings:** `models/generate_embeddings.py`
- **Fine-tune the classifier:** `models/train_phase5.py`. It trains a DINOv2 dual-stream model (full train + emblem crop) with a frozen backbone and triplet + ArcFace loss. Logs go to Weights & Biases when available. The output is `phase5_best.pth`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| App hangs on title/description step | Make sure Ollama is running and both models are pulled |
| CUDA out of memory | Close other GPU apps; the app unloads models between stages, but 8 GB+ VRAM is recommended |
| "DocLayout YOLO missing" | Put the `.pt` file in the app root, not `models/` |
| Grounding DINO fails offline | See *Offline Grounding DINO Setup* in [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) |
| MongoDB timeouts | Check the internet connection and that your IP is allowed in Atlas Network Access |
| Model files missing / fail to load | Re-copy the model files shipped with the build (see *Model files*) |
