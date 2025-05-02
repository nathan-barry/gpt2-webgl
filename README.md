# GPT-2 WebGL Inference Demo

A browser-based, WebGL2 implementation of GPT-2.

![Demo](assets/demo.gif)

## 🚀 Features

- Full GPT-2 small (117M) forward pass in the GPU via WebGL2 shaders  
- BPE tokenization using `js-tiktoken` in the browser (no WASM fetch)
- Simple Python script to download the pretrained weights

---

## 📋 Prerequisites

- **Node.js** ≥ 16.x and **npm**  
- **Python** ≥ 3.8  
- A modern browser with WebGL2 support (Chrome, Firefox, Safari, Edge)

---

## 🐍 Download the GPT-2 Weights

We rely on HuggingFace’s `transformers` to pull down the official GPT-2 weights and emit raw `Float32Array` blobs:

1. Install Python dependencies:
   ```bash
   pip install torch numpy transformers
   ```
2. Run the downloader:
   ```bash
   python download_weights.py
   ```
   This will fetch:
   - `wte.bin` (token embeddings)  
   - `wpe.bin` (positional embeddings)  
   - `c_attn_q_w_0.bin` … `c_attn_q_w_11.bin`  
   - `c_attn_k_w_0.bin` … etc.  
   - `lm_head_w.bin`, `lm_head_b.bin`  
   - And a generated `manifest.json` mapping names → URLs  

---

## ⚙️  Front-end Setup with Vite

We use Vite to bundle TS, serve ESM modules & handle `js-tiktoken`:

1. Install JS dependencies:
   ```bash
   npm install
   ```
2. Start the local dev server:
   ```bash
   npm run dev
   ```
3. Open your browser at  
   ```
   http://localhost:5173
   ```

Any changes under `src/` will trigger HMR and live-reload.

---

## 📦 Production Build

When you’re ready to deploy:

```bash
npm run build
```

- Vite will compile and bundle everything into `dist/`  
- Your `index.html` and assets will be output there  
- Simply serve `dist/` with any static file server

---

## 📁 Project Structure

```
.
├── public/                  # static assets served at `/`
│   └── weights/             # GPT-2 weight binaries + manifest.json
├── src/
│   ├── gpt2_webgl.ts        # WebGL2 inference + shaders + tokenizer
│   └── main.ts              # bootstrap: loads manifest, sets up UI
├── download_weights.py      # Python script to fetch & dump weights
├── index.html               # (copied by Vite) entrypoint HTML
├── vite.config.ts           # Vite config
├── package.json
└── tsconfig.json
```
