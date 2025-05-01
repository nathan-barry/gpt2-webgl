import { encodingForModel, Tiktoken } from "js-tiktoken";

const DEBUG_LAYER = 11;

interface WeightManifest {
  [key: string]: string;
}

export class GPT2WebGL {
  private gl: WebGL2RenderingContext;
  private programs: { [name: string]: WebGLProgram } = {};
  private textures: { [name: string]: WebGLTexture } = {};
  private weightArrays: { [key: string]: Float32Array } = {};
  private manifest: WeightManifest;

  // Model config
  private nEmbeds = 768;
  private nLayers = 12;
  private vocabSize = 50257;

  // tiling params per weight
  private tileInfo: { [name: string]: { width: number; height: number } } = {};

  // full-screen quad
  private quadBuffer!: WebGLBuffer;

  // tokenizer
  private tokenizer: Tiktoken;

  constructor(canvas: HTMLCanvasElement, manifest: WeightManifest) {
    const gl = canvas.getContext("webgl2");
    if (!gl) throw new Error("WebGL2 not supported");

    // enable float‐buffer rendering
    if (!gl.getExtension("EXT_color_buffer_float")) {
      throw new Error(
        "EXT_color_buffer_float not supported — cannot render to float textures"
      );
    }
this.gl = gl;
    this.manifest = manifest;

    const maxTex = this.gl.getParameter(this.gl.MAX_TEXTURE_SIZE);
    console.log("MAX_TEXTURE_SIZE =", maxTex);

    // setup tokenizer
    this.tokenizer = encodingForModel("gpt2");

    // build shaders and quad
    this._initShaders();
    this._initDisplayPass();
    this._initQuad();
  }

  private _initQuad() {
    const gl = this.gl;
    const buf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW
    );
    this.quadBuffer = buf;
  }

  private _initDisplayPass() {
    const vsrc = `#version 300 es
      in vec2 a_position;
      out vec2 v_uv;
      void main() {
        v_uv = a_position * .5 + .5;
        gl_Position = vec4(a_position, 0, 1);
      }`;
    const fsrc = `#version 300 es
      precision highp float;
      in vec2 v_uv;
      uniform sampler2D u_tex;
      out vec4 o;
      void main() {
        o = texture(u_tex, v_uv);
      }`;
    this.programs.display = this._createProgram(vsrc, fsrc);
  }

  /** Load all weights as float32 textures */
  async loadWeights(): Promise<void> {
    console.log("LOADING WEIGHTS");

    const gl = this.gl;
    const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;

    for (const name in this.manifest) {
      const url = this.manifest[name];
      const res = await fetch(url);
      const buf = await res.arrayBuffer();
      const arr = new Float32Array(buf);
      this.weightArrays[name] = arr;

      // determine dims: only keep the 768×768 shortcut for c_attn and c_proj
      let W: number, H: number;
      if (name.startsWith("c_attn") || name.startsWith("c_proj")) {
        W = this.nEmbeds;
        H = this.nEmbeds;
      } else if (name.startsWith("mlp_fc")) {
        W = 4 * this.nEmbeds;
        H = this.nEmbeds;
      } else if (name.startsWith("mlp_proj")) {
        W = this.nEmbeds;
        H = 4 * this.nEmbeds;
      } else {
        // generic tiling for everything else (including lm_head_w):
        const total = arr.length;
        W = Math.min(total, maxTex);
        H = Math.ceil(total / W);
      }
      this.tileInfo[name] = { width: W, height: H };

      // pad to W*H if needed
      const needed = W * H;
      const uploadData =
        arr.length === needed
          ? arr
          : (() => {
              const tmp = new Float32Array(needed);
              tmp.set(arr, 0);
              return tmp;
            })();

      // upload GPU texture
      const tex = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.R32F,
        W,
        H,
        0,
        gl.RED,
        gl.FLOAT,
        uploadData
      );
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      // clamp to edge to be safe
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      this.textures[name] = tex;
    }

    console.log("LOADING WEIGHTS DONE");
  }

  /** Compile & link all our little passes */
  private _initShaders() {
    console.log("INITIALIZING SHADERS");

    const vsrc = `#version 300 es
      in vec2 a_position;
      out vec2 v_uv;
      void main(){
        v_uv = a_position * .5 + .5;
        gl_Position = vec4(a_position,0,1);
      }`;

    // matMul
    const matmul = `#version 300 es
      precision highp float;
      uniform sampler2D u_A, u_B;
      uniform int u_K;
      in vec2 v_uv;
      out vec4 outColor;
      void main(){
        ivec2 c = ivec2(gl_FragCoord.xy);
        float sum = 0.0;
        for(int k=0;k<u_K;++k){
          sum += texelFetch(u_A, ivec2(k,c.y),0).r
               * texelFetch(u_B, ivec2(c.x,k),0).r;
        }
        outColor = vec4(sum,0,0,1);
      }`;

    // addBias
    const addBias = `#version 300 es
      precision highp float;
      uniform sampler2D u_X, u_bias;
      in vec2 v_uv; out vec4 o;
      void main(){
        ivec2 c = ivec2(gl_FragCoord.xy);
        float x = texelFetch(u_X,c,0).r;
        float b = texelFetch(u_bias, ivec2(c.x,0),0).r;
        o = vec4(x+b,0,0,1);
      }`;

    // gelu
    const gelu = `#version 300 es
      precision highp float;
      uniform sampler2D u_X;
      in vec2 v_uv; out vec4 o;
      void main(){
        ivec2 c = ivec2(gl_FragCoord.xy);
        float x = texelFetch(u_X,c,0).r;
        float t = 0.5*(1.0 + tanh(0.79788456*(x + 0.044715*x*x*x)));
        o = vec4(x*t,0,0,1);
      }`;

    // layerNorm
    const layerNorm = `#version 300 es
      precision highp float;
      uniform sampler2D u_X, u_gamma, u_beta;
      uniform int u_N;
      in vec2 v_uv;
      out vec4 o;
      void main(){
        // c.x = feature index, c.y = sequence position
        ivec2 c = ivec2(gl_FragCoord.xy);

        // 1) compute mean over features for this token (row = c.y)
        float mean = 0.0;
        for(int i = 0; i < u_N; ++i) {
          mean += texelFetch(u_X, ivec2(i, c.y), 0).r;
        }
        mean /= float(u_N);

        // 2) compute variance over the same
        float var = 0.0;
        for(int i = 0; i < u_N; ++i) {
          float diff = texelFetch(u_X, ivec2(i, c.y), 0).r - mean;
          var += diff * diff;
        }
        var /= float(u_N);
        float invStd = inversesqrt(var + 1e-5);

        // 3) fetch the “current” element and its gamma/beta
        float x = texelFetch(u_X, ivec2(c.x, c.y), 0).r;
        float g = texelFetch(u_gamma, ivec2(c.x, 0), 0).r;
        float b = texelFetch(u_beta,  ivec2(c.x, 0), 0).r;

        // 4) normalize, scale, shift
        float y = (x - mean) * invStd * g + b;
        o = vec4(y, 0, 0, 1);
      }`;

    // attnScore
    const attnScore = `#version 300 es
      precision highp float;
      uniform sampler2D u_Q, u_K;
      uniform int u_D;
      in vec2 v_uv;
      out vec4 o;
      void main(){
        ivec2 c = ivec2(gl_FragCoord.xy);
        int i = c.y;
        int j = c.x;
        float sum = 0.0;
        for(int d = 0; d < u_D; ++d){
          sum += texelFetch(u_Q, ivec2(d, i), 0).r
               * texelFetch(u_K, ivec2(d, j), 0).r;
        }
        float score = sum / sqrt(float(u_D));
        // --- causal mask: forbid attending to future tokens ---
        if (j > i) {
          score += -1e10;  
        }
        o = vec4(score, 0, 0, 1);
      }`;

    // softmax
    const softmax = `#version 300 es
      precision highp float;
      uniform sampler2D u_S;
      uniform int u_L;
      in vec2 v_uv; out vec4 o;
      void main(){
        ivec2 c = ivec2(gl_FragCoord.xy);
        int row = c.y;
        float m = -1e20;
        for(int i=0;i<u_L;++i){
          m = max(m, texelFetch(u_S, ivec2(i,row),0).r);
        }
        float sum=0.0;
        for(int i=0;i<u_L;++i){
          sum += exp(texelFetch(u_S, ivec2(i,row),0).r - m);
        }
        float val = exp(texelFetch(u_S, c,0).r - m) / sum;
        o = vec4(val,0,0,1);
      }`;

    // reuse matmul for matMul2
    const matmul2 = matmul;

    // residual sum 
    const add = `#version 300 es
      precision highp float;
      uniform sampler2D u_A, u_B;
      in vec2 v_uv;
      out vec4 o;
      void main() {
        ivec2 c = ivec2(gl_FragCoord.xy);
        float a = texelFetch(u_A, c, 0).r;
        float b = texelFetch(u_B, c, 0).r;
        o = vec4(a + b, 0, 0, 1);
      }
    `;
    this.programs["add"] = this._createProgram(vsrc, add);

    // after your matmul / add / etc.
    // display‐vertex shader stays the same (vsrc)
    const lmHeadFS = `#version 300 es
      precision highp float;
      uniform int u_row;
      // hidden vector: [N × 1]
      uniform sampler2D u_X;
      // lm_head_w tiled: [M_w × H_w]
      uniform sampler2D u_W;
      // uniforms:
      uniform int u_N;    // hiddenSize
      uniform int u_M;    // weight‐tile width (M_w)
      uniform int u_O;    // output‐tile width (Wlg)
      uniform int u_V;    // vocabSize
      in vec2 v_uv;
      out vec4 o;
      void main() {
        int px = int(gl_FragCoord.x);
        int py = int(gl_FragCoord.y);
        int v  = py * u_O + px;
        if (v >= u_V) {
          o = vec4(0,0,0,1);
          return;
        }
        float sum = 0.0;
        for (int f = 0; f < u_N; ++f) {
          float x = texelFetch(u_X, ivec2(f, u_row), 0).r;
          // row-major flatten: idx = v*u_N + f
          int idx = v * u_N + f;
          int tx = idx % u_M;
          int ty = idx / u_M;
          float w = texelFetch(u_W, ivec2(tx,ty), 0).r;
          sum += x * w;
        }
        o = vec4(sum, 0, 0, 1);

      }`;
    this.programs["lmHead"] = this._createProgram(vsrc, lmHeadFS);

    // split head
    const sliceHeadFS = `#version 300 es
      precision highp float;
      uniform sampler2D u_X;
      uniform int u_headOffset;       // in [0, 64, 128, …, 704]
      in vec2 v_uv;
      out vec4 o;
      void main(){
        ivec2 c = ivec2(gl_FragCoord.xy);
        // c.x + u_headOffset samples the correct feature slice
        float v = texelFetch(u_X, ivec2(c.x + u_headOffset, c.y), 0).r;
        o = vec4(v, 0, 0, 1);
      }`;
    this.programs["sliceHead"] = this._createProgram(vsrc, sliceHeadFS);

    for (const [name, src] of [
      ["matMul", matmul],
      ["addBias", addBias],
      ["gelu", gelu],
      ["layerNorm", layerNorm],
      ["attnScore", attnScore],
      ["softmax", softmax],
      ["matMul2", matmul2],
    ] as [string, string][]) {
      this.programs[name] = this._createProgram(vsrc, src);
    }
  }

  private _createProgram(vsSrc: string, fsSrc: string): WebGLProgram {
    const gl = this.gl;
    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vsSrc);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(vs) || "VS compile error");
    }
    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, fsSrc);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(fs) || "FS compile error");
    }
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.bindAttribLocation(prog, 0, "a_position");
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(prog) || "Link error");
    }
    return prog;
  }

  /**
   * Create an empty R32F texture of logical size wRaw×hRaw.
   * If either dimension exceeds maxTex, it tiles into a smaller texture.
   */
  private _createEmptyTex(wRaw: number, hRaw: number): WebGLTexture {
    const gl = this.gl;
    const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;

    let W = wRaw;
    let H = hRaw;
    if (wRaw > maxTex || hRaw > maxTex) {
      const total = wRaw * hRaw;
      W = Math.min(total, maxTex);
      H = Math.ceil(total / W);
    }

    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.R32F,
      W,
      H,
      0,
      gl.RED,
      gl.FLOAT,
      null
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  private _drawTexture(tex: WebGLTexture) {
    const gl = this.gl;

    // clear the canvas
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // draw full-screen quad with display shader
    const prog = this.programs.display;
    gl.useProgram(prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    const pa = gl.getAttribLocation(prog, "a_position");
    gl.enableVertexAttribArray(pa);
    gl.vertexAttribPointer(pa, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    const loc = gl.getUniformLocation(prog, "u_tex");
    gl.uniform1i(loc, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  /**
   * Read back an R32F texture of size W×H into a Float32Array.
   */
  private _readTex(tex: WebGLTexture, W: number, H: number): Float32Array {
    const gl = this.gl;
    const fb = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

    const out = new Float32Array(W * H);
    gl.readPixels(0, 0, W, H, gl.RED, gl.FLOAT, out);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(fb);
    return out;
  }

  /**
   * Pretty-print a texture by showing a small “window”:
   * first 3 rows, …, last 3 rows; and in each row
   * first 3 columns, …, last 3 columns.
   */
  private debugPrint(stage: string, tex: WebGLTexture, W: number, H: number) {
    const data = this._readTex(tex, W, H);
    console.group(stage);
    console.log(`shape = [${H} × ${W}]`);
    
    const topRows = 3;
    const botRows = 3;
    const printAllRows = H <= topRows + botRows;

    const printRow = (r: number) => {
      const vals: string[] = [];
      if (W <= 6) {
        for (let c = 0; c < W; c++) {
          vals.push(data[r * W + c].toFixed(4));
        }
      } else {
        // first 3 cols
        for (let c = 0; c < 3; c++) {
          vals.push(data[r * W + c].toFixed(4));
        }
        vals.push("…");
        // last 3 cols
        for (let c = W - 3; c < W; c++) {
          vals.push(data[r * W + c].toFixed(4));
        }
      }
      console.log(vals.join(" "));
    };

    // print top rows
    for (let r = 0; r < (printAllRows ? H : topRows); r++) {
      printRow(r);
    }

    // if too many rows, elide middle
    if (!printAllRows) {
      console.log("…");
      for (let r = H - botRows; r < H; r++) {
        printRow(r);
      }
    }

    console.groupEnd();
  }

  private _runPass(
    name: string,
    inputs: { [u: string]: WebGLTexture },
    ints: { [u: string]: number },
    outTex: WebGLTexture,
    W: number,
    H: number
  ) {
    const gl = this.gl;
    const prog = this.programs[name];
    gl.useProgram(prog);

    // bind inputs
    let unit = 0;
    for (const uni in inputs) {
      const loc = gl.getUniformLocation(prog, uni);
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, inputs[uni]);
      gl.uniform1i(loc, unit);
      unit++;
    }
    for (const uni in ints) {
      const loc = gl.getUniformLocation(prog, uni);
      gl.uniform1i(loc, ints[uni]);
    }

    // bind output FBO
    const fb = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      outTex,
      0
    );

    // draw quad
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    const pa = gl.getAttribLocation(prog, "a_position");
    gl.enableVertexAttribArray(pa);
    gl.vertexAttribPointer(pa, 2, gl.FLOAT, false, 0, 0);

    gl.viewport(0, 0, W, H);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private async forward(inputIds: number[]): Promise<Float32Array> {
    const gl = this.gl;
    const L = inputIds.length;
    const nHeads = 12;
    const headDim = this.nEmbeds / nHeads; // 64

    // 1) Token + position embeddings → CPU array
    const emb = new Float32Array(L * this.nEmbeds);
    for (let i = 0; i < L; i++) {
      const id = inputIds[i];
      for (let d = 0; d < this.nEmbeds; d++) {
        emb[i * this.nEmbeds + d] =
          this.weightArrays["wte"][id * this.nEmbeds + d] +
          this.weightArrays["wpe"][i * this.nEmbeds + d];
      }
    }

    // 2) Upload embeddings as a texture (width=features, height=seq)
    const texX = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texX);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.R32F,
      this.nEmbeds, L, 0,     // ← swapped
      gl.RED, gl.FLOAT,
      emb
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    let cur = texX;

    for (let layer = 0; layer < this.nLayers; layer++) {
      if (layer === DEBUG_LAYER) {
        this.debugPrint(`Block ${layer} — input x`, cur, this.nEmbeds, L);
      }

      // 3) First LayerNorm → [features × seq]
      const norm1 = this._createEmptyTex(this.nEmbeds, L);
      this._runPass(
        "layerNorm",
        {
          u_X:      cur,
          u_gamma: this.textures[`ln_1_g_${layer}`],
          u_beta:  this.textures[`ln_1_b_${layer}`],
        },
        { u_N: this.nEmbeds },
        norm1,
        this.nEmbeds, L      // ← swapped
      );
      if (layer === DEBUG_LAYER) {
        this.debugPrint(`Block ${layer} — LayerNorm 1`, norm1, this.nEmbeds, L);
      }

      // 4) Q/K/V projections + bias → all [features × seq]
      const Q  = this._createEmptyTex(this.nEmbeds, L);
      const Qb = this._createEmptyTex(this.nEmbeds, L);
      this._runPass(
        "matMul",
        { u_A: norm1, u_B: this.textures[`c_attn_q_w_${layer}`] },
        { u_K: this.nEmbeds },
        Q,
        this.nEmbeds, L
      );

      this._runPass(
        "addBias",
        { u_X: Q, u_bias: this.textures[`c_attn_q_b_${layer}`] },
        {},
        Qb,
        this.nEmbeds, L
      );

      const K  = this._createEmptyTex(this.nEmbeds, L);
      const Kb = this._createEmptyTex(this.nEmbeds, L);
      this._runPass(
        "matMul",
        { u_A: norm1, u_B: this.textures[`c_attn_k_w_${layer}`] },
        { u_K: this.nEmbeds },
        K,
        this.nEmbeds, L
      );
      this._runPass(
        "addBias",
        { u_X: K, u_bias: this.textures[`c_attn_k_b_${layer}`] },
        {},
        Kb,
        this.nEmbeds, L
      );

      const V  = this._createEmptyTex(this.nEmbeds, L);
      const Vb = this._createEmptyTex(this.nEmbeds, L);
      this._runPass(
        "matMul",
        { u_A: norm1, u_B: this.textures[`c_attn_v_w_${layer}`] },
        { u_K: this.nEmbeds },
        V,
        this.nEmbeds, L
      );
      this._runPass(
        "addBias",
        { u_X: V, u_bias: this.textures[`c_attn_v_b_${layer}`] },
        {},
        Vb,
        this.nEmbeds, L
      );
      if (layer === DEBUG_LAYER) {
        this.debugPrint(`Block ${layer} — Q`, Qb, this.nEmbeds, L);
        this.debugPrint(`Block ${layer} — K`, Kb, this.nEmbeds, L);
        this.debugPrint(`Block ${layer} — V`, Vb, this.nEmbeds, L);
      }


      // 5) Multi-head attention: slice → attend → merge
      const Qheads: WebGLTexture[] = [];
      const Kheads: WebGLTexture[] = [];
      const Vheads: WebGLTexture[] = [];
      for (let h = 0; h < nHeads; ++h) {
        const off = h * headDim;
        const Qh = this._createEmptyTex(headDim, L);
        this._runPass("sliceHead", { u_X: Qb }, { u_headOffset: off }, Qh, headDim, L);
        Qheads.push(Qh);

        const Kh = this._createEmptyTex(headDim, L);
        this._runPass("sliceHead", { u_X: Kb }, { u_headOffset: off }, Kh, headDim, L);
        Kheads.push(Kh);

        const Vh = this._createEmptyTex(headDim, L);
        this._runPass("sliceHead", { u_X: Vb }, { u_headOffset: off }, Vh, headDim, L);
        Vheads.push(Vh);
      }

      const AhHeads: WebGLTexture[] = [];
      for (let h = 0; h < nHeads; ++h) {
        // attnScore + softmax → [seq×seq]
        const S = this._createEmptyTex(L, L);
        this._runPass("attnScore", { u_Q: Qheads[h], u_K: Kheads[h] }, { u_D: headDim }, S, L, L);

        const P = this._createEmptyTex(L, L);
        this._runPass("softmax", { u_S: S }, { u_L: L }, P, L, L);

        // P @ Vh → [headDim × seq]
        const Ah = this._createEmptyTex(headDim, L);
        this._runPass("matMul", { u_A: P, u_B: Vheads[h] }, { u_K: L }, Ah, headDim, L);

        if (layer === DEBUG_LAYER && h === 0) {
          this.debugPrint(`Block ${layer} — head 0 scores`, S, L, L);
          this.debugPrint(`Block ${layer} — head 0 probs`, P, L, L);
          this.debugPrint(`Block ${layer} — head 0 output`, Ah, headDim, L);
        }

        AhHeads.push(Ah);
      }

      // merge heads → [features × seq]
      const A_merged = this._createEmptyTex(this.nEmbeds, L);
      const fb = gl.createFramebuffer()!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, A_merged, 0);

      gl.useProgram(this.programs.display);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
      const pa = gl.getAttribLocation(this.programs.display, "a_position");
      gl.enableVertexAttribArray(pa);
      gl.vertexAttribPointer(pa, 2, gl.FLOAT, false, 0, 0);
      const loc = gl.getUniformLocation(this.programs.display, "u_tex");

      for (let h = 0; h < nHeads; ++h) {
        gl.viewport(h * headDim, 0, headDim, L);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, AhHeads[h]);
        gl.uniform1i(loc, 0);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      if (layer === DEBUG_LAYER) {
        this.debugPrint(`Block ${layer} — merged heads`, A_merged, this.nEmbeds, L);
      }

      // 6) Project & bias → [features × seq]
      const AP  = this._createEmptyTex(this.nEmbeds, L);
      const APb = this._createEmptyTex(this.nEmbeds, L);
      this._runPass("matMul", { u_A: A_merged, u_B: this.textures[`c_proj_w_${layer}`] }, { u_K: this.nEmbeds }, AP,  this.nEmbeds, L);
      this._runPass("addBias", { u_X: AP, u_bias: this.textures[`c_proj_b_${layer}`] }, {},          APb, this.nEmbeds, L);
      if (layer === DEBUG_LAYER) {
        this.debugPrint(`Block ${layer} — projection output`, APb, this.nEmbeds, L);
      }

      // 7) Residual 1: cur = cur + APb
      const res1 = this._createEmptyTex(this.nEmbeds, L);
      this._runPass("add", { u_A: cur, u_B: APb }, {}, res1, this.nEmbeds, L);
      if (layer === DEBUG_LAYER) {
        this.debugPrint(`Block ${layer} — post-attention + residual`, res1, this.nEmbeds, L);
      }

      // 8) Second LayerNorm → [features × seq]
      const norm2 = this._createEmptyTex(this.nEmbeds, L);
      this._runPass(
        "layerNorm",
        { u_X: res1, u_gamma: this.textures[`ln_2_g_${layer}`], u_beta: this.textures[`ln_2_b_${layer}`] },
        { u_N: this.nEmbeds },
        norm2,
        this.nEmbeds, L
      );
      if (layer === DEBUG_LAYER) {
        this.debugPrint(`Block ${layer} — LayerNorm 2`, norm2, this.nEmbeds, L);
      }

      // 9) FFN: up → GELU → down → [features × seq]
      const fc1  = this._createEmptyTex(4 * this.nEmbeds, L);
      const fc1b = this._createEmptyTex(4 * this.nEmbeds, L);
      this._runPass("matMul", { u_A: norm2, u_B: this.textures[`mlp_fc_w_${layer}`] }, { u_K: this.nEmbeds }, fc1,  4 * this.nEmbeds, L);
      this._runPass("addBias", { u_X: fc1, u_bias: this.textures[`mlp_fc_b_${layer}`] }, {},                 fc1b, 4 * this.nEmbeds, L);

      const geluOut = this._createEmptyTex(4 * this.nEmbeds, L);
      this._runPass("gelu", { u_X: fc1b }, {}, geluOut, 4 * this.nEmbeds, L);
      if (layer === DEBUG_LAYER) {
        this.debugPrint(`Block ${layer} — FFN intermediate`, geluOut, 4 * this.nEmbeds, L);
      }

      const fc2  = this._createEmptyTex(this.nEmbeds, L);
      const fc2b = this._createEmptyTex(this.nEmbeds, L);
      this._runPass("matMul", { u_A: geluOut, u_B: this.textures[`mlp_proj_w_${layer}`] }, { u_K: 4 * this.nEmbeds }, fc2,  this.nEmbeds, L);
      this._runPass("addBias", { u_X: fc2, u_bias: this.textures[`mlp_proj_b_${layer}`] }, {},             fc2b, this.nEmbeds, L);
      if (layer === DEBUG_LAYER) {
        this.debugPrint(`Block ${layer} — FFN output`, fc2b, this.nEmbeds, L);
      }

      // 10) Residual 2: cur = res1 + fc2b
      const res2 = this._createEmptyTex(this.nEmbeds, L);
      this._runPass("add", { u_A: res1, u_B: fc2b }, {}, res2, this.nEmbeds, L);
      if (layer === DEBUG_LAYER) {
        this.debugPrint(`Block ${layer} — Block output`, res2, this.nEmbeds, L);
      }

      cur = res2;
    }
    this.debugPrint(`All layers output`, cur, this.nEmbeds, L);

    // Final LayerNorm → [features × seq]
    const normF = this._createEmptyTex(this.nEmbeds, L);
    this._runPass(
      "layerNorm",
      { u_X: cur, u_gamma: this.textures["ln_f_g"], u_beta: this.textures["ln_f_b"] },
      { u_N: this.nEmbeds },
      normF,
      this.nEmbeds, L
    );
    this.debugPrint(`Final layer norm output`, normF, this.nEmbeds, L);

    // compute output‐tile dims
    const V   = this.vocabSize;
    const max = this.gl.getParameter(this.gl.MAX_TEXTURE_SIZE) as number;
    const Wlg = Math.min(V, max);
    const Hlg = Math.ceil(V / Wlg);

    // allocate the logits texture (still using your existing createEmptyTex)
    const lg = this._createEmptyTex(V, 1);

    // run the new shader:
    const row = L - 1;
    this._runPass(
      "lmHead",
      { u_X: normF,
        u_W: this.textures["lm_head_w"] },
      { u_N: this.nEmbeds,
        u_M: max,       // weight‐tile width
        u_O: Wlg,       // output‐tile width
        u_V: V,          // vocabSize
        u_row: row               // ← pass the last-row index here
      },
      lg,
      Wlg, Hlg
    );
    this.debugPrint(`LM-head output`, lg, Wlg, Hlg);

    const raw = this._readTex(lg, Wlg, Hlg);
    const logits = new Float32Array(V);
    for (let v = 0; v < V; v++) {
      const x = v % Wlg;
      const y = Math.floor(v / Wlg);
      logits[v] = raw[y * Wlg + x];
    }

    // draw & read back
    this._drawTexture(lg);

    return logits;
  }


  private softmax(x: Float32Array) {
    let m = -Infinity;
    for (const v of x) m = Math.max(m, v);
    let sum = 0;
    const y = new Float32Array(x.length);
    for (let i = 0; i < y.length; i++) {
      y[i] = Math.exp(x[i] - m);
      sum += y[i];
    }
    for (let i = 0; i < y.length; i++) y[i] /= sum;
    return y;
  }

  private sample(probs: Float32Array) {
    // max
    let m = 0,
      idx = 0;
    for (let i = 0; i < probs.length; i++) {
      if (probs[i] > m) {
        m = probs[i];
        idx = i;
      }
    }
    return idx;

    // // stochastic
    // const r = Math.random();
    // let cum = 0;
    // for (let i = 0; i < probs.length; i++) {
    //   cum += probs[i];
    //   if (r < cum) return i;
    // }
    // return probs.length - 1;
  }

  async generate(
    prompt: string,
    onToken: (t: string) => void,
    shouldStop: () => boolean
  ) {
    let ids = Array.from(this.tokenizer.encode(prompt));

    // allow UI to repaint before token loop
    await new Promise<void>((resolve) =>
      requestAnimationFrame(() => resolve())
    );

    while (!shouldStop()) {
      const logits = await this.forward(ids);
      const probs = this.softmax(logits);
      const nxt = this.sample(probs);
      if (nxt === 50256) break;
      const tok = this.tokenizer.decode([nxt]);
      onToken(tok);
      ids.push(nxt);
      // yield for UI responsiveness
      await new Promise<void>((resolve) =>
        requestAnimationFrame(() => resolve())
      );
    }
  }
}
