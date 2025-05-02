import { GPT2WebGL } from "./gpt2_webgl";

const PRINT_OUTPUT = true;

(async () => {
  const canvas = document.getElementById("glcanvas") as HTMLCanvasElement;
  const input  = document.getElementById("inputText") as HTMLTextAreaElement;
  const startB = document.getElementById("startBtn") as HTMLButtonElement;
  const stopB  = document.getElementById("stopBtn") as HTMLButtonElement;

  const raw = await fetch("weights/manifest.json").then(r => r.json());
  const manifest: Record<string,string> = {};
  for (const [name, path] of Object.entries(raw)) {
    manifest[name] = `weights/${path}`;
  }

  const model = new GPT2WebGL(canvas, manifest);
  await model.loadWeights();

  const layerContainer = document.getElementById("layerCanvases")!;
  const gridSize  = 28;
  const blockSize = 4;
  for (let i = 0; i < model.nLayers; i++) {
    const c = document.createElement("canvas");
    c.id     = `layerCanvas${i}`;
    c.width  = gridSize * blockSize;  // 112
    c.height = gridSize * blockSize;  // 112
    c.style.border = "1px solid #ccc";
    layerContainer.appendChild(c);
    model.registerLayerCanvas(i, c);
  }
  const firstAttentionContainer = document.getElementById("firstAttentionCanvases")!;
  for (let i = 0; i < model.nHeads; i++) {
    const c = document.createElement("canvas");
    c.id     = `attnHead${i}`;
    c.width  = gridSize * blockSize;  // 112
    c.height = gridSize * blockSize;  // 112
    c.style.border = "1px solid #ccc";
    firstAttentionContainer.appendChild(c);
    model.registerFirstAttentionCanvas(i, c);
  }
  const lastAttentionContainer = document.getElementById("lastAttentionCanvases")!;
  for (let i = 0; i < model.nHeads; i++) {
    const c = document.createElement("canvas");
    c.id     = `attnHead${i}`;
    c.width  = gridSize * blockSize;  // 112
    c.height = gridSize * blockSize;  // 112
    c.style.border = "1px solid #ccc";
    lastAttentionContainer.appendChild(c);
    model.registerLastAttentionCanvas(i, c);
  }


  let stopping = false;
  startB.onclick = () => {
    console.log("CLICKED START")
    stopping = false;
    startB.disabled = true;
    stopB.disabled  = false;
    model.generate(
      input.value,
      tok => {
        input.value += tok;
        if (PRINT_OUTPUT) {
            console.log(input.value);
        }
      },
      () => stopping
    ).then(() => {
      startB.disabled = false;
      stopB.disabled  = true;
    });
  };
  stopB.onclick = () => {
    console.log("CLICKED STOP")
    stopping = true;
    stopB.disabled = true;
    startB.disabled  = false;
  }
})();
