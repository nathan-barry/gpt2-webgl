import { GPT2WebGL } from "./gpt2_webgl";

const PRINT_OUTPUT = true;

(async () => {
  const canvas = document.getElementById("glcanvas") as HTMLCanvasElement;
  const input  = document.getElementById("inputText") as HTMLTextAreaElement;
  const startB = document.getElementById("startBtn") as HTMLButtonElement;
  const stopB  = document.getElementById("stopBtn") as HTMLButtonElement;

  const manifest: Record<string,string> =
    await fetch("/weights/manifest.json").then(r => {
      if (!r.ok) throw new Error(`Manifest load ${r.status}`);
      return r.json();
    });

  const model = new GPT2WebGL(canvas, manifest);
  await model.loadWeights();

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
