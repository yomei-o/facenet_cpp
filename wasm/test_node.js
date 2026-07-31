const createFaceNet = require('./facenet.js');
const fs = require('fs');
function f32(p){ const b=fs.readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.length/4); }
createFaceNet().then(M => {
  M.FS.mkdir('/facenet');
  M.FS.writeFile('/facenet/manifest.txt', fs.readFileSync('../weights/facenet/manifest.txt'));
  M.FS.writeFile('/facenet/weights_fp16.bin', fs.readFileSync('../weights/facenet/weights_fp16.bin'));
  M._fn_ready();
  const chw=f32('../pure/ref/input.bin'), ref=f32('../pure/ref/embed.bin');
  const ptr=M._malloc(chw.length*4); M.HEAPF32.set(chw, ptr>>2);
  const out=M._fn_embed_chw(ptr);
  let worst=0,l2=0; for(let i=0;i<512;i++){const v=M.HEAPF32[(out>>2)+i]; worst=Math.max(worst,Math.abs(v-ref[i])); l2+=v*v;}
  M._free(ptr);
  console.log('WASM(fetch weights) vs ref: L2norm='+Math.sqrt(l2).toFixed(6)+' worst='+worst.toExponential(3)+' '+(worst<5e-3?'OK':'HIGH'));
});
