export async function loadHRTFData(): Promise<Float32Array> {
  const response = await fetch('/src/hrtf/RIEC_hrir_subject_080.sofa');
  const arrayBuffer = await response.arrayBuffer();
  return new Float32Array(arrayBuffer);
}