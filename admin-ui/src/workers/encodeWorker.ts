self.onmessage = async (e) => {
  try {
    const { buffer } = e.data;
    // Convert ArrayBuffer to base64 in chunks to avoid stack limits
    const uint8 = new Uint8Array(buffer);
    const chunkSize = 0x8000;
    let binary = '';
    for (let i = 0; i < uint8.length; i += chunkSize) {
      const slice = uint8.subarray(i, i + chunkSize);
      binary += String.fromCharCode.apply(null, Array.from(slice));
    }
    const b64 = btoa(binary);
    self.postMessage({ base64: b64 });
  } catch (err) {
    self.postMessage({ error: String(err) });
  }
};
