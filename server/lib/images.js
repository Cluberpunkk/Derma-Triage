// server/lib/images.js
// =========================================================================
// EXIF / metadata stripping for stored images. Applied to EVERY image at
// ingestion — a guarantee on top of the fact that both client capture paths
// already re-encode through a canvas (which drops metadata).
//
// Pure JS: walks the JPEG marker segments and removes APP1..APPF (EXIF, XMP,
// GPS, ICC-in-APP2 is dropped too — acceptable for a training corpus) and COM
// comment segments, keeping APP0/JFIF and all image data. No native dependency.
//
// Face blur is done CLIENT-SIDE before upload (the raw face never reaches the
// server); the client sends a face_blurred flag we simply record.
// =========================================================================

// Decode a base64 (optionally data-URL) string to a Buffer.
export function b64ToBuffer(b64) {
  const comma = b64.indexOf(",");
  const raw = b64.startsWith("data:") && comma !== -1 ? b64.slice(comma + 1) : b64;
  return Buffer.from(raw, "base64");
}

// Remove metadata segments from a JPEG buffer. If the buffer isn't a JPEG we
// return it unchanged (the caller only ever feeds us canvas-encoded JPEGs).
export function stripExif(buf) {
  if (buf.length < 4 || buf[0] !== 0xff || buf[1] !== 0xd8) return buf;

  const out = [buf.subarray(0, 2)]; // SOI
  let i = 2;
  while (i + 1 < buf.length) {
    if (buf[i] !== 0xff) { out.push(buf.subarray(i)); break; }
    const marker = buf[i + 1];

    // Start of scan: everything after is entropy-coded image data — copy as-is.
    if (marker === 0xda) { out.push(buf.subarray(i)); break; }
    // Standalone markers (RSTn, EOI, TEM) have no length payload.
    if (marker === 0xd9 || (marker >= 0xd0 && marker <= 0xd7) || marker === 0x01) {
      out.push(buf.subarray(i, i + 2));
      i += 2;
      continue;
    }

    const len = (buf[i + 2] << 8) | buf[i + 3]; // segment length incl. these 2 bytes
    const segEnd = i + 2 + len;
    const isApp1toF = marker >= 0xe1 && marker <= 0xef; // APP1..APP15
    const isCom = marker === 0xfe;
    if (!isApp1toF && !isCom) out.push(buf.subarray(i, segEnd)); // keep everything else
    i = segEnd;
  }
  return Buffer.concat(out);
}
