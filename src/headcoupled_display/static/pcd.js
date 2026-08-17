/**
 * Parse an ASCII PCD file's text content.
 *
 * @param {string} text - Raw PCD file content.
 * @returns {{
 *   positions: Float32Array,
 *   colors: Float32Array,
 *   pointCount: number,
 *   bounds: { min: number[], max: number[], center: number[] },
 * }} Parsed point cloud. `bounds` is the axis-aligned bounding box (AABB) of
 *   `positions`, computed in the same pass as parsing. `bounds.center` is the
 *   AABB midpoint `(min + max) / 2`, NOT the point centroid.
 * @throws {Error} If the PCD is malformed (see individual checks below), or if
 *   it declares zero points. Zero points would make `min`/`max`/`center`
 *   undefined (Infinity/-Infinity or NaN); rather than returning those
 *   silently, this function throws so callers cannot mistake an empty cloud
 *   for a valid one.
 */
export function parseAsciiPcd(text) {
  const lines = text.replace(/\r/g, "").split("\n");
  const header = new Map();
  let dataIndex = -1;
  for (let index = 0; index < lines.length; index += 1) {
    const raw = lines[index].trim();
    if (!raw || raw.startsWith("#")) continue;
    const [key, ...rest] = raw.split(/\s+/);
    header.set(key.toUpperCase(), rest);
    if (key.toUpperCase() === "DATA") {
      dataIndex = index + 1;
      break;
    }
  }
  if (dataIndex < 0) throw new Error("PCD DATA header is missing");
  const dataType = (header.get("DATA") || [""])[0].toLowerCase();
  if (dataType !== "ascii") {
    throw new Error(`Only ASCII PCD is supported by the offline renderer, got ${dataType}`);
  }
  const fields = header.get("FIELDS") || [];
  const xIndex = fields.indexOf("x");
  const yIndex = fields.indexOf("y");
  const zIndex = fields.indexOf("z");
  if (xIndex < 0 || yIndex < 0 || zIndex < 0) {
    throw new Error("PCD must contain x y z fields");
  }
  const rIndex = fields.indexOf("r");
  const gIndex = fields.indexOf("g");
  const bIndex = fields.indexOf("b");
  const rgbIndex = fields.indexOf("rgb");
  const requestedPoints = Number.parseInt((header.get("POINTS") || ["0"])[0], 10);
  const positions = [];
  const colors = [];
  const decoder = new ArrayBuffer(4);
  const floatView = new Float32Array(decoder);
  const intView = new Uint32Array(decoder);
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;

  for (let index = dataIndex; index < lines.length; index += 1) {
    const raw = lines[index].trim();
    if (!raw) continue;
    const values = raw.split(/\s+/);
    const x = Number(values[xIndex]);
    const y = Number(values[yIndex]);
    const z = Number(values[zIndex]);
    positions.push(x, y, z);
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
    if (rIndex >= 0 && gIndex >= 0 && bIndex >= 0) {
      colors.push(
        Number(values[rIndex]) / 255,
        Number(values[gIndex]) / 255,
        Number(values[bIndex]) / 255,
      );
    } else if (rgbIndex >= 0) {
      const rawRgb = Number(values[rgbIndex]);
      if (Number.isInteger(rawRgb)) {
        intView[0] = rawRgb >>> 0;
      } else {
        floatView[0] = rawRgb;
      }
      const packed = intView[0];
      colors.push(((packed >> 16) & 255) / 255, ((packed >> 8) & 255) / 255, (packed & 255) / 255);
    } else {
      colors.push(0.80, 0.85, 0.92);
    }
  }
  const pointCount = positions.length / 3;
  if (requestedPoints > 0 && pointCount !== requestedPoints) {
    throw new Error(`PCD declared ${requestedPoints} points but parsed ${pointCount}`);
  }
  if (pointCount === 0) {
    throw new Error("PCD contains no points; cannot compute bounds");
  }
  const bounds = {
    min: [minX, minY, minZ],
    max: [maxX, maxY, maxZ],
    center: [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2],
  };
  return {
    positions: new Float32Array(positions),
    colors: new Float32Array(colors),
    pointCount,
    bounds,
  };
}

export async function loadAsciiPcd(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`Unable to load PCD: HTTP ${response.status}`);
  return parseAsciiPcd(await response.text());
}
