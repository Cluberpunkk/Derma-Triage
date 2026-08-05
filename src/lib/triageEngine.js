// triageEngine.js
// Calls the vision model THROUGH your backend proxy (server/proxy.js).
//
// Never call api.anthropic.com directly from the browser: it fails CORS and
// would expose your API key. The proxy holds the key server-side.
//
// When you have a real (fine-tuned) model, you usually only change proxy.js —
// this file and the UI stay the same.

export async function runTriage({ imageB64, details }) {
  const res = await fetch("/api/triage", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ imageB64, details }),
  });
  if (!res.ok) throw new Error("Triage request failed");

  const { text } = await res.json();

  // The model is instructed to return pure JSON, but parse defensively.
  let t = (text || "").replace(/```json/gi, "").replace(/```/g, "").trim();
  const start = t.indexOf("{");
  const end = t.lastIndexOf("}");
  if (start !== -1 && end !== -1) t = t.slice(start, end + 1);

  const parsed = JSON.parse(t);
  if (!["routine", "soon", "urgent"].includes(parsed.urgency)) parsed.urgency = "soon";
  return parsed;
}
