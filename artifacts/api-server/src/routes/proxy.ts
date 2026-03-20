import { Router, Request, Response } from "express";

const router = Router();

async function proxyHandler(req: Request, res: Response) {
  const vmUrl = req.headers["x-vm-url"] as string | undefined;
  if (!vmUrl) {
    res.status(400).json({ error: "X-VM-Url header is required" });
    return;
  }

  // req.url is the path relative to where this router is mounted, e.g. "/health"
  const targetUrl = `${vmUrl.replace(/\/$/, "")}${req.url}`;

  const forwardHeaders: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (req.headers.authorization) {
    forwardHeaders["Authorization"] = req.headers.authorization;
  }

  const init: RequestInit = { method: req.method, headers: forwardHeaders };

  if (req.method !== "GET" && req.method !== "HEAD" && req.body) {
    init.body = JSON.stringify(req.body);
  }

  try {
    const vmRes = await fetch(targetUrl, init);

    res.status(vmRes.status);
    // Disable buffering so NDJSON streaming works end-to-end
    res.setHeader("Cache-Control", "no-cache, no-store");
    res.setHeader("X-Accel-Buffering", "no");

    const ct = vmRes.headers.get("content-type");
    if (ct) res.setHeader("Content-Type", ct);

    if (!vmRes.body) {
      res.send(await vmRes.text());
      return;
    }

    // Pipe the stream chunk-by-chunk so NDJSON lines reach the browser live
    const reader = vmRes.body.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      res.write(value);
    }
    res.end();
  } catch (err) {
    const message = err instanceof Error ? err.message : "Proxy error";
    if (!res.headersSent) {
      res.status(502).json({ error: message });
    } else {
      res.end();
    }
  }
}

// Mount as middleware so all sub-paths (/health, /tools, /chat, /api/healthz, …) are handled
router.use("/", proxyHandler);

export default router;
