/**
 * All requests to the VM backend are routed through the Replit API server proxy
 * at /api/proxy/... to avoid mixed-content blocking (HTTPS frontend → HTTP backend).
 *
 * Usage:
 *   vmFetch('/health', settings)              → GET
 *   vmFetch('/chat', settings, { body, ... }) → POST with streaming
 */

import { Settings } from './types';

export function vmProxyUrl(path: string): string {
  // path must start with /
  const normalised = path.startsWith('/') ? path : `/${path}`;
  return `/api/proxy${normalised}`;
}

export function vmHeaders(settings: Settings): Record<string, string> {
  const h: Record<string, string> = {
    'X-VM-Url': settings.vmBackendUrl,
    'Content-Type': 'application/json',
  };
  if (settings.vmBearerToken) {
    h['Authorization'] = `Bearer ${settings.vmBearerToken}`;
  }
  return h;
}

export async function vmFetch(
  path: string,
  settings: Settings,
  init?: RequestInit,
): Promise<Response> {
  return fetch(vmProxyUrl(path), {
    ...init,
    headers: {
      ...vmHeaders(settings),
      ...(init?.headers ?? {}),
    },
  });
}
