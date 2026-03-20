import { useCallback, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { useAppStore } from '@/lib/store';
import { AgentEvent, TraceStep } from '@/lib/types';
import { vmProxyUrl, vmHeaders } from '@/lib/vm-fetch';
import { useToast } from '@/hooks/use-toast';

/**
 * Parse a raw NDJSON stream line into a typed AgentEvent.
 * Returns null for blank lines, [DONE] sentinels, or unknown payloads.
 */
function parseStreamLine(line: string): AgentEvent | null {
  const trimmed = line.trim();
  if (!trimmed || trimmed === '[DONE]') return null;

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    console.warn('[agent] unparseable stream line:', line);
    return null;
  }

  if (parsed === null || typeof parsed !== 'object' || !('type' in parsed)) return null;

  const raw = parsed as Record<string, unknown>;
  const type = typeof raw.type === 'string' ? raw.type : null;
  if (!type) return null;

  const content = raw.content;

  switch (type) {
    case 'thought':
      return { type: 'thought', content: String(content ?? '') };

    case 'action': {
      const c = (content && typeof content === 'object') ? content as Record<string, unknown> : {};
      return {
        type: 'action',
        content: {
          tool: String(c.tool ?? ''),
          is_destructive: Boolean(c.is_destructive),
        },
      };
    }

    case 'observation':
      return {
        type: 'observation',
        content: typeof content === 'string' ? content : JSON.stringify(content ?? ''),
      };

    case 'confirmation_required': {
      const c = (content && typeof content === 'object') ? content as Record<string, unknown> : {};
      return {
        type: 'confirmation_required',
        content: {
          message: String(c.message ?? ''),
          session_id: String(c.session_id ?? ''),
          tool: String(c.tool ?? ''),
        },
      };
    }

    case 'result': {
      const c = (content && typeof content === 'object') ? content as Record<string, unknown> : {};
      const status = c.status;
      const validStatus = status === 'SUCCESS' || status === 'ERROR' || status === 'CANCELLED'
        ? status
        : 'ERROR';
      return {
        type: 'result',
        content: {
          status: validStatus,
          message: String(c.message ?? ''),
          session_id: c.session_id ? String(c.session_id) : undefined,
        },
      };
    }

    default:
      console.warn('[agent] unknown event type:', type);
      return null;
  }
}

export function useAgent() {
  const { state, dispatch, activeSession } = useAppStore();
  const { toast } = useToast();
  const abortControllerRef = useRef<AbortController | null>(null);
  const agentMsgIdRef = useRef<string>(uuidv4());

  /**
   * Module-level-style ref for the VM session ID.
   * Using a ref (not state) means it is written and read synchronously —
   * no async React re-render cycle, so the value is always current when
   * the next sendQuery fires.
   */
  const vmSessionIdRef = useRef<string | null>(null);

  // Reset the session ID whenever the user switches to a different session.
  const activeSessionId = activeSession?.id;
  const prevSessionIdRef = useRef<string | undefined>(undefined);
  useEffect(() => {
    if (activeSessionId !== prevSessionIdRef.current) {
      vmSessionIdRef.current = null;
      prevSessionIdRef.current = activeSessionId;
      console.log('[agent] session changed — session_id cleared');
    }
  }, [activeSessionId]);

  const processEvent = useCallback((event: AgentEvent) => {
    const agentMsgId = agentMsgIdRef.current;
    const base: Omit<TraceStep, 'type'> = { id: uuidv4(), timestamp: Date.now() };

    switch (event.type) {
      case 'thought':
        dispatch({ type: 'ENSURE_AGENT_MESSAGE', agentMsgId });
        dispatch({ type: 'ADD_TRACE', trace: { ...base, type: 'thought', content: event.content }, agentMsgId });
        break;

      case 'action':
        dispatch({ type: 'ENSURE_AGENT_MESSAGE', agentMsgId });
        dispatch({
          type: 'ADD_TRACE',
          trace: { ...base, type: 'action', tool: event.content.tool, is_destructive: event.content.is_destructive },
          agentMsgId,
        });
        break;

      case 'observation':
        dispatch({ type: 'ADD_TRACE', trace: { ...base, type: 'observation', content: event.content }, agentMsgId });
        break;

      case 'confirmation_required':
        // ★ Store synchronously in ref — available immediately on next call
        vmSessionIdRef.current = event.content.session_id;
        console.log('[agent] session_id captured from confirmation_required:', event.content.session_id);
        dispatch({
          type: 'ADD_TRACE',
          trace: {
            ...base,
            type: 'confirmation_required',
            message: event.content.message,
            session_id: event.content.session_id,
            tool: event.content.tool,
          },
          agentMsgId,
        });
        dispatch({ type: 'SET_VM_SESSION_ID', sessionId: event.content.session_id });
        dispatch({
          type: 'SET_CONFIRMATION',
          request: {
            message: event.content.message,
            session_id: event.content.session_id,
            tool: event.content.tool,
          },
        });
        dispatch({ type: 'SET_STREAMING', isStreaming: false });
        break;

      case 'result': {
        const { status, message, session_id } = event.content;
        // ★ Store synchronously in ref — available immediately on next call
        if (session_id) {
          vmSessionIdRef.current = session_id;
          console.log('[agent] session_id captured from result:', session_id);
          dispatch({ type: 'SET_VM_SESSION_ID', sessionId: session_id });
        }
        dispatch({ type: 'ENSURE_AGENT_MESSAGE', agentMsgId });
        dispatch({ type: 'ADD_TRACE', trace: { ...base, type: 'result', status, content: message }, agentMsgId });
        dispatch({ type: 'SET_STREAMING', isStreaming: false });
        if (status === 'ERROR') {
          toast({ title: 'Agent Error', description: message, variant: 'destructive' });
        }
        break;
      }
    }
  }, [dispatch, toast]);

  /** Drain an NDJSON response body, dispatching typed events for each line. */
  const drainStream = useCallback(async (body: ReadableStream<Uint8Array>) => {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';
      for (const line of lines) {
        const ev = parseStreamLine(line);
        if (ev) processEvent(ev);
      }
    }
    if (buffer.trim()) {
      const ev = parseStreamLine(buffer);
      if (ev) processEvent(ev);
    }
  }, [processEvent]);

  const sendQuery = useCallback(async (query: string) => {
    if (!query.trim() || state.isStreaming || !activeSession) return;

    agentMsgIdRef.current = uuidv4();
    dispatch({ type: 'ADD_MESSAGE', message: { id: uuidv4(), role: 'user', content: query, timestamp: Date.now() } });
    dispatch({ type: 'SET_STREAMING', isStreaming: true });

    // ── DEMO MODE ────────────────────────────────────────────────────────────
    if (!state.settings.vmBackendUrl) {
      const fire = (ev: AgentEvent, ms: number) => setTimeout(() => processEvent(ev), ms);
      fire({ type: 'thought', content: 'I need to find the Marketing project GID first, then create the task.' }, 500);
      fire({ type: 'action', content: { tool: 'search_projects', is_destructive: false } }, 1500);
      fire({ type: 'observation', content: 'Found project: Marketing (GID: 1234567890)' }, 3000);
      fire({ type: 'thought', content: "Now I'll create the task in this project." }, 4000);
      fire({ type: 'action', content: { tool: 'create_task', is_destructive: false } }, 5000);

      if (query.toLowerCase().includes('delete')) {
        const demoSid = 'demo-session-' + uuidv4();
        setTimeout(() => processEvent({
          type: 'confirmation_required',
          content: { message: 'I\'m about to permanently delete the task "Design Review" from the Marketing project. This cannot be undone. Proceed?', session_id: demoSid, tool: 'delete_task' },
        }), 6000);
        return;
      }

      fire({ type: 'observation', content: 'Task created successfully: GID 9876543210' }, 6500);
      setTimeout(() => {
        processEvent({ type: 'result', content: { status: 'SUCCESS', message: "I've created the task 'Design Review' in the Marketing project. Task GID: 9876543210" } });
      }, 7500);
      return;
    }

    // ── REAL BACKEND (via proxy) ──────────────────────────────────────────────
    try {
      abortControllerRef.current = new AbortController();

      // ★ Read from ref — always synchronously current, never stale
      const payload: Record<string, unknown> = { message: query };
      if (vmSessionIdRef.current) {
        payload.session_id = vmSessionIdRef.current;
        console.log('[agent] sending session_id:', vmSessionIdRef.current);
      }

      const res = await fetch(vmProxyUrl('/chat'), {
        method: 'POST',
        headers: vmHeaders(state.settings),
        body: JSON.stringify(payload),
        signal: abortControllerRef.current.signal,
      });

      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      if (!res.body) throw new Error('No response body');

      await drainStream(res.body);

      if (!state.pendingConfirmation) {
        dispatch({ type: 'SET_STREAMING', isStreaming: false });
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') return;
      const message = err instanceof Error ? err.message : 'Failed to communicate with VM';
      dispatch({ type: 'SET_STREAMING', isStreaming: false });
      toast({ title: 'Connection Error', description: message, variant: 'destructive' });
      processEvent({ type: 'result', content: { status: 'ERROR', message } });
    }
  }, [state.settings, state.isStreaming, state.pendingConfirmation, activeSession, dispatch, processEvent, drainStream, toast]);

  const confirmAction = useCallback(async (confirmed: boolean) => {
    // ★ Read from ref — always has the latest session_id
    const sessionId = state.pendingConfirmation?.session_id ?? vmSessionIdRef.current;
    dispatch({ type: 'SET_CONFIRMATION', request: null });

    // ── DEMO MODE ────────────────────────────────────────────────────────────
    if (!state.settings.vmBackendUrl) {
      if (confirmed) {
        dispatch({ type: 'SET_STREAMING', isStreaming: true });
        setTimeout(() => processEvent({ type: 'observation', content: 'Task deleted successfully.' }), 1000);
        setTimeout(() => processEvent({ type: 'result', content: { status: 'SUCCESS', message: 'The task has been permanently deleted.' } }), 2000);
      } else {
        processEvent({ type: 'result', content: { status: 'CANCELLED', message: 'Action cancelled by user.' } });
      }
      return;
    }

    // ── REAL BACKEND (via proxy) ──────────────────────────────────────────────
    try {
      dispatch({ type: 'SET_STREAMING', isStreaming: true });

      const payload: Record<string, unknown> = { message: '', confirmation: confirmed };
      if (sessionId) payload.session_id = sessionId;

      const res = await fetch(vmProxyUrl('/chat'), {
        method: 'POST',
        headers: vmHeaders(state.settings),
        body: JSON.stringify(payload),
      });

      if (res.body) await drainStream(res.body);
      dispatch({ type: 'SET_STREAMING', isStreaming: false });
    } catch (err) {
      dispatch({ type: 'SET_STREAMING', isStreaming: false });
      const message = err instanceof Error ? err.message : 'Failed to send confirmation';
      toast({ title: 'Error', description: message, variant: 'destructive' });
    }
  }, [state.settings, state.pendingConfirmation, dispatch, processEvent, drainStream, toast]);

  const stopStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      dispatch({ type: 'SET_STREAMING', isStreaming: false });
    }
  }, [dispatch]);

  return { sendQuery, confirmAction, stopStream };
}
