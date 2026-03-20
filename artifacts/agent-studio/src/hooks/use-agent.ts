import { useCallback, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { useAppStore } from '@/lib/store';
import { AgentEvent, TraceStep } from '@/lib/types';
import { useToast } from '@/hooks/use-toast';

/** Parse a raw stream line into a typed AgentEvent, or null if unparseable/ignored. */
function parseStreamLine(line: string): AgentEvent | null {
  const trimmed = line.trim();
  if (!trimmed) return null;

  const jsonStr = trimmed.startsWith('data: ') ? trimmed.slice(6) : trimmed;
  if (jsonStr === '[DONE]') return null;

  let parsed: unknown;
  try {
    parsed = JSON.parse(jsonStr);
  } catch {
    console.warn('[agent] unparseable stream line:', line);
    return null;
  }

  if (
    parsed === null ||
    typeof parsed !== 'object' ||
    !('type' in parsed) ||
    typeof (parsed as Record<string, unknown>).type !== 'string'
  ) {
    return null;
  }

  const raw = parsed as Record<string, unknown>;
  const type = raw.type as string;

  switch (type) {
    case 'thought':
      return { type: 'thought', content: String(raw.content ?? '') };
    case 'action':
      return {
        type: 'action',
        tool: String(raw.tool ?? ''),
        args: (raw.args && typeof raw.args === 'object' && !Array.isArray(raw.args))
          ? (raw.args as Record<string, unknown>)
          : {},
      };
    case 'observation':
      return { type: 'observation', content: String(raw.content ?? '') };
    case 'final_answer':
      return { type: 'final_answer', content: String(raw.content ?? '') };
    case 'requires_confirmation':
      return {
        type: 'requires_confirmation',
        action_type: String(raw.action_type ?? ''),
        resource: String(raw.resource ?? ''),
        workspace: String(raw.workspace ?? ''),
        consequence: String(raw.consequence ?? ''),
      };
    case 'error':
      return { type: 'error', message: String(raw.message ?? 'Unknown error') };
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

  const processEvent = useCallback((event: AgentEvent) => {
    const agentMsgId = agentMsgIdRef.current;
    const baseTrace: Omit<TraceStep, 'type'> = { id: uuidv4(), timestamp: Date.now() };

    switch (event.type) {
      case 'thought':
        dispatch({ type: 'ENSURE_AGENT_MESSAGE', agentMsgId });
        dispatch({ type: 'ADD_TRACE', trace: { ...baseTrace, type: 'thought', content: event.content }, agentMsgId });
        break;
      case 'action':
        dispatch({ type: 'ENSURE_AGENT_MESSAGE', agentMsgId });
        dispatch({ type: 'ADD_TRACE', trace: { ...baseTrace, type: 'action', tool: event.tool, args: event.args }, agentMsgId });
        break;
      case 'observation':
        dispatch({ type: 'ADD_TRACE', trace: { ...baseTrace, type: 'observation', content: event.content }, agentMsgId });
        break;
      case 'final_answer':
        dispatch({ type: 'ENSURE_AGENT_MESSAGE', agentMsgId });
        dispatch({ type: 'ADD_TRACE', trace: { ...baseTrace, type: 'final_answer', content: event.content }, agentMsgId });
        break;
      case 'requires_confirmation':
        dispatch({
          type: 'ADD_TRACE',
          trace: {
            ...baseTrace,
            type: 'requires_confirmation',
            action_type: event.action_type,
            resource: event.resource,
            workspace: event.workspace,
            consequence: event.consequence,
          },
          agentMsgId,
        });
        dispatch({
          type: 'SET_CONFIRMATION',
          request: {
            action_type: event.action_type,
            resource: event.resource,
            workspace: event.workspace,
            consequence: event.consequence,
          },
        });
        dispatch({ type: 'SET_STREAMING', isStreaming: false });
        break;
      case 'error':
        dispatch({ type: 'ADD_TRACE', trace: { ...baseTrace, type: 'error', message: event.message }, agentMsgId });
        dispatch({ type: 'SET_STREAMING', isStreaming: false });
        toast({ title: 'Error', description: event.message, variant: 'destructive' });
        break;
    }
  }, [dispatch, toast]);

  /** Read an NDJSON/SSE response body, emitting typed events via processEvent. */
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
        const event = parseStreamLine(line);
        if (event) processEvent(event);
      }
    }

    // Handle any remaining buffered content
    if (buffer.trim()) {
      const event = parseStreamLine(buffer);
      if (event) processEvent(event);
    }
  }, [processEvent]);

  const sendQuery = useCallback(async (query: string) => {
    if (!query.trim() || state.isStreaming || !activeSession) return;

    agentMsgIdRef.current = uuidv4();

    dispatch({
      type: 'ADD_MESSAGE',
      message: { id: uuidv4(), role: 'user', content: query, timestamp: Date.now() },
    });
    dispatch({ type: 'SET_STREAMING', isStreaming: true });

    // DEMO MODE
    if (!state.settings.vmBackendUrl) {
      const fire = (ev: AgentEvent, ms: number) => setTimeout(() => processEvent(ev), ms);
      fire({ type: 'thought', content: 'I need to find the Marketing project GID first, then create the task.' }, 500);
      fire({ type: 'action', tool: 'search_projects', args: { query: 'Marketing' } }, 1500);
      fire({ type: 'observation', content: 'Found project: Marketing (GID: 1234567890)' }, 3000);
      fire({ type: 'thought', content: "Now I'll create the task in this project." }, 4000);
      fire({ type: 'action', tool: 'create_task', args: { project_gid: '1234567890', name: 'Design Review', notes: '' } }, 5000);

      if (query.toLowerCase().includes('delete')) {
        fire({
          type: 'requires_confirmation',
          action_type: 'DELETE TASK',
          resource: 'Design Review',
          workspace: 'Marketing',
          consequence: 'This will permanently delete the task and it cannot be recovered.',
        }, 6000);
        return;
      }

      fire({ type: 'observation', content: 'Task created successfully: GID 9876543210' }, 6500);
      setTimeout(() => {
        processEvent({ type: 'final_answer', content: "I've created the task 'Design Review' in the Marketing project. Task GID: 9876543210" });
        dispatch({ type: 'SET_STREAMING', isStreaming: false });
      }, 7500);
      return;
    }

    // REAL FETCH STREAM
    try {
      abortControllerRef.current = new AbortController();
      const payload = {
        query,
        sessionId: activeSession.id,
        history: activeSession.messages.map(m => ({ role: m.role, content: m.content })),
      };

      const res = await fetch(`${state.settings.vmBackendUrl}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(state.settings.vmBearerToken ? { Authorization: `Bearer ${state.settings.vmBearerToken}` } : {}),
        },
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
      processEvent({ type: 'error', message });
    }
  }, [state.settings, state.isStreaming, state.pendingConfirmation, activeSession, dispatch, processEvent, drainStream, toast]);

  const confirmAction = useCallback(async (confirmed: boolean) => {
    dispatch({ type: 'SET_CONFIRMATION', request: null });

    if (!state.settings.vmBackendUrl) {
      if (confirmed) {
        dispatch({ type: 'SET_STREAMING', isStreaming: true });
        setTimeout(() => processEvent({ type: 'observation', content: 'Action executed successfully.' }), 1000);
        setTimeout(() => {
          processEvent({ type: 'final_answer', content: 'The requested action has been completed.' });
          dispatch({ type: 'SET_STREAMING', isStreaming: false });
        }, 2000);
      } else {
        processEvent({ type: 'final_answer', content: 'Action cancelled by user.' });
      }
      return;
    }

    try {
      dispatch({ type: 'SET_STREAMING', isStreaming: true });
      const res = await fetch(`${state.settings.vmBackendUrl}/confirm`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(state.settings.vmBearerToken ? { Authorization: `Bearer ${state.settings.vmBearerToken}` } : {}),
        },
        body: JSON.stringify({ sessionId: activeSession?.id, confirmed }),
      });

      if (res.body) {
        await drainStream(res.body);
      }
      dispatch({ type: 'SET_STREAMING', isStreaming: false });
    } catch (err) {
      dispatch({ type: 'SET_STREAMING', isStreaming: false });
      const message = err instanceof Error ? err.message : 'Failed to send confirmation';
      toast({ title: 'Error', description: message, variant: 'destructive' });
    }
  }, [state.settings, activeSession?.id, dispatch, processEvent, drainStream, toast]);

  const stopStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      dispatch({ type: 'SET_STREAMING', isStreaming: false });
    }
  }, [dispatch]);

  return { sendQuery, confirmAction, stopStream };
}
