import { useCallback, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { useAppStore } from '@/lib/store';
import { TraceStep } from '@/lib/types';
import { useToast } from '@/hooks/use-toast';

export function useAgent() {
  const { state, dispatch, activeSession } = useAppStore();
  const { toast } = useToast();
  const abortControllerRef = useRef<AbortController | null>(null);

  const processEvent = useCallback((event: any) => {
    if (!event || !event.type) return;

    const baseTrace: Omit<TraceStep, 'type'> = {
      id: uuidv4(),
      timestamp: Date.now(),
    };

    switch (event.type) {
      case 'thought':
        dispatch({ type: 'ADD_TRACE', trace: { ...baseTrace, type: 'thought', content: event.content } });
        break;
      case 'action':
        dispatch({ type: 'ADD_TRACE', trace: { ...baseTrace, type: 'action', tool: event.tool, args: event.args } });
        break;
      case 'observation':
        dispatch({ type: 'ADD_TRACE', trace: { ...baseTrace, type: 'observation', content: event.content } });
        break;
      case 'final_answer':
        dispatch({ type: 'ADD_TRACE', trace: { ...baseTrace, type: 'final_answer', content: event.content } });
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
            consequence: event.consequence
          } 
        });
        dispatch({
          type: 'SET_CONFIRMATION',
          request: {
            action_type: event.action_type,
            resource: event.resource,
            workspace: event.workspace,
            consequence: event.consequence
          }
        });
        // We stop streaming visually, wait for user
        dispatch({ type: 'SET_STREAMING', isStreaming: false });
        break;
      case 'error':
        dispatch({ type: 'ADD_TRACE', trace: { ...baseTrace, type: 'error', message: event.message } });
        dispatch({ type: 'SET_STREAMING', isStreaming: false });
        toast({ title: 'Error', description: event.message, variant: 'destructive' });
        break;
    }
  }, [dispatch, toast]);

  const sendQuery = useCallback(async (query: string) => {
    if (!query.trim() || state.isStreaming || !activeSession) return;

    // Add user message
    dispatch({
      type: 'ADD_MESSAGE',
      message: { id: uuidv4(), role: 'user', content: query, timestamp: Date.now() }
    });

    dispatch({ type: 'SET_STREAMING', isStreaming: true });

    // DEMO MODE
    if (!state.settings.vmBackendUrl) {
      setTimeout(() => processEvent({ type: 'thought', content: 'I need to find the Marketing project GID first, then create the task.' }), 500);
      setTimeout(() => processEvent({ type: 'action', tool: 'search_projects', args: { query: 'Marketing' } }), 1500);
      setTimeout(() => processEvent({ type: 'observation', content: 'Found project: Marketing (GID: 1234567890)' }), 3000);
      setTimeout(() => processEvent({ type: 'thought', content: "Now I'll create the task in this project." }), 4000);
      setTimeout(() => processEvent({ type: 'action', tool: 'create_task', args: { project_gid: '1234567890', name: 'Design Review', notes: '' } }), 5000);
      
      // Simulate confirmation requirement for demo
      if (query.toLowerCase().includes('delete')) {
        setTimeout(() => processEvent({ 
          type: 'requires_confirmation', 
          action_type: 'DELETE TASK', 
          resource: 'Design Review', 
          workspace: 'Marketing', 
          consequence: 'This will permanently delete the task and it cannot be recovered.' 
        }), 6000);
        return; // Pause demo here
      }

      setTimeout(() => processEvent({ type: 'observation', content: 'Task created successfully: GID 9876543210' }), 6500);
      setTimeout(() => {
        processEvent({ type: 'final_answer', content: "I've created the task 'Design Review' in the Marketing project. Task GID: 9876543210" });
        dispatch({ type: 'SET_STREAMING', isStreaming: false });
      }, 7500);
      return;
    }

    // REAL FETCH STREAM
    try {
      abortControllerRef.current = new AbortController();
      
      // We send the history without the new message since we just added it locally,
      // but usually the backend wants the full history.
      const payload = {
        query,
        sessionId: activeSession.id,
        history: activeSession.messages.map(m => ({ role: m.role, content: m.content }))
      };

      const res = await fetch(`${state.settings.vmBackendUrl}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(state.settings.vmBearerToken ? { Authorization: `Bearer ${state.settings.vmBearerToken}` } : {})
        },
        body: JSON.stringify(payload),
        signal: abortControllerRef.current.signal
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }

      if (!res.body) throw new Error("No response body");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        
        // Split by newlines to parse JSON lines or SSE
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep the last incomplete line in buffer

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          
          try {
            // Handle raw JSON lines or SSE "data: {...}" format
            const jsonStr = trimmed.startsWith('data: ') ? trimmed.slice(6) : trimmed;
            if (jsonStr === '[DONE]') continue;
            
            const event = JSON.parse(jsonStr);
            processEvent(event);
            
            if (event.type === 'final_answer' || event.type === 'error' || event.type === 'requires_confirmation') {
              // Wait if it's confirmation, otherwise stop streaming state will be handled
            }
          } catch (e) {
            console.warn('Failed to parse stream line:', line);
          }
        }
      }
      
      // The backend should send a final_answer which will turn off streaming,
      // but as a fallback, if stream ends:
      if (!state.pendingConfirmation) {
         dispatch({ type: 'SET_STREAMING', isStreaming: false });
      }

    } catch (err: any) {
      if (err.name === 'AbortError') return;
      dispatch({ type: 'SET_STREAMING', isStreaming: false });
      toast({ title: 'Connection Error', description: err.message || 'Failed to communicate with VM', variant: 'destructive' });
      processEvent({ type: 'error', message: err.message || 'Connection failed' });
    }
  }, [state.settings, activeSession, state.isStreaming, dispatch, processEvent, toast]);

  const confirmAction = useCallback(async (confirmed: boolean) => {
    dispatch({ type: 'SET_CONFIRMATION', request: null });
    
    if (!state.settings.vmBackendUrl) {
      // Demo mode continuation
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
          ...(state.settings.vmBearerToken ? { Authorization: `Bearer ${state.settings.vmBearerToken}` } : {})
        },
        body: JSON.stringify({ sessionId: activeSession?.id, confirmed })
      });
      
      // If the confirm endpoint also streams the rest of the execution:
      if (res.body) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; 

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;
            try {
              const jsonStr = trimmed.startsWith('data: ') ? trimmed.slice(6) : trimmed;
              if (jsonStr === '[DONE]') continue;
              processEvent(JSON.parse(jsonStr));
            } catch (e) {}
          }
        }
      }
      dispatch({ type: 'SET_STREAMING', isStreaming: false });
    } catch (err: any) {
      dispatch({ type: 'SET_STREAMING', isStreaming: false });
      toast({ title: 'Error', description: 'Failed to send confirmation', variant: 'destructive' });
    }

  }, [state.settings, activeSession?.id, dispatch, processEvent, toast]);

  const stopStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      dispatch({ type: 'SET_STREAMING', isStreaming: false });
    }
  }, [dispatch]);

  return { sendQuery, confirmAction, stopStream };
}
