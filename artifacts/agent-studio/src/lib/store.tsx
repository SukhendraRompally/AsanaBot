import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Session, Message, TraceStep, Settings, ConfirmationRequest, ConnectionStatus, Tool } from './types';

interface AppState {
  sessions: Session[];
  activeSessionId: string;
  isStreaming: boolean;
  showTrace: boolean;
  showSettings: boolean;
  settings: Settings;
  connectionStatus: ConnectionStatus;
  pendingConfirmation: ConfirmationRequest | null;
  vmSessionId: string | null;
  tools: Tool[];
}

type Action =
  | { type: 'INIT_STATE'; state: Partial<AppState> }
  | { type: 'NEW_SESSION' }
  | { type: 'SWITCH_SESSION'; id: string }
  | { type: 'ADD_MESSAGE'; message: Message }
  | { type: 'ENSURE_AGENT_MESSAGE'; agentMsgId: string }
  | { type: 'UPDATE_AGENT_MESSAGE'; agentMsgId: string; content: string }
  | { type: 'SET_RESULT_MESSAGE'; agentMsgId: string; content: string }
  | { type: 'ADD_TRACE'; trace: TraceStep; agentMsgId: string }
  | { type: 'SET_STREAMING'; isStreaming: boolean }
  | { type: 'TOGGLE_TRACE' }
  | { type: 'TOGGLE_SETTINGS'; show?: boolean }
  | { type: 'UPDATE_SETTINGS'; settings: Partial<Settings> }
  | { type: 'SET_CONNECTION_STATUS'; status: ConnectionStatus }
  | { type: 'SET_CONFIRMATION'; request: ConfirmationRequest | null }
  | { type: 'SET_VM_SESSION_ID'; sessionId: string | null }
  | { type: 'SET_TOOLS'; tools: Tool[] };

const createNewSession = (): Session => ({
  id: uuidv4(),
  title: 'New Conversation',
  messages: [],
  traces: [],
  timestamp: Date.now(),
});

const DEFAULT_VM_URL = 'http://20.98.68.88:8001';

const defaultSettings: Settings = {
  vmBackendUrl: DEFAULT_VM_URL,
  vmBearerToken: '',
  vmHealthPath: '/health',
};

const initialState: AppState = {
  sessions: [createNewSession()],
  activeSessionId: '',
  isStreaming: false,
  showTrace: true,
  showSettings: false,
  settings: defaultSettings,
  connectionStatus: 'unknown',
  pendingConfirmation: null,
  vmSessionId: null,
  tools: [],
};

initialState.activeSessionId = initialState.sessions[0].id;

const AppContext = createContext<{
  state: AppState;
  dispatch: React.Dispatch<Action>;
  activeSession: Session | undefined;
} | null>(null);

function appReducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'INIT_STATE':
      return { ...state, ...action.state };
    case 'NEW_SESSION': {
      const newSession = createNewSession();
      return {
        ...state,
        sessions: [newSession, ...state.sessions],
        activeSessionId: newSession.id,
        vmSessionId: null,
      };
    }
    case 'SWITCH_SESSION':
      return { ...state, activeSessionId: action.id, vmSessionId: null };
    case 'ADD_MESSAGE': {
      return {
        ...state,
        sessions: state.sessions.map((s) => {
          if (s.id !== state.activeSessionId) return s;
          let title = s.title;
          if (s.messages.length === 0 && action.message.role === 'user') {
            title = action.message.content.slice(0, 40) + (action.message.content.length > 40 ? '...' : '');
          }
          return { ...s, title, messages: [...s.messages, action.message], timestamp: Date.now() };
        }),
      };
    }
    case 'ENSURE_AGENT_MESSAGE': {
      return {
        ...state,
        sessions: state.sessions.map((s) => {
          if (s.id !== state.activeSessionId) return s;
          const lastUserIdx = [...s.messages].reverse().findIndex(m => m.role === 'user');
          const messagesAfterUser = lastUserIdx === -1 ? s.messages.length : lastUserIdx;
          const agentMsgExistsForThisTurn = messagesAfterUser > 0 &&
            s.messages.slice(s.messages.length - messagesAfterUser).some(m => m.role === 'agent');
          if (agentMsgExistsForThisTurn) return s;
          return {
            ...s,
            messages: [...s.messages, { id: action.agentMsgId, role: 'agent' as const, content: '', timestamp: Date.now() }],
          };
        }),
      };
    }
    case 'UPDATE_AGENT_MESSAGE': {
      return {
        ...state,
        sessions: state.sessions.map((s) => {
          if (s.id !== state.activeSessionId) return s;
          return {
            ...s,
            messages: s.messages.map((m) =>
              m.id === action.agentMsgId ? { ...m, content: action.content } : m
            ),
          };
        }),
      };
    }
    // Single atomic action: find-and-update the agent bubble, or create it with content
    // already set — no two-step ENSURE+UPDATE sequencing needed.
    case 'SET_RESULT_MESSAGE': {
      return {
        ...state,
        sessions: state.sessions.map((s) => {
          if (s.id !== state.activeSessionId) return s;
          const existingIdx = s.messages.findIndex((m) => m.id === action.agentMsgId);
          if (existingIdx !== -1) {
            const msgs = [...s.messages];
            msgs[existingIdx] = { ...msgs[existingIdx], content: action.content };
            return { ...s, messages: msgs };
          }
          // No agent bubble yet — create one with content already populated
          return {
            ...s,
            messages: [
              ...s.messages,
              { id: action.agentMsgId, role: 'agent' as const, content: action.content, timestamp: Date.now() },
            ],
          };
        }),
      };
    }
    case 'ADD_TRACE': {
      return {
        ...state,
        sessions: state.sessions.map((s) => {
          if (s.id !== state.activeSessionId) return s;

          let updatedMessages = [...s.messages];

          if (action.trace.type === 'action' && action.trace.tool) {
            const targetIdx = updatedMessages.findIndex(m => m.id === action.agentMsgId);
            if (targetIdx !== -1) {
              updatedMessages[targetIdx] = {
                ...updatedMessages[targetIdx],
                toolCalls: [
                  ...(updatedMessages[targetIdx].toolCalls || []),
                  { tool: action.trace.tool, is_destructive: action.trace.is_destructive ?? false },
                ],
              };
            }
          } else if (action.trace.type === 'result' && action.trace.content) {
            const targetIdx = updatedMessages.findIndex(m => m.id === action.agentMsgId);
            if (targetIdx !== -1) {
              updatedMessages[targetIdx] = {
                ...updatedMessages[targetIdx],
                content: action.trace.content,
              };
            }
          }

          return { ...s, traces: [...s.traces, action.trace], messages: updatedMessages };
        }),
      };
    }
    case 'SET_STREAMING':
      return { ...state, isStreaming: action.isStreaming };
    case 'TOGGLE_TRACE':
      return { ...state, showTrace: !state.showTrace };
    case 'TOGGLE_SETTINGS':
      return { ...state, showSettings: action.show !== undefined ? action.show : !state.showSettings };
    case 'UPDATE_SETTINGS':
      return { ...state, settings: { ...state.settings, ...action.settings } };
    case 'SET_CONNECTION_STATUS':
      return { ...state, connectionStatus: action.status };
    case 'SET_CONFIRMATION':
      return { ...state, pendingConfirmation: action.request };
    case 'SET_VM_SESSION_ID':
      return { ...state, vmSessionId: action.sessionId };
    case 'SET_TOOLS':
      return { ...state, tools: action.tools };
    default:
      return state;
  }
}

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  useEffect(() => {
    try {
      const savedData = localStorage.getItem('agentStudioState');
      if (savedData) {
        const parsed = JSON.parse(savedData);
        const storedSettings = parsed.settings ?? {};
        dispatch({
          type: 'INIT_STATE',
          state: {
            sessions: parsed.sessions?.length ? parsed.sessions : initialState.sessions,
            activeSessionId: parsed.activeSessionId || initialState.activeSessionId,
            settings: {
              ...defaultSettings,
              ...storedSettings,
              // Migrate: if the stored URL is blank, fall back to the new default
              vmBackendUrl: storedSettings.vmBackendUrl || defaultSettings.vmBackendUrl,
            },
          },
        });
      }
    } catch (e) {
      console.error('Failed to load state from localStorage', e);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('agentStudioState', JSON.stringify({
      sessions: state.sessions,
      activeSessionId: state.activeSessionId,
      settings: state.settings,
    }));
  }, [state.sessions, state.activeSessionId, state.settings]);

  const activeSession = state.sessions.find((s) => s.id === state.activeSessionId);

  return (
    <AppContext.Provider value={{ state, dispatch, activeSession }}>
      {children}
    </AppContext.Provider>
  );
}

export function useAppStore() {
  const context = useContext(AppContext);
  if (!context) throw new Error('useAppStore must be used within AppProvider');
  return context;
}
