import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Session, Message, TraceStep, Settings, ConfirmationRequest, ConnectionStatus } from './types';

interface AppState {
  sessions: Session[];
  activeSessionId: string;
  isStreaming: boolean;
  showTrace: boolean;
  showSettings: boolean;
  settings: Settings;
  connectionStatus: ConnectionStatus;
  pendingConfirmation: ConfirmationRequest | null;
}

type Action =
  | { type: 'INIT_STATE'; state: Partial<AppState> }
  | { type: 'NEW_SESSION' }
  | { type: 'SWITCH_SESSION'; id: string }
  | { type: 'ADD_MESSAGE'; message: Message }
  | { type: 'ADD_TRACE'; trace: TraceStep }
  | { type: 'SET_STREAMING'; isStreaming: boolean }
  | { type: 'TOGGLE_TRACE' }
  | { type: 'TOGGLE_SETTINGS'; show?: boolean }
  | { type: 'UPDATE_SETTINGS'; settings: Partial<Settings> }
  | { type: 'SET_CONNECTION_STATUS'; status: ConnectionStatus }
  | { type: 'SET_CONFIRMATION'; request: ConfirmationRequest | null };

const createNewSession = (): Session => ({
  id: uuidv4(),
  title: 'New Conversation',
  messages: [],
  traces: [],
  timestamp: Date.now(),
});

const defaultSettings: Settings = {
  vmBackendUrl: '',
  vmBearerToken: '',
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
};

// Fix initial active session ID
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
      };
    }
    case 'SWITCH_SESSION':
      return { ...state, activeSessionId: action.id };
    case 'ADD_MESSAGE': {
      return {
        ...state,
        sessions: state.sessions.map((s) => {
          if (s.id !== state.activeSessionId) return s;
          
          // Auto-generate title from first user message if it's currently "New Conversation"
          let title = s.title;
          if (s.messages.length === 0 && action.message.role === 'user') {
            title = action.message.content.slice(0, 40) + (action.message.content.length > 40 ? '...' : '');
          }

          // If the message has tool calls, and the last message was an agent message,
          // we might want to append the tool call to it instead of creating a new message.
          // For simplicity, we just add it to the messages array.
          
          return {
            ...s,
            title,
            messages: [...s.messages, action.message],
            timestamp: Date.now(),
          };
        }),
      };
    }
    case 'ADD_TRACE': {
      return {
        ...state,
        sessions: state.sessions.map((s) => {
          if (s.id !== state.activeSessionId) return s;
          
          // Update the last agent message with tool badges if this is an action
          let updatedMessages = [...s.messages];
          if (action.trace.type === 'action' && action.trace.tool) {
             const lastMsgIndex = updatedMessages.findLastIndex(m => m.role === 'agent');
             if (lastMsgIndex !== -1) {
                const lastMsg = updatedMessages[lastMsgIndex];
                updatedMessages[lastMsgIndex] = {
                   ...lastMsg,
                   toolCalls: [...(lastMsg.toolCalls || []), { tool: action.trace.tool, args: action.trace.args }]
                };
             } else {
                // If no agent message exists yet for this turn, create an empty one to hold badges
                updatedMessages.push({
                   id: uuidv4(),
                   role: 'agent',
                   content: '',
                   timestamp: Date.now(),
                   toolCalls: [{ tool: action.trace.tool, args: action.trace.args }]
                });
             }
          } else if (action.trace.type === 'final_answer' && action.trace.content) {
             // Append final answer content to the last agent message
             const lastMsgIndex = updatedMessages.findLastIndex(m => m.role === 'agent');
             if (lastMsgIndex !== -1) {
                updatedMessages[lastMsgIndex] = {
                   ...updatedMessages[lastMsgIndex],
                   content: action.trace.content
                };
             } else {
                updatedMessages.push({
                   id: uuidv4(),
                   role: 'agent',
                   content: action.trace.content,
                   timestamp: Date.now()
                });
             }
          }

          return {
            ...s,
            traces: [...s.traces, action.trace],
            messages: updatedMessages
          };
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
    default:
      return state;
  }
}

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  // Load from local storage on mount
  useEffect(() => {
    try {
      const savedData = localStorage.getItem('agentStudioState');
      if (savedData) {
        const parsed = JSON.parse(savedData);
        dispatch({ 
          type: 'INIT_STATE', 
          state: {
            sessions: parsed.sessions?.length ? parsed.sessions : initialState.sessions,
            activeSessionId: parsed.activeSessionId || initialState.activeSessionId,
            settings: parsed.settings || initialState.settings,
          } 
        });
      }
    } catch (e) {
      console.error('Failed to load state from localStorage', e);
    }
  }, []);

  // Save to local storage on change
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
