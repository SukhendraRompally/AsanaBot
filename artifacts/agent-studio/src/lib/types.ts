export type Role = 'user' | 'agent';

export type TraceType = 
  | 'thought' 
  | 'action' 
  | 'observation' 
  | 'final_answer' 
  | 'error' 
  | 'requires_confirmation';

export interface ToolCall {
  tool: string;
  args: Record<string, unknown>;
}

export interface Message {
  id: string;
  role: Role;
  content: string;
  timestamp: number;
  toolCalls?: ToolCall[];
}

export interface TraceStep {
  id: string;
  type: TraceType;
  content?: string;
  tool?: string;
  args?: Record<string, unknown>;
  action_type?: string;
  resource?: string;
  workspace?: string;
  consequence?: string;
  message?: string;
  timestamp: number;
}

export interface Session {
  id: string;
  title: string;
  messages: Message[];
  traces: TraceStep[];
  timestamp: number;
}

export interface Settings {
  vmBackendUrl: string;
  vmBearerToken: string;
  vmHealthPath: string;
}

/** Typed union of all streamed agent event shapes */
export type AgentEvent =
  | { type: 'thought'; content: string }
  | { type: 'action'; tool: string; args: Record<string, unknown> }
  | { type: 'observation'; content: string }
  | { type: 'final_answer'; content: string }
  | { type: 'requires_confirmation'; action_type: string; resource: string; workspace: string; consequence: string }
  | { type: 'error'; message: string };

export interface ConfirmationRequest {
  action_type: string;
  resource: string;
  workspace: string;
  consequence: string;
}

export type ConnectionStatus = 'unknown' | 'checking' | 'connected' | 'disconnected';
