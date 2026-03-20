export type Role = 'user' | 'agent';

export type TraceType =
  | 'thought'
  | 'action'
  | 'observation'
  | 'result'
  | 'error'
  | 'confirmation_required';

export interface ToolCall {
  tool: string;
  is_destructive: boolean;
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
  timestamp: number;
  content?: string;
  tool?: string;
  is_destructive?: boolean;
  status?: 'SUCCESS' | 'ERROR' | 'CANCELLED';
  session_id?: string;
  message?: string;
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

export interface Tool {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  is_destructive: boolean;
}

/** Typed union of all streamed agent event shapes from the VM backend */
export type AgentEvent =
  | { type: 'thought'; content: string }
  | { type: 'action'; content: { tool: string; is_destructive: boolean } }
  | { type: 'observation'; content: string }
  | { type: 'confirmation_required'; content: { message: string; session_id: string; tool: string } }
  | { type: 'result'; content: { status: 'SUCCESS' | 'ERROR' | 'CANCELLED'; message: string } };

export interface ConfirmationRequest {
  message: string;
  session_id: string;
  tool: string;
}

export type ConnectionStatus = 'unknown' | 'checking' | 'connected' | 'disconnected';
