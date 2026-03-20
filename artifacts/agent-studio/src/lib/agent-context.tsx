import { createContext, useContext, type ReactNode } from 'react';
import { useAgent } from '@/hooks/use-agent';

interface AgentContextValue {
  sendQuery: (query: string) => Promise<void>;
  confirmAction: (confirmed: boolean) => Promise<void>;
  stopStream: () => void;
}

const AgentContext = createContext<AgentContextValue | null>(null);

/** Single source of truth for the agent hook — instantiated once at the top of the tree. */
export function AgentProvider({ children }: { children: ReactNode }) {
  const agent = useAgent();
  return <AgentContext.Provider value={agent}>{children}</AgentContext.Provider>;
}

export function useAgentContext(): AgentContextValue {
  const ctx = useContext(AgentContext);
  if (!ctx) throw new Error('useAgentContext must be used within AgentProvider');
  return ctx;
}
