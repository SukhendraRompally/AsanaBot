import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Send, User, Bot, Wrench, History } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAppStore } from '@/lib/store';
import { useAgentContext } from '@/lib/agent-context';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';

export function ChatPanel() {
  const { state, dispatch, activeSession } = useAppStore();
  const { sendQuery } = useAgentContext();
  const [input, setInput] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [activeSession?.messages, state.isStreaming]);

  const handleSubmit = () => {
    if (!input.trim() || state.isStreaming) return;
    sendQuery(input);
    setInput('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const messages = activeSession?.messages || [];

  const newestSessionId = state.sessions.reduce(
    (max, s) => (s.timestamp > (state.sessions.find(x => x.id === max)?.timestamp ?? 0) ? s.id : max),
    state.sessions[0]?.id ?? ''
  );
  const isHistorical = activeSession && activeSession.id !== newestSessionId && messages.length > 0;

  return (
    <div className="flex-1 flex flex-col h-full bg-background relative min-w-0">
      {isHistorical && (
        <div className="flex items-center gap-2 px-4 py-2 bg-amber-500/10 border-b border-amber-500/20 text-amber-400 text-xs shrink-0">
          <History className="w-3.5 h-3.5 shrink-0" />
          <span>You are viewing a historical session. Start a new session to continue chatting.</span>
          <Button
            variant="ghost"
            size="sm"
            className="ml-auto h-6 px-2 text-xs text-amber-400 hover:text-amber-300 hover:bg-amber-500/20"
            onClick={() => dispatch({ type: 'NEW_SESSION' })}
          >
            New Session
          </Button>
        </div>
      )}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 py-6 sm:px-6 md:px-8 space-y-8 scroll-smooth"
      >
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-muted-foreground">
            <Bot className="w-12 h-12 mb-4 opacity-20" />
            <h3 className="text-lg font-medium text-foreground mb-2">How can I help you?</h3>
            <p className="text-sm text-center max-w-md">
              Ask me to manage your Asana tasks, projects, or workspaces. 
              Try asking "Create a task called Design Review in the Marketing project".
            </p>
          </div>
        ) : (
          <AnimatePresence initial={false}>
            {messages.map((msg) => (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex gap-4 max-w-3xl ${msg.role === 'user' ? 'ml-auto flex-row-reverse' : 'mr-auto'}`}
              >
                {/* Avatar */}
                <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-1 shadow-sm ${
                  msg.role === 'user' ? 'bg-secondary text-secondary-foreground' : 'bg-primary text-primary-foreground'
                }`}>
                  {msg.role === 'user' ? <User className="w-4 h-4" /> : <span className="text-xs font-bold font-mono">MW</span>}
                </div>

                {/* Message Content */}
                <div className={`flex flex-col gap-2 min-w-0 ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                  
                  {/* Tool Badges */}
                  {msg.toolCalls && msg.toolCalls.length > 0 && (
                    <div className="flex flex-wrap gap-2 mb-1">
                      {msg.toolCalls.map((tc, idx) => (
                        <div key={idx} className="flex items-center gap-1.5 px-2 py-1 bg-amber-500/10 border border-amber-500/20 text-amber-500 rounded text-[11px] font-mono shadow-sm">
                          <Wrench className="w-3 h-3" />
                          <span>{tc.tool}</span>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Text Bubble */}
                  {msg.content && (
                    <div className={`px-5 py-3.5 rounded-2xl text-sm leading-relaxed shadow-sm break-words ${
                      msg.role === 'user' 
                        ? 'bg-secondary text-secondary-foreground rounded-tr-sm' 
                        : 'bg-card border border-border text-card-foreground rounded-tl-sm markdown-body'
                    }`}>
                      {msg.role === 'user' ? (
                        <div className="whitespace-pre-wrap">{msg.content}</div>
                      ) : (
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      )}
                    </div>
                  )}
                  
                  {/* Typing indicator if agent message is empty (waiting for final answer) */}
                  {msg.role === 'agent' && !msg.content && state.isStreaming && (
                    <div className="px-5 py-3.5 rounded-2xl bg-card border border-border rounded-tl-sm flex items-center gap-1.5 h-12">
                      <span className="w-2 h-2 bg-primary/50 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <span className="w-2 h-2 bg-primary/50 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <span className="w-2 h-2 bg-primary/50 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        )}
      </div>

      {/* Input Area */}
      <div className="p-4 bg-background border-t border-border shrink-0">
        <div className="max-w-3xl mx-auto relative group">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={state.isStreaming ? "Agent is working..." : "Ask your AsanaBot..."}
            disabled={state.isStreaming || !activeSession}
            className="min-h-[60px] max-h-40 resize-none pr-14 py-4 bg-input/50 focus-visible:bg-input border-border focus-visible:ring-primary/50 rounded-xl shadow-sm transition-all duration-200"
            rows={1}
          />
          <Button 
            size="icon" 
            onClick={handleSubmit}
            disabled={!input.trim() || state.isStreaming || !activeSession}
            className="absolute right-2 bottom-2 rounded-lg bg-primary hover:bg-primary/90 text-primary-foreground shadow-sm transition-all disabled:opacity-50"
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
