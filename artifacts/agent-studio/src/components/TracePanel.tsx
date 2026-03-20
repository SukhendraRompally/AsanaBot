import { useRef, useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BrainCircuit, Zap, CheckCircle2, Star, AlertTriangle, X, ChevronDown, ChevronRight } from 'lucide-react';
import { useAppStore } from '@/lib/store';
import { TraceStep } from '@/lib/types';
import { Button } from '@/components/ui/button';

/** Tokenize a JSON string into typed segments for safe colorized rendering */
type JsonToken = { kind: 'key' | 'string' | 'number' | 'boolean' | 'null' | 'punct'; value: string };

function tokenizeJson(obj: Record<string, unknown>): JsonToken[] {
  const raw = JSON.stringify(obj, null, 2);
  const tokens: JsonToken[] = [];
  let i = 0;
  while (i < raw.length) {
    if (raw[i] === '"') {
      let j = i + 1;
      while (j < raw.length && !(raw[j] === '"' && raw[j - 1] !== '\\')) j++;
      const full = raw.slice(i, j + 1);
      const isKey = raw.slice(j + 1).trimStart().startsWith(':');
      tokens.push({ kind: isKey ? 'key' : 'string', value: full });
      i = j + 1;
    } else if (/[\d\-]/.test(raw[i])) {
      let j = i;
      while (j < raw.length && /[\d.eE+\-]/.test(raw[j])) j++;
      tokens.push({ kind: 'number', value: raw.slice(i, j) });
      i = j;
    } else if (raw.startsWith('true', i) || raw.startsWith('false', i)) {
      const val = raw.startsWith('true', i) ? 'true' : 'false';
      tokens.push({ kind: 'boolean', value: val });
      i += val.length;
    } else if (raw.startsWith('null', i)) {
      tokens.push({ kind: 'null', value: 'null' });
      i += 4;
    } else {
      tokens.push({ kind: 'punct', value: raw[i] });
      i++;
    }
  }
  return tokens;
}

const TOKEN_COLORS: Record<JsonToken['kind'], string> = {
  key: 'text-sky-400',
  string: 'text-green-400',
  number: 'text-amber-400',
  boolean: 'text-purple-400',
  null: 'text-red-400',
  punct: 'text-muted-foreground',
};

function JsonView({ args }: { args: Record<string, unknown> }) {
  const tokens = tokenizeJson(args);
  return (
    <pre className="text-[11px] font-mono bg-muted/50 p-2 rounded border border-border/50 overflow-x-auto whitespace-pre-wrap break-all">
      {tokens.map((tok, idx) => (
        <span key={idx} className={TOKEN_COLORS[tok.kind]}>{tok.value}</span>
      ))}
    </pre>
  );
}

function TraceCard({ step }: { step: TraceStep }) {
  const [expanded, setExpanded] = useState(false);

  let icon, color, bg, border, title;

  switch (step.type) {
    case 'thought':
      icon = <BrainCircuit className="w-4 h-4" />;
      color = 'text-[#06b6d4]'; // cyan
      bg = 'bg-[#06b6d4]/10';
      border = 'border-l-[#06b6d4]';
      title = 'Thought';
      break;
    case 'action':
      icon = <Zap className="w-4 h-4" />;
      color = 'text-[#f59e0b]'; // amber
      bg = 'bg-[#f59e0b]/10';
      border = 'border-l-[#f59e0b]';
      title = 'Action';
      break;
    case 'observation':
      icon = <CheckCircle2 className="w-4 h-4" />;
      color = 'text-[#10b981]'; // green
      bg = 'bg-[#10b981]/10';
      border = 'border-l-[#10b981]';
      title = 'Observation';
      break;
    case 'final_answer':
      icon = <Star className="w-4 h-4" />;
      color = 'text-primary';
      bg = 'bg-primary/10';
      border = 'border-l-primary';
      title = 'Final Answer';
      break;
    case 'error':
    case 'requires_confirmation':
      icon = <AlertTriangle className="w-4 h-4" />;
      color = 'text-destructive';
      bg = 'bg-destructive/10';
      border = 'border-l-destructive';
      title = step.type === 'error' ? 'Error' : 'Requires Confirmation';
      break;
  }

  return (
    <motion.div 
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className={`border border-border border-l-4 ${border} bg-card rounded-lg p-3 shadow-sm`}
    >
      <div className="flex items-center gap-2 mb-2">
        <div className={`p-1 rounded-md ${bg} ${color}`}>{icon}</div>
        <span className={`text-xs font-semibold uppercase tracking-wider ${color}`}>{title}</span>
        <span className="ml-auto text-[10px] text-muted-foreground font-mono">
          {new Date(step.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute:'2-digit', second:'2-digit', fractionalSecondDigits: 3 })}
        </span>
      </div>

      <div className="pl-1">
        {step.type === 'thought' && (
          <p className="text-sm italic text-muted-foreground leading-relaxed">{step.content}</p>
        )}
        
        {step.type === 'action' && (
          <div className="space-y-2">
            <div className="font-mono text-xs px-2 py-1 bg-muted rounded inline-block text-foreground border border-border/50">
              {step.tool}
            </div>
            {step.args && (
              <div className="mt-2">
                <button
                  onClick={() => setExpanded(!expanded)}
                  className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors mb-1"
                >
                  {expanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                  Arguments
                </button>
                {expanded && <JsonView args={step.args} />}
              </div>
            )}
          </div>
        )}

        {step.type === 'observation' && (
          <div className="text-sm text-foreground/90 bg-muted/30 p-2 rounded font-mono text-[11px] overflow-x-auto border border-border/50 max-h-32 overflow-y-auto">
            {step.content}
          </div>
        )}

        {step.type === 'final_answer' && (
          <p className="text-sm text-foreground leading-relaxed font-medium">{step.content}</p>
        )}

        {step.type === 'requires_confirmation' && (
          <div className="text-sm space-y-1">
            <div className="font-mono text-xs px-2 py-1 bg-destructive/20 text-destructive rounded inline-block mb-1">
              {step.action_type}
            </div>
            <p className="font-medium text-foreground">{step.resource}</p>
            <p className="text-muted-foreground text-xs">{step.consequence}</p>
          </div>
        )}

        {step.type === 'error' && (
          <p className="text-sm text-destructive">{step.message}</p>
        )}
      </div>
    </motion.div>
  );
}

export function TracePanel() {
  const { state, dispatch, activeSession } = useAppStore();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [activeSession?.traces, state.isStreaming]);

  if (!state.showTrace) return null;

  const traces = activeSession?.traces || [];

  return (
    <motion.div 
      initial={{ width: 0, opacity: 0 }}
      animate={{ width: 400, opacity: 1 }}
      exit={{ width: 0, opacity: 0 }}
      transition={{ type: "spring", bounce: 0, duration: 0.3 }}
      className="h-full border-l border-border bg-sidebar/50 backdrop-blur-sm flex flex-col shrink-0 overflow-hidden"
    >
      <div className="h-14 px-4 flex items-center justify-between border-b border-border bg-card/50 shrink-0">
        <h2 className="font-semibold text-sm text-foreground tracking-tight flex items-center gap-2">
          <BrainCircuit className="w-4 h-4 text-primary" />
          Thought Trace
        </h2>
        <Button 
          variant="ghost" 
          size="icon" 
          className="h-8 w-8 text-muted-foreground hover:text-foreground"
          onClick={() => dispatch({ type: 'TOGGLE_TRACE' })}
        >
          <X className="w-4 h-4" />
        </Button>
      </div>

      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-4 space-y-4 scroll-smooth"
      >
        {traces.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-muted-foreground/50">
            <Zap className="w-8 h-8 mb-2 opacity-50" />
            <p className="text-xs text-center max-w-[200px]">Traces will appear here when the agent runs.</p>
          </div>
        ) : (
          <AnimatePresence initial={false}>
            {traces.map(step => (
              <TraceCard key={step.id} step={step} />
            ))}
          </AnimatePresence>
        )}

        {state.isStreaming && !state.pendingConfirmation && (
          <motion.div 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="flex items-center justify-center gap-2 py-4 text-xs text-muted-foreground font-mono uppercase tracking-widest"
          >
            <span className="w-1.5 h-1.5 bg-primary/70 rounded-full animate-ping" style={{ animationDelay: '0ms' }} />
            <span className="w-1.5 h-1.5 bg-primary/70 rounded-full animate-ping" style={{ animationDelay: '200ms' }} />
            <span className="w-1.5 h-1.5 bg-primary/70 rounded-full animate-ping" style={{ animationDelay: '400ms' }} />
            <span className="ml-2">Agent is reasoning</span>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}
