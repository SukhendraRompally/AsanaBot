import { useRef, useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BrainCircuit, Zap, CheckCircle2, Star, AlertTriangle, X, ChevronDown, ChevronRight, ShieldAlert } from 'lucide-react';
import { useAppStore } from '@/lib/store';
import { TraceStep } from '@/lib/types';
import { Button } from '@/components/ui/button';

/** Safe syntax-highlighted JSON rendered as React text nodes — no dangerouslySetInnerHTML */
type JsonToken = { kind: 'key' | 'string' | 'number' | 'boolean' | 'null' | 'punct'; value: string };

function tokenizeJson(raw: string): JsonToken[] {
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

function JsonTextView({ raw }: { raw: string }) {
  let formatted: string;
  try {
    formatted = JSON.stringify(JSON.parse(raw), null, 2);
  } catch {
    formatted = raw;
  }
  const tokens = tokenizeJson(formatted);
  return (
    <pre className="text-[11px] font-mono bg-muted/50 p-2 rounded border border-border/50 overflow-x-auto whitespace-pre-wrap break-all max-h-48 overflow-y-auto">
      {tokens.map((tok, idx) => (
        <span key={idx} className={TOKEN_COLORS[tok.kind]}>{tok.value}</span>
      ))}
    </pre>
  );
}

function TraceCard({ step }: { step: TraceStep }) {
  const [expanded, setExpanded] = useState(true);

  let icon: React.ReactNode;
  let color: string;
  let bg: string;
  let border: string;
  let title: string;

  switch (step.type) {
    case 'thought':
      icon = <BrainCircuit className="w-4 h-4" />;
      color = 'text-[#06b6d4]';
      bg = 'bg-[#06b6d4]/10';
      border = 'border-l-[#06b6d4]';
      title = 'Thought';
      break;
    case 'action':
      icon = <Zap className="w-4 h-4" />;
      color = step.is_destructive ? 'text-red-400' : 'text-[#f59e0b]';
      bg = step.is_destructive ? 'bg-red-400/10' : 'bg-[#f59e0b]/10';
      border = step.is_destructive ? 'border-l-red-400' : 'border-l-[#f59e0b]';
      title = 'Action';
      break;
    case 'observation':
      icon = <CheckCircle2 className="w-4 h-4" />;
      color = 'text-[#10b981]';
      bg = 'bg-[#10b981]/10';
      border = 'border-l-[#10b981]';
      title = 'Observation';
      break;
    case 'result': {
      const s = step.status;
      if (s === 'SUCCESS') {
        icon = <Star className="w-4 h-4" />;
        color = 'text-primary';
        bg = 'bg-primary/10';
        border = 'border-l-primary';
        title = 'Result · Success';
      } else if (s === 'CANCELLED') {
        icon = <X className="w-4 h-4" />;
        color = 'text-muted-foreground';
        bg = 'bg-muted/30';
        border = 'border-l-muted-foreground';
        title = 'Result · Cancelled';
      } else {
        icon = <AlertTriangle className="w-4 h-4" />;
        color = 'text-destructive';
        bg = 'bg-destructive/10';
        border = 'border-l-destructive';
        title = 'Result · Error';
      }
      break;
    }
    case 'confirmation_required':
      icon = <ShieldAlert className="w-4 h-4" />;
      color = 'text-destructive';
      bg = 'bg-destructive/10';
      border = 'border-l-destructive';
      title = 'Confirmation Required';
      break;
    case 'error':
    default:
      icon = <AlertTriangle className="w-4 h-4" />;
      color = 'text-destructive';
      bg = 'bg-destructive/10';
      border = 'border-l-destructive';
      title = 'Error';
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
        {step.type === 'action' && step.is_destructive && (
          <span className="text-[9px] font-bold uppercase tracking-widest px-1.5 py-0.5 rounded bg-red-500/20 text-red-400 border border-red-500/30">
            Approval Required
          </span>
        )}
        <span className="ml-auto text-[10px] text-muted-foreground font-mono">
          {new Date(step.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}
        </span>
      </div>

      <div className="pl-1">
        {step.type === 'thought' && (
          <p className="text-sm italic text-muted-foreground leading-relaxed">{step.content}</p>
        )}

        {step.type === 'action' && (
          <div className="font-mono text-xs px-2 py-1 bg-muted rounded inline-block text-foreground border border-border/50">
            {step.tool}
          </div>
        )}

        {step.type === 'observation' && step.content && (
          <div>
            <button
              onClick={() => setExpanded(!expanded)}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors mb-1"
            >
              {expanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              Raw response
            </button>
            {expanded && <JsonTextView raw={step.content} />}
          </div>
        )}

        {step.type === 'result' && (
          <p className="text-sm text-foreground leading-relaxed">{step.content}</p>
        )}

        {step.type === 'confirmation_required' && (
          <div className="space-y-2">
            <div className="inline-flex items-center gap-1.5 px-2 py-1 rounded bg-destructive/20 text-destructive text-xs font-mono font-bold">
              <ShieldAlert className="w-3 h-3" />
              {step.tool}
            </div>
            <p className="text-sm text-foreground/90 leading-relaxed">{step.message}</p>
          </div>
        )}

        {step.type === 'error' && (
          <p className="text-sm text-destructive">{step.message ?? step.content}</p>
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
      transition={{ type: 'spring', bounce: 0, duration: 0.3 }}
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
