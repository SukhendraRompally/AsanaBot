import { useState, useEffect } from 'react';
import { formatDistanceToNow } from 'date-fns';
import { MessageSquare, Plus, PanelLeftClose, PanelLeftOpen, Wrench, ShieldAlert, ChevronDown, ChevronRight } from 'lucide-react';
import { useAppStore } from '@/lib/store';
import { Tool } from '@/lib/types';
import { vmProxyUrl, vmHeaders } from '@/lib/vm-fetch';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { motion, AnimatePresence } from 'framer-motion';

function ToolPalette({ tools, collapsed }: { tools: Tool[]; collapsed: boolean }) {
  const [open, setOpen] = useState(true);

  if (collapsed || tools.length === 0) return null;

  return (
    <div className="border-t border-sidebar-border mt-2 pt-2 px-3 pb-2">
      <button
        className="flex items-center gap-2 w-full text-xs font-semibold text-sidebar-foreground/60 uppercase tracking-widest mb-2 hover:text-sidebar-foreground transition-colors"
        onClick={() => setOpen(!open)}
      >
        <Wrench className="w-3 h-3" />
        <span className="flex-1 text-left">Tools ({tools.length})</span>
        {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="space-y-1.5 max-h-64 overflow-y-auto pr-1">
              {tools.map(tool => (
                <div key={tool.name} className="group px-2.5 py-2 rounded-lg bg-sidebar-accent/40 border border-sidebar-border/50 hover:bg-sidebar-accent/70 transition-colors cursor-default">
                  <div className="flex items-center gap-1.5 mb-0.5">
                    <span className="text-[11px] font-mono font-semibold text-sidebar-foreground truncate flex-1">
                      {tool.name}
                    </span>
                    {tool.is_destructive && (
                      <span className="flex items-center gap-0.5 text-[9px] font-bold uppercase tracking-wider text-red-400 bg-red-500/15 px-1 py-0.5 rounded border border-red-500/20 shrink-0">
                        <ShieldAlert className="w-2.5 h-2.5" />
                        destructive
                      </span>
                    )}
                  </div>
                  <p className="text-[10px] text-sidebar-foreground/50 leading-tight line-clamp-2">
                    {tool.description}
                  </p>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function Sidebar() {
  const { state, dispatch } = useAppStore();
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    if (!state.settings.vmBackendUrl) return;

    const fetchTools = async () => {
      try {
        const res = await fetch(vmProxyUrl('/tools'), {
          headers: vmHeaders(state.settings),
        });
        if (!res.ok) return;
        const data: unknown = await res.json();
        if (Array.isArray(data)) {
          dispatch({ type: 'SET_TOOLS', tools: data as Tool[] });
        }
      } catch {
        // Tools palette is non-critical; swallow errors silently
      }
    };

    fetchTools();
  }, [state.settings.vmBackendUrl, state.settings.vmBearerToken, dispatch]);

  return (
    <motion.aside
      initial={false}
      animate={{ width: collapsed ? 64 : 260 }}
      className="h-full border-r border-border bg-sidebar flex flex-col shrink-0 overflow-hidden"
    >
      <div className="h-14 flex items-center justify-between px-3 border-b border-sidebar-border shrink-0">
        <AnimatePresence>
          {!collapsed && (
            <motion.span
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="font-medium text-sidebar-foreground text-sm pl-1"
            >
              Sessions
            </motion.span>
          )}
        </AnimatePresence>
        <Button
          variant="ghost"
          size="icon"
          className="text-sidebar-foreground/70 hover:text-sidebar-foreground"
          onClick={() => setCollapsed(!collapsed)}
        >
          {collapsed ? <PanelLeftOpen className="w-4 h-4" /> : <PanelLeftClose className="w-4 h-4" />}
        </Button>
      </div>

      <div className="p-3 shrink-0">
        <Button
          onClick={() => dispatch({ type: 'NEW_SESSION' })}
          className={`w-full bg-sidebar-primary hover:bg-sidebar-primary/90 text-sidebar-primary-foreground ${collapsed ? 'px-0 justify-center' : 'justify-start gap-2'}`}
        >
          <Plus className="w-4 h-4" />
          {!collapsed && <span>New Session</span>}
        </Button>
      </div>

      <ScrollArea className="flex-1 px-3 pb-1">
        <div className="space-y-1">
          {state.sessions.map((session) => {
            const isActive = session.id === state.activeSessionId;
            return (
              <button
                key={session.id}
                onClick={() => dispatch({ type: 'SWITCH_SESSION', id: session.id })}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all duration-200 text-left ${
                  isActive
                    ? 'bg-sidebar-accent text-sidebar-accent-foreground shadow-sm ring-1 ring-sidebar-border'
                    : 'text-sidebar-foreground/70 hover:bg-sidebar-accent/50 hover:text-sidebar-foreground'
                } ${collapsed ? 'justify-center px-0' : ''}`}
                title={collapsed ? session.title : undefined}
              >
                <MessageSquare className={`w-4 h-4 shrink-0 ${isActive ? 'text-primary' : ''}`} />
                {!collapsed && (
                  <div className="flex-1 overflow-hidden">
                    <div className="truncate font-medium leading-tight">{session.title}</div>
                    <div className="text-[10px] text-muted-foreground mt-0.5 opacity-80">
                      {formatDistanceToNow(session.timestamp, { addSuffix: true })}
                    </div>
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </ScrollArea>

      <ToolPalette tools={state.tools} collapsed={collapsed} />
    </motion.aside>
  );
}
