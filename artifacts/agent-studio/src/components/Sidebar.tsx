import { useState } from 'react';
import { formatDistanceToNow } from 'date-fns';
import { MessageSquare, Plus, PanelLeftClose, PanelLeftOpen } from 'lucide-react';
import { useAppStore } from '@/lib/store';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { motion, AnimatePresence } from 'framer-motion';

export function Sidebar() {
  const { state, dispatch } = useAppStore();
  const [collapsed, setCollapsed] = useState(false);

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

      <div className="p-3">
        <Button 
          onClick={() => dispatch({ type: 'NEW_SESSION' })}
          className={`w-full bg-sidebar-primary hover:bg-sidebar-primary/90 text-sidebar-primary-foreground ${collapsed ? 'px-0 justify-center' : 'justify-start gap-2'}`}
        >
          <Plus className="w-4 h-4" />
          {!collapsed && <span>New Session</span>}
        </Button>
      </div>

      <ScrollArea className="flex-1 px-3 pb-3">
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
                    <div className="truncate font-medium leading-tight">
                      {session.title}
                    </div>
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
    </motion.aside>
  );
}
