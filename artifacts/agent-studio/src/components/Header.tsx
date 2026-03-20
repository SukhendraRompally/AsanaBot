import { Zap, Settings, Activity, PanelRight } from 'lucide-react';
import { useAppStore } from '@/lib/store';
import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

export function Header() {
  const { state, dispatch } = useAppStore();

  const statusColor = 
    state.connectionStatus === 'connected' ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]' :
    state.connectionStatus === 'disconnected' ? 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.6)]' :
    state.connectionStatus === 'checking' ? 'bg-yellow-500 animate-pulse' :
    'bg-gray-500';

  const statusText = 
    state.connectionStatus === 'connected' ? 'Connected to VM' :
    state.connectionStatus === 'disconnected' ? 'VM Unreachable' :
    state.connectionStatus === 'checking' ? 'Checking connection...' :
    'Status Unknown';

  return (
    <header className="h-14 border-b border-border bg-card/50 backdrop-blur-md px-4 flex items-center justify-between shrink-0 sticky top-0 z-30">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center border border-primary/30">
          <Zap className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h1 className="font-semibold text-sm leading-tight text-foreground tracking-tight">Moveworks Agent Studio</h1>
          <p className="text-xs text-muted-foreground leading-tight">Asana ReAct Console</p>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-secondary/50 border border-border/50 text-xs text-muted-foreground mr-2 cursor-help">
              <span className={`w-2 h-2 rounded-full ${statusColor}`} />
              {state.settings.vmBackendUrl ? (
                <span className="max-w-[120px] truncate">
                  {(() => { try { return new URL(state.settings.vmBackendUrl).hostname; } catch { return state.settings.vmBackendUrl.slice(0, 20); } })()}
                </span>
              ) : (
                <span>Demo Mode</span>
              )}
            </div>
          </TooltipTrigger>
          <TooltipContent side="bottom">{statusText}</TooltipContent>
        </Tooltip>

        <Button 
          variant={state.showTrace ? "secondary" : "ghost"} 
          size="sm" 
          onClick={() => dispatch({ type: 'TOGGLE_TRACE' })}
          className={`gap-2 ${state.showTrace ? 'bg-primary/10 text-primary hover:bg-primary/20 border border-primary/20' : ''}`}
        >
          <PanelRight className="w-4 h-4" />
          <span className="hidden sm:inline">Trace</span>
        </Button>
        
        <Button 
          variant="ghost" 
          size="icon"
          onClick={() => dispatch({ type: 'TOGGLE_SETTINGS' })}
          className={state.showSettings ? 'bg-accent text-accent-foreground' : ''}
        >
          <Settings className="w-4 h-4" />
        </Button>
      </div>
    </header>
  );
}
