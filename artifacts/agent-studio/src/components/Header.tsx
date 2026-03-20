import { Zap, PanelRight } from 'lucide-react';
import { useAppStore } from '@/lib/store';
import { Button } from '@/components/ui/button';

export function Header() {
  const { state, dispatch } = useAppStore();

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
        <Button
          variant={state.showTrace ? 'secondary' : 'ghost'}
          size="sm"
          onClick={() => dispatch({ type: 'TOGGLE_TRACE' })}
          className={`gap-2 ${state.showTrace ? 'bg-primary/10 text-primary hover:bg-primary/20 border border-primary/20' : ''}`}
        >
          <PanelRight className="w-4 h-4" />
          <span className="hidden sm:inline">Trace</span>
        </Button>
      </div>
    </header>
  );
}
