import { Zap, PanelRight, Github } from 'lucide-react';
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
          <h1 className="font-semibold text-sm leading-tight text-foreground tracking-tight">Agent Studio</h1>
          <p className="text-xs text-muted-foreground leading-tight">Asana ReAct Console</p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <a
          href="https://github.com/SukhendraRompally/AsanaBot"
          target="_blank"
          rel="noopener noreferrer"
          className="hidden sm:flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors duration-150"
        >
          <Github className="w-4 h-4 shrink-0" />
          <span>Read the full documentation on Github</span>
        </a>

        <div className="w-px h-4 bg-border hidden sm:block" />

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
