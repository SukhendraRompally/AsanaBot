import { motion, AnimatePresence } from 'framer-motion';
import { AlertOctagon, CheckCircle, XCircle } from 'lucide-react';
import { useAppStore } from '@/lib/store';
import { useAgent } from '@/hooks/use-agent';
import { Button } from '@/components/ui/button';

export function ConfirmationModal() {
  const { state } = useAppStore();
  const { confirmAction } = useAgent();
  const req = state.pendingConfirmation;

  // Handle keyboard shortcuts when modal is open
  if (req) {
    window.onkeydown = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        confirmAction(false);
      } else if (e.key === 'Enter') {
        e.preventDefault();
        confirmAction(true);
      }
    };
  } else {
    window.onkeydown = null;
  }

  return (
    <AnimatePresence>
      {req && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-background/80 backdrop-blur-sm"
            onClick={() => confirmAction(false)}
          />
          
          <motion.div 
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="relative w-full max-w-lg bg-card border border-destructive/20 shadow-[0_0_40px_rgba(220,38,38,0.1)] rounded-2xl overflow-hidden"
          >
            <div className="bg-destructive/10 border-b border-destructive/20 p-6 flex flex-col items-center text-center">
              <div className="w-16 h-16 bg-destructive/20 rounded-full flex items-center justify-center mb-4 text-destructive shadow-inner">
                <AlertOctagon className="w-8 h-8" />
              </div>
              <h2 className="text-xl font-bold text-foreground tracking-tight">Action Requires Confirmation</h2>
              <p className="text-sm text-muted-foreground mt-1">The agent is about to perform a destructive operation.</p>
            </div>

            <div className="p-6 space-y-6">
              <div className="space-y-4">
                <div>
                  <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-1 block">Action Type</span>
                  <div className="inline-flex items-center px-2.5 py-1 rounded bg-destructive/20 text-destructive text-xs font-mono font-bold">
                    {req.action_type}
                  </div>
                </div>

                <div>
                  <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-1 block">Target Resource</span>
                  <div className="text-lg font-semibold text-foreground">{req.resource}</div>
                  {req.workspace && (
                    <div className="text-xs text-muted-foreground mt-1">Workspace/Project: <span className="text-foreground">{req.workspace}</span></div>
                  )}
                </div>

                <div className="p-4 bg-muted/50 border border-border rounded-lg">
                  <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-2 block">Consequence</span>
                  <p className="text-sm text-foreground/90 font-medium leading-relaxed">{req.consequence}</p>
                </div>
              </div>

              <div className="flex gap-3 pt-2">
                <Button 
                  variant="outline" 
                  className="flex-1 h-12 text-sm font-semibold gap-2 border-border hover:bg-muted"
                  onClick={() => confirmAction(false)}
                >
                  <XCircle className="w-4 h-4" />
                  Cancel (Esc)
                </Button>
                <Button 
                  className="flex-1 h-12 text-sm font-semibold gap-2 bg-destructive hover:bg-destructive/90 text-destructive-foreground shadow-lg shadow-destructive/20"
                  onClick={() => confirmAction(true)}
                >
                  <CheckCircle className="w-4 h-4" />
                  Confirm Action (Enter)
                </Button>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
