import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Save, Activity } from 'lucide-react';
import { useAppStore } from '@/lib/store';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

export function SettingsDrawer() {
  const { state, dispatch } = useAppStore();
  const [url, setUrl] = useState(state.settings.vmBackendUrl);
  const [token, setToken] = useState(state.settings.vmBearerToken);
  const [testing, setTesting] = useState(false);

  // Sync internal state when opened
  useEffect(() => {
    if (state.showSettings) {
      setUrl(state.settings.vmBackendUrl);
      setToken(state.settings.vmBearerToken);
    }
  }, [state.showSettings, state.settings]);

  const handleSave = () => {
    dispatch({ type: 'UPDATE_SETTINGS', settings: { vmBackendUrl: url, vmBearerToken: token } });
    dispatch({ type: 'TOGGLE_SETTINGS', show: false });
  };

  const handleTest = async () => {
    if (!url) return;
    setTesting(true);
    dispatch({ type: 'SET_CONNECTION_STATUS', status: 'checking' });
    
    try {
      const res = await fetch(`${url}/health`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {}
      });
      if (res.ok) {
        dispatch({ type: 'SET_CONNECTION_STATUS', status: 'connected' });
      } else {
        dispatch({ type: 'SET_CONNECTION_STATUS', status: 'disconnected' });
      }
    } catch (e) {
      dispatch({ type: 'SET_CONNECTION_STATUS', status: 'disconnected' });
    } finally {
      setTesting(false);
    }
  };

  // Poll health every 30s if we have a URL and drawer is closed
  useEffect(() => {
    if (!state.settings.vmBackendUrl) return;
    
    const check = async () => {
      try {
        const res = await fetch(`${state.settings.vmBackendUrl}/health`, {
          headers: state.settings.vmBearerToken ? { Authorization: `Bearer ${state.settings.vmBearerToken}` } : {}
        });
        dispatch({ type: 'SET_CONNECTION_STATUS', status: res.ok ? 'connected' : 'disconnected' });
      } catch (e) {
        dispatch({ type: 'SET_CONNECTION_STATUS', status: 'disconnected' });
      }
    };
    
    check();
    const int = setInterval(check, 30000);
    return () => clearInterval(int);
  }, [state.settings.vmBackendUrl, state.settings.vmBearerToken, dispatch]);

  return (
    <AnimatePresence>
      {state.showSettings && (
        <>
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-background/50 backdrop-blur-sm z-40"
            onClick={() => dispatch({ type: 'TOGGLE_SETTINGS', show: false })}
          />
          <motion.div 
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: "spring", bounce: 0, duration: 0.4 }}
            className="fixed inset-y-0 right-0 w-96 bg-popover border-l border-border shadow-2xl z-50 flex flex-col"
          >
            <div className="h-14 px-4 flex items-center justify-between border-b border-border shrink-0">
              <h2 className="font-semibold text-foreground">Settings</h2>
              <Button variant="ghost" size="icon" onClick={() => dispatch({ type: 'TOGGLE_SETTINGS', show: false })}>
                <X className="w-4 h-4" />
              </Button>
            </div>

            <div className="p-6 space-y-6 flex-1 overflow-y-auto">
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="vmUrl">VM Backend URL</Label>
                  <Input 
                    id="vmUrl"
                    value={url}
                    onChange={e => setUrl(e.target.value)}
                    placeholder="https://your-vm.example.com"
                    className="font-mono text-xs"
                  />
                  <p className="text-[10px] text-muted-foreground">
                    The URL where your Claude Code agent is running. Leave blank for Demo Mode.
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="token">Bearer Token (Optional)</Label>
                  <Input 
                    id="token"
                    type="password"
                    value={token}
                    onChange={e => setToken(e.target.value)}
                    placeholder="ey..."
                    className="font-mono text-xs"
                  />
                </div>

                <Button 
                  variant="outline" 
                  className="w-full gap-2 mt-2" 
                  onClick={handleTest}
                  disabled={testing || !url}
                >
                  <Activity className={`w-4 h-4 ${testing ? 'animate-spin' : ''}`} />
                  Test Connection
                </Button>
                
                {state.connectionStatus !== 'unknown' && url && (
                  <div className={`text-xs text-center p-2 rounded ${
                    state.connectionStatus === 'connected' ? 'bg-green-500/10 text-green-500' : 
                    state.connectionStatus === 'checking' ? 'bg-yellow-500/10 text-yellow-500' :
                    'bg-red-500/10 text-red-500'
                  }`}>
                    {state.connectionStatus === 'connected' ? 'Successfully connected to VM health endpoint.' :
                     state.connectionStatus === 'checking' ? 'Testing connection...' :
                     'Failed to connect to VM health endpoint.'}
                  </div>
                )}
              </div>
            </div>

            <div className="p-4 border-t border-border shrink-0 bg-muted/20">
              <Button className="w-full gap-2" onClick={handleSave}>
                <Save className="w-4 h-4" />
                Save Settings
              </Button>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
