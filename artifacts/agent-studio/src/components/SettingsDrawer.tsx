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
  const [healthPath, setHealthPath] = useState(state.settings.vmHealthPath);
  const [testing, setTesting] = useState(false);

  useEffect(() => {
    if (state.showSettings) {
      setUrl(state.settings.vmBackendUrl);
      setToken(state.settings.vmBearerToken);
      setHealthPath(state.settings.vmHealthPath);
    }
  }, [state.showSettings, state.settings]);

  const handleSave = () => {
    dispatch({ type: 'UPDATE_SETTINGS', settings: { vmBackendUrl: url, vmBearerToken: token, vmHealthPath: healthPath } });
    dispatch({ type: 'TOGGLE_SETTINGS', show: false });
  };

  const buildHealthUrl = (baseUrl: string, path: string) => {
    const trimmedBase = baseUrl.replace(/\/$/, '');
    const trimmedPath = path.startsWith('/') ? path : `/${path}`;
    return `${trimmedBase}${trimmedPath}`;
  };

  const handleTest = async () => {
    if (!url) return;
    setTesting(true);
    dispatch({ type: 'SET_CONNECTION_STATUS', status: 'checking' });
    try {
      const res = await fetch(buildHealthUrl(url, healthPath), {
        headers: token ? { Authorization: `Bearer ${token}` } : {}
      });
      dispatch({ type: 'SET_CONNECTION_STATUS', status: res.ok ? 'connected' : 'disconnected' });
    } catch {
      dispatch({ type: 'SET_CONNECTION_STATUS', status: 'disconnected' });
    } finally {
      setTesting(false);
    }
  };

  useEffect(() => {
    if (!state.settings.vmBackendUrl) return;

    const check = async () => {
      try {
        const res = await fetch(buildHealthUrl(state.settings.vmBackendUrl, state.settings.vmHealthPath), {
          headers: state.settings.vmBearerToken ? { Authorization: `Bearer ${state.settings.vmBearerToken}` } : {}
        });
        dispatch({ type: 'SET_CONNECTION_STATUS', status: res.ok ? 'connected' : 'disconnected' });
      } catch {
        dispatch({ type: 'SET_CONNECTION_STATUS', status: 'disconnected' });
      }
    };

    check();
    const int = setInterval(check, 30000);
    return () => clearInterval(int);
  }, [state.settings.vmBackendUrl, state.settings.vmBearerToken, state.settings.vmHealthPath, dispatch]);

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
            transition={{ type: 'spring', bounce: 0, duration: 0.4 }}
            className="fixed inset-y-0 right-0 w-96 bg-popover border-l border-border shadow-2xl z-50 flex flex-col"
          >
            <div className="h-14 px-4 flex items-center justify-between border-b border-border shrink-0">
              <h2 className="font-semibold text-foreground">Settings</h2>
              <Button variant="ghost" size="icon" onClick={() => dispatch({ type: 'TOGGLE_SETTINGS', show: false })}>
                <X className="w-4 h-4" />
              </Button>
            </div>

            <div className="p-6 space-y-5 flex-1 overflow-y-auto">
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
                  The URL where your Claude Code agent backend is running. Leave blank for Demo Mode.
                </p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="healthPath">Health Check Path</Label>
                <Input
                  id="healthPath"
                  value={healthPath}
                  onChange={e => setHealthPath(e.target.value)}
                  placeholder="/health"
                  className="font-mono text-xs"
                />
                <p className="text-[10px] text-muted-foreground">
                  Endpoint to ping for connection status. Common values: <span className="font-mono">/health</span>, <span className="font-mono">/api/healthz</span>, <span className="font-mono">/status</span>.
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
                className="w-full gap-2"
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
                  {state.connectionStatus === 'connected' ? 'Connected to health endpoint.' :
                   state.connectionStatus === 'checking' ? 'Testing connection...' :
                   'Failed to reach health endpoint.'}
                </div>
              )}
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
