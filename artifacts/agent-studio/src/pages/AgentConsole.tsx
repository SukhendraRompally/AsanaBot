import { Header } from '@/components/Header';
import { Sidebar } from '@/components/Sidebar';
import { ChatPanel } from '@/components/ChatPanel';
import { TracePanel } from '@/components/TracePanel';
import { ConfirmationModal } from '@/components/ConfirmationModal';
import { SettingsDrawer } from '@/components/SettingsDrawer';

export default function AgentConsole() {
  return (
    <div className="h-screen w-full flex flex-col overflow-hidden bg-background text-foreground dark">
      <Header />
      <div className="flex-1 flex overflow-hidden relative">
        <Sidebar />
        <ChatPanel />
        <TracePanel />
      </div>
      <ConfirmationModal />
      <SettingsDrawer />
    </div>
  );
}
