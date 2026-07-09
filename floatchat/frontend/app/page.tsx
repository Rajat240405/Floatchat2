"use client";

import { useMemo } from "react";
import dynamic from "next/dynamic";
import { MainLayout } from "@/components/Layout/MainLayout";
import { ChatPanel } from "@/components/Chat/ChatPanel";
import { ResultsPanel } from "@/components/Results/ResultsPanel";
import { PromptInput } from "@/components/Input/PromptInput";
import { useChat } from "@/hooks/useChat";

// Leaflet requires `window` — must be client-only
const MapPanel = dynamic(
  () => import("@/components/Map/MapPanel").then((mod) => mod.MapPanel),
  { ssr: false }
);

export default function HomePage() {
  const {
    messages,
    input,
    setInput,
    isLoading,
    sendMessage,
    handleKeyDown,
    messagesEndRef,
    scrollContainerRef,
    onScroll,
    selectedFloat,
    setSelectedFloat,
    currentMapData,
  } = useChat();

  const lastAssistantMessage = useMemo(() => {
    return [...messages].reverse().find((m) => m.role === "assistant");
  }, [messages]);

  return (
    <MainLayout>
      {/* Top Row: Map + Chat */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-5 gap-3 min-h-0">
        {/* Map — 40% on large screens */}
        <div className="lg:col-span-2 min-h-[300px] lg:min-h-0">
          <MapPanel
            mapData={currentMapData}
            selectedFloat={selectedFloat}
            onSelectFloat={setSelectedFloat}
          />
        </div>

        {/* Chat — 60% on large screens */}
        <div className="lg:col-span-3 min-h-[300px] lg:min-h-0">
          <ChatPanel
            messages={messages}
            messagesEndRef={messagesEndRef}
            scrollContainerRef={scrollContainerRef}
            onScroll={onScroll}
          />
        </div>
      </div>

      {/* Middle: Results */}
      <div className="h-[320px] min-h-[320px]">
        <ResultsPanel
          lastAssistantMessage={lastAssistantMessage}
          selectedFloat={selectedFloat}
          mapData={currentMapData}
          onClearSelection={() => setSelectedFloat(null)}
        />
      </div>

      {/* Bottom: Input */}
      <div className="flex-shrink-0">
        <PromptInput
          input={input}
          setInput={setInput}
          isLoading={isLoading}
          onSend={sendMessage}
          onKeyDown={handleKeyDown}
        />
      </div>
    </MainLayout>
  );
}
