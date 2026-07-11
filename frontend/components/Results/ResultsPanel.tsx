"use client";

import { motion, AnimatePresence } from "framer-motion";
import { LayoutDashboard } from "lucide-react";
import { ChatMessage, MapData } from "@/types";
import { SummaryCards } from "./SummaryCards";
import { PlotlyChart } from "./PlotlyChart";
import { FloatDetailCard } from "./FloatDetailCard";

interface ResultsPanelProps {
  lastAssistantMessage?: ChatMessage;
  selectedFloat: string | null;
  mapData: MapData[];
  onClearSelection: () => void;
}

export function ResultsPanel({
  lastAssistantMessage,
  selectedFloat,
  mapData,
  onClearSelection,
}: ResultsPanelProps) {
  const hasResult = lastAssistantMessage && !lastAssistantMessage.isLoading;
  const selectedFloatData = selectedFloat
    ? mapData.find((m) => m.float_id === selectedFloat)
    : undefined;

  return (
    <div className="flex flex-col h-full bg-surface-900/50 border border-surface-800/60 rounded-xl overflow-hidden">
      {/* Panel Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-surface-800/60 bg-surface-900/80">
        <LayoutDashboard className="w-4 h-4 text-ocean-400" />
        <span className="text-sm font-medium text-surface-300">Results</span>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto scrollbar-thin p-4">
        <AnimatePresence mode="wait">
          {!hasResult ? (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center h-full text-surface-600 gap-3"
            >
              <div className="w-12 h-12 rounded-full bg-surface-800 flex items-center justify-center">
                <LayoutDashboard className="w-5 h-5 text-surface-500" />
              </div>
              <p className="text-sm">Results will appear here after you send a query.</p>
              <p className="text-xs text-surface-700">
                Try: &quot;oxygen in arabian sea&quot; or &quot;temperature profile&quot;
              </p>
            </motion.div>
          ) : (
            <motion.div
              key="result"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col gap-4"
            >
              <SummaryCards
                summary={lastAssistantMessage.summary}
                intent={lastAssistantMessage.intent}
              />

              <AnimatePresence>
                {selectedFloatData && (
                  <FloatDetailCard
                    float={selectedFloatData}
                    onClear={onClearSelection}
                  />
                )}
              </AnimatePresence>

              <PlotlyChart
                figure={lastAssistantMessage.figure}
                selectedFloat={selectedFloat}
                onClearSelection={onClearSelection}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
